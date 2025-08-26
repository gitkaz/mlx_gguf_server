from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Request, File, UploadFile, Form, Depends
from fastapi.responses import Response, StreamingResponse, JSONResponse, FileResponse
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

import argparse
import asyncio
import json
import os
import time
import psutil
import uuid
import uvicorn
from typing import Union, Optional, Dict
from multiprocessing import Process, Queue, Manager

import tts.kokoro_tts.run_process
import embedding.run_process

from core.process_manager import LLMProcess
from schemas import CompletionParams, TokenCountParams, ModelLoadParams, ProcessCleanParams, KokoroTtsParams, EmbeddingsParams
from embedding.embedding_schemas import OpenAICompatibleEmbeddings
from utils.utils import create_model_list
from utils.kv_cache_utils import prepare_temp_dir, validate_session_id, validate_filename, process_upload_file
from whisper_stt.whisper import AudioTranscriber
# from logging import getLogger
import logging

logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting....")

    # ===== 创建日志目录并设置记录器 =====
    os.makedirs("logs", exist_ok=True)

    # 设置专用的访问日志记录器
    access_logger = logging.getLogger("access")
    access_logger.setLevel(logging.INFO)

    # 定义 JSON 格式化器（如需自定义可在此处修改）
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
                "ip": getattr(record, 'ip', ''),
                "method": getattr(record, 'method', ''),
                "path": getattr(record, 'path', ''),
                "status": getattr(record, 'status', ''),
                "req_size": getattr(record, 'req_size', 0),
                "duration_ms": int(getattr(record, 'duration', 0) * 1000),
                "params": getattr(record, 'params', {})
            }
            return json.dumps(log_data, ensure_ascii=False)

    # 设置文件处理器
    file_handler = logging.FileHandler("logs/access.log")
    file_handler.setFormatter(JsonFormatter())
    access_logger.addHandler(file_handler)
    # =====================================================

    app.state.tmpdir = "temp"
    app.state.models = create_model_list()
    app.state.llm_processes = {}
    init_task_scheduler()
    yield
    logger.info("stopping....")

app = FastAPI(lifespan=lifespan)


def init_task_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(recurring_task_cleanup, 'interval', minutes=5)
    scheduler.start()        


def recurring_task_cleanup():
    logger.debug("Under Construction...")
    pass

@app.middleware("http")
async def access_logger(request: Request, call_next):
    # 收集请求信息
    start_time = time.time()
    body = await request.body()
    request._body = body  # 可重复使用

    # 执行处理
    response = await call_next(request)

    # 输出日志
    duration = time.time() - start_time

    # 安全收集参数（排除 messages 字段）
    try:
        body_json = json.loads(body)
    # 完全删除 messages 字段（比 [REDACTED] 更安全）
        if "messages" in body_json:
            body_json["messages"] = "[REDACTED]"
        elif "prompt" in body_json:
            body_json["prompt"] = "[REDACTED]"
    # 如果参数过大则仅记录前三项
        safe_params = dict(list(body_json.items())[:3]) if len(body_json) > 3 else body_json
    except:
        safe_params = {"raw_body_size": len(body)}

    logger = logging.getLogger("access")
    logger.info("", extra={
        "ip": request.client.host,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "req_size": len(body),
        "duration": duration,
        "params": safe_params
    })

    return response

def get_llm_process(model_id: str) -> LLMProcess:
    if model_id in app.state.llm_processes:
        llm_process: LLMProcess = app.state.llm_processes[model_id]
        return llm_process
    else:
        raise HTTPException(status_code=400, detail=f"No LLM process found for model_id: {model_id}")

def terminate_llm_process(model_id: str):
    llm_process = app.state.llm_processes.get(model_id)
    if llm_process:
        llm_process.stop()
        del app.state.llm_processes[model_id]
        return True, ""
    else:
        error_message = f"Terminate process failed. model_id = {model_id} not found."
        return False, error_message
    

def create_llm_process(model_id: str) -> LLMProcess:
    llm_process = LLMProcess()
    app.state.llm_processes[model_id] = llm_process
    llm_process.start()
    return llm_process

@app.get("/management/processes")
async def get_management_processes():
    processes_info = {}
    for model_id, llm_process in app.state.llm_processes.items():
        process_info = {
            "model_id": model_id,
            "model_name": llm_process.model_info["model_name"],
            "model_path": llm_process.model_info["model_path"],
            "model_type": llm_process.model_info["model_type"],
            "context_length": llm_process.model_info["context_length"],
            "process_id": llm_process.process.pid,
            "cpu_usage": llm_process.get_cpu_usage(),
            "memory_usage": llm_process.get_memory_usage(),
            "current_queue": llm_process.get_queues_info()
        }
        processes_info[model_id] = process_info
    return {"processes": processes_info}

@app.post("/management/process/clean-up")
async def post_process_clean_up(params: ProcessCleanParams, model_id: str = Header(default="0", alias="X-Model-Id")):
    llm_process: LLMProcess = get_llm_process(model_id)
    llm_process.clean_queues(params.timeout)
    return {"process_clean_up": "success"}


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def post_completion(request:Request, params: CompletionParams, model_id: str = Header(default="0", alias="X-Model-Id")):
    llm_process: LLMProcess = get_llm_process(model_id)

    if request.url.path == "/v1/completions" and params.prompt == "":
        raise HTTPException(status_code=400, detail="/v1/completions needs prompt string")
    if request.url.path == "/v1/chat/completions" and params.messages == []:
        raise HTTPException(status_code=400, detail="/v1/chat/completions needs messages list")

    if request.url.path == "/v1/chat/completions":
        params.apply_chat_template = True

    if params.stream:
        async def stream_completion():
            async for chunk in llm_process.task_request_streaming('completions_stream', params):
                yield f"data: {json.dumps(chunk)}\n\n"
        return StreamingResponse(stream_completion(), media_type="text/event-stream")
    else:
        status_code, response = await llm_process.task_request('completions_stream', params)
        if status_code == 200:
            return response
        else: 
            raise HTTPException(status_code=status_code, detail=response)


@app.post("/v1/internal/token-count")
async def post_token_count(params: TokenCountParams, model_id: str = Header(default="0", alias="X-Model-Id")):
    llm_process: LLMProcess = get_llm_process(model_id)

    status_code, response = await llm_process.task_request('token-count', params)
    if status_code == 200:
        return {'length': response}
    else: 
        raise HTTPException(status_code=status_code, detail=response)


@app.get("/v1/internal/model/info")
async def get_v1_internal_model_info(model_id: str = Header(default="0", alias="X-Model-Id")):
    llm_process: LLMProcess = get_llm_process(model_id)
    model_name = llm_process.model_info["model_name"]
    model_type = llm_process.model_info["model_type"]
    return {"model_name": model_name, "model_type": model_type}


@app.get("/v1/internal/model/list")
def get_v1_internal_model_list():
    app.state.models = create_model_list()
    model_names = list(app.state.models.keys())
    model_names = sorted(model_names)
    return {"model_names": model_names}


@app.post("/v1/internal/model/unload")
def post_model_unload(model_id: str = Header(default="0", alias="X-Model-Id")):
    llm_process: LLMProcess = get_llm_process(model_id)
    success, message =  terminate_llm_process(model_id)
    if success:
        return {"unload": "success"}
    else:
        raise HTTPException(status_code=500, detail=f"Unload model failed. {message=}")


@app.post("/v1/internal/model/load")
def post_model_load(params: ModelLoadParams, model_id: str = Header(default="0", alias="X-Model-Id")):
    model_name = params.llm_model_name
 
    if model_name not in app.state.models:
        raise HTTPException(status_code=400, detail=f"Error: {model_name} not found.")
    if app.state.llm_processes.get(model_id):
        raise HTTPException(status_code=400, detail=f"Error: model_id {model_id} already exists. Unload first.")

    params.llm_model_path = app.state.models[model_name]['path']

    llm_process: LLMProcess = create_llm_process(model_id)

    status_code, response = asyncio.run(llm_process.task_request('load', params))
    if status_code == 200:
        llm_process.model_info = response
        return {"load": "success"}
    else: 
        success, message = terminate_llm_process(model_id)
        if not success:
            raise HTTPException(status_code=500, detail=f"Unload model failed. {message=}")
        raise HTTPException(status_code=status_code, detail=response)
    

@app.post("/kokoro/generate")
async def post_kokoro_generate(params: KokoroTtsParams):

    if not app.state.enable_kokoro_tts:
        raise HTTPException(status_code=400, detail="Kokoro TTS is not enabled")

    try:
        queue = Queue()
        process = Process(target=tts.kokoro_tts.run_process.run, args=(params.model_dump(), queue))
        process.start()
        
    audio_data = queue.get() # 从队列获取结果
        process.join()

        if isinstance(audio_data,str):
          try:
              # 如果是 json 字符串则解析
            audio_data = json.loads(audio_data)
            if "error" in audio_data:
                  raise HTTPException(status_code=500, detail=f"Error at kokoro-TTS: {audio_data['error']}")
          except Exception: # 不是 json 的情况则原样处理
              pass

        if isinstance(audio_data,dict):
            raise HTTPException(status_code=500, detail=f"Error at kokoro-TTS: {audio_data}")

    # 以字节序列返回
        logger.debug(f"debug: kokoro response: {audio_data[:100] if isinstance(audio_data, bytes) else str(audio_data)[:100]} ...")
        return Response(
            content=audio_data,
            media_type="audio/mpeg"  # 如果以 mp3 格式返回
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error at kokoro-TTS: {str(e)}")

@app.get("/v1/audio/transcriptions")
async def get_audio_transcribe_status():
    """
    Check if the audio transcription endpoint is enabled.
    """
    if app.state.enable_whisper:
        return JSONResponse(content={"status": "ready"}, status_code=200)
    else:
        raise HTTPException(status_code=400, detail="Whisper transcription is not enabled")

@app.post("/v1/audio/transcriptions")
async def post_audio_transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):

    if not app.state.enable_whisper:
        raise HTTPException(status_code=400, detail="Whisper transcription is not enabled")
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', 'webm')):
        raise HTTPException(status_code=400, detail="Only WAV, MP3, M4A or WEBM files are allowed")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    new_filename = f"audio_{timestamp}{file_extension}"
    file_path = os.path.join("whisper_stt/uploads", new_filename)
        
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    transcriber = AudioTranscriber(model_path=app.state.whisper_model, file_path=file_path)
    result = transcriber.transcribe(language=language)
    transcriber.delete_file()

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return JSONResponse(content={"filename": new_filename, "text": result["text"]}, status_code=200)


@app.post("/v1/embeddings", response_model=OpenAICompatibleEmbeddings)
async def post_embeddings(params: EmbeddingsParams):
    try:
        queue = Queue()
        process = Process(target=embedding.run_process.run, args=(params.model_dump(), queue))
        process.start()

        embeddings = queue.get()
        process.join()

        if isinstance(embeddings, OpenAICompatibleEmbeddings):
            return embeddings
        elif isinstance(embeddings, Exception):
            raise HTTPException(status_code=500, detail=str(embeddings))
        else:
            raise HTTPException(status_code=500, detail="Error at embedding: Unexpected data type")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error at embedding: {str(e)}")

@app.get("/v1/internal/model/kv_cache/export/{session_id}")
async def export_kv_cache(session_id: str):
    """
    Export a KV Cache file with optional compression.
    """
    # Validate Session ID is UUIDv4
    validate_session_id(session_id=session_id)

    cache_dir = "llm_process/kv_cache/"
    file_path = os.path.join(cache_dir, f"{session_id}.safetensors")

    if not os.path.exists(file_path):
        raise HTTPException(404, "Cache file not found")

    response_path = file_path
    media_type = "application/octet-stream"
    filename = f"{session_id}.safetensors"

    return FileResponse(
        path=response_path,
        media_type=media_type,
        filename=filename
    )


@app.post("/v1/internal/model/kv_cache/import/{session_id}")
async def import_kv_cache(session_id: str, file: UploadFile):
    """
    Import a KV Cache file for a specific session with strict validation.
    Uploaded KV Cache file must be safetensors file or gz compressed safetensors.
    Uploaded KV Cache filename must be <UUIDv4>.safetensors or <UUIDv4>.safetensors.gz. This UUID is used as session_id
    """
    try:
        # Validate inputs
        validate_session_id(session_id=session_id)
        validate_filename(session_id=session_id, filename=file.filename)

        # Create temporary directory inside llm_process/kv_cache/tmp/
        temp_dir = prepare_temp_dir(tmp_base_dir=app.state.tmpdir)
        upload_file_path = os.path.join(temp_dir, file.filename)

        # process uploaded file
        upload_file_path = await process_upload_file(file=file, file_path=upload_file_path)

        # Save to target location
        cache_dir = "llm_process/kv_cache/"
        target_path = os.path.join(cache_dir, f"{session_id}.safetensors")
        os.rename(upload_file_path, target_path)

        return {"status": "success", "message": f"Cache {session_id} imported"}

    except Exception as e:
        # Cleanup temp files on error
        if os.path.exists(upload_file_path):
            os.remove(upload_file_path)
        if os.path.exists(upload_file_path):
            os.remove(upload_file_path)
        os.rmdir(temp_dir)
        raise HTTPException(500, str(e)) from e

    finally:
        # Ensure temp directory is cleaned up
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAPI server arguments')
    parser.add_argument('-a', '--addr', type=str, default='127.0.0.1', help='ip addr to listen (default: 127.0.0.1)')
    parser.add_argument('-p', '--port', type=int, default=4000, help='port number to listen (default: 4000)')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help='Log level')
    parser.add_argument('--max-kv-size', type=int, default=10, help='max kv cache files size (GB)')
    parser.add_argument('--whisper-model', type=str, help='HuggingFace path or local filepath to Whisper model', default=None)
    parser.add_argument('--kokoro-config', type=str, help='Kokoro-82M config path', default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    if args.whisper_model:
        app.state.whisper_model = args.whisper_model
        app.state.enable_whisper = True
    else:
        app.state.enable_whisper = False

    if args.kokoro_config:
        os.environ["KOKORO_CONFIG"] = str(args.kokoro_config)
        app.state.enable_kokoro_tts = True
    else:
        app.state.enable_kokoro_tts = False

    os.environ["MAX_KV_SIZE_GB"] = str(args.max_kv_size)
    os.environ["LOG_LEVEL"]      = args.log_level.upper()

    uvicorn.run(app, host=args.addr, port=args.port, access_log=True, log_level=args.log_level)
