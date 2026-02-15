from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import Response, StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import uuid
import uvicorn
from typing import Optional
from multiprocessing import Process, Queue

import tts.kokoro_tts.run_process
import embedding.run_process

from core.process_manager import LLMProcess
from schemas import CompletionParams, TokenCountParams, ModelLoadParams, ProcessCleanParams, KokoroTtsParams, EmbeddingsParams
from embedding.embedding_schemas import OpenAICompatibleEmbeddings
from utils.utils import create_model_list, create_adapter_list, get_model_size, get_unload_candidates
from utils.kv_cache_utils import prepare_temp_dir, validate_filename, process_upload_file, get_kv_cache_abs_path
from whisper_stt.whisper import AudioTranscriber
# from logging import getLogger
import logging

logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting....")

    # ===== ログディレクトリの作成とロガー設定 =====
    os.makedirs("logs", exist_ok=True)

    # アクセスログ専用ロガーの設定
    access_logger = logging.getLogger("access")
    access_logger.setLevel(logging.INFO)

    # JSONフォーマッタの定義（別途定義が必要な場合はここに記述）
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

    # ファイルハンドラーの設定
    file_handler = logging.FileHandler("logs/access.log")
    file_handler.setFormatter(JsonFormatter())
    access_logger.addHandler(file_handler)
    # =====================================================

    # set Env-based params
    app.state.max_memory_gb = int(os.environ.get("MAX_MEMORY_GB", "0"))
    
    app.state.models = create_model_list()
    app.state.loaded_models = {}

    # Initialize adapters list
    try:
        app.state.adapters = create_adapter_list()
    except Exception:
        app.state.adapters = {}

    app.state.llm_processes = {}
    init_task_scheduler()
    yield
    logger.info("stopping....")

app = FastAPI(lifespan=lifespan)
app.mount("/webui/static", StaticFiles(directory="webui/static"), name="static")
templates = Jinja2Templates(directory="webui/templates")

def init_task_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(recurring_task_cleanup, 'interval', minutes=5)
    logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)
    scheduler.start()


def recurring_task_cleanup():
    logger.debug("Under Construction...")
    pass

@app.middleware("http")
async def access_logger(request: Request, call_next):
    # リクエスト情報収集
    start_time = time.time()
    body = await request.body()
    request._body = body  # 再利用可能に

    # 処理実行
    response = await call_next(request)

    # ログ出力
    duration = time.time() - start_time

    # パラメータを安全に収集（messages 除外）
    try:
        body_json = json.loads(body)
        # messages フィールドを完全削除（[REDACTED] よりも安全）
        if "messages" in body_json:
            body_json["messages"] = "[REDACTED]"
        elif "prompt" in body_json:
            body_json["prompt"] = "[REDACTED]"
        # 大きすぎる場合は先頭3項目のみ記録
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

def _get_next_auto_load_id():
    """Get next available ID in the auto-load range (e.g., 100-199)"""
    base = 100
    max_auto_loads = 100

    for i in range(max_auto_loads):
        candidate_id = str(base + i)
        if candidate_id not in app.state.llm_processes:
            return candidate_id

    raise RuntimeError("No available model IDs in auto-load range")

async def _load_model(params: ModelLoadParams, model_id: str) -> tuple[bool, str]:
    """
    Internal function to load a model (used by both API and auto-load)

    Args:
        params: Complete ModelLoadParams object with all configuration
        model_id: The model ID to use for the process

    Returns:
        (success, message) tuple
    """
    model_name = params.llm_model_name

    # Validation checks
    if model_name not in app.state.models:
        return False, f"Error: {model_name} not found."
    if model_name in app.state.loaded_models:
        return False, f"Error: {model_name} already loaded at {app.state.loaded_models[model_name]}."
    if app.state.llm_processes.get(model_id):
        return False, f"Error: model_id {model_id} already exists. Unload first."

    # Set default values if not provided
    if params.auto_unload is None:
        params.auto_unload = True
    if params.priority is None:
        params.priority = 0

    # Get model path
    params.llm_model_path = app.state.models[model_name]['path']
    required_memory = get_model_size(params.llm_model_path)

    # Auto unload handling if enabled
    if app.state.max_memory_gb > 0:
        # Collect memory requirements
        max_memory_gb = app.state.max_memory_gb
        logger.info(f"{required_memory=}, {max_memory_gb=}")

        # Check if requested model alone exceeds memory limit
        if required_memory > max_memory_gb:
            return False, f"Error: The requested model exceeds the max memory size."

        # Calculate current memory usage
        current_usage = 0
        for name, model_info in app.state.loaded_models.items():
            if isinstance(model_info, dict):
                model_path = app.state.models.get(name, {}).get('path', '')
                if model_path:
                    current_usage += get_model_size(model_path)
        total_required_memory = current_usage + required_memory

        # Unload candidates if needed
        if total_required_memory > max_memory_gb:
            unload_candidates = get_unload_candidates(app.state.loaded_models)
            while total_required_memory > max_memory_gb and unload_candidates:
                candidate_name, candidate = unload_candidates.pop(0)
                logger.info(f"Attempting to unload {candidate_name} to free memory")

                success, message = _unload_model(candidate["id"])
                if success:
                    # Recalculate memory usage
                    current_usage = sum(
                        entry["memory_usage"] 
                        for entry in app.state.loaded_models.values() 
                        if isinstance(entry, dict)
                    )
                    total_required_memory = current_usage + required_memory
                    logger.info(f"Memory freed. New total: {total_required_memory} GB")
                else:
                    logger.warning(f"Failed to unload {candidate_name}: {message}")

    # Adapter handling
    adapter_name = getattr(params, "adapter_name", None)
    # using empty str to disable adapter
    if isinstance(adapter_name, str):
        adapter_name = adapter_name.strip() or None
    if adapter_name:
        adapters = getattr(app.state, "adapters", {}) or {}
        if adapter_name not in adapters:
            return False, f"Error: adapter {adapter_name} not found."
        setattr(params, "adapter_path", adapters[adapter_name]["path"])
    else:
        if getattr(params, "adapter_path", None):
            return False, "Direct adapter_path is not allowed. Use adapter_name."

    # Create and start process
    llm_process: LLMProcess = create_llm_process(model_id)
    # Start background listener
    llm_process.listener_task = asyncio.create_task(llm_process.start_listener())


    try:
        # Pass the complete params object to the worker process
        # status_code, response = await asyncio.run(llm_process.task_request('load', params))
        status_code, response = await llm_process.task_request('load', params)
        if status_code == 200:
            llm_process.model_info = response
            app.state.loaded_models[model_name] = {
                "id": model_id,
                "memory_usage": required_memory,
                "auto_unload": params.auto_unload,
                "trust_remote_code": params.trust_remote_code if params.trust_remote_code is not None else False,
                "priority": params.priority,
                "last_accessed": time.time()
            }
            return True, "success"
        else:
            # Clean up on failure
            success, message = terminate_llm_process(model_id)
            if not success:
                logger.error(f"Failed to clean up after failed load: {message}")
            return False, response
    except Exception as e:
        # Additional cleanup if needed
        if model_id in app.state.llm_processes:
            terminate_llm_process(model_id)
        return False, str(e)

def _unload_model(model_id: str) -> tuple[bool, str]:
    """Internal function to unload a model by ID (used by both API and auto-unload)"""
    try:
        # First terminate the process
        success, message = terminate_llm_process(model_id)
        if not success:
            return False, message

        # Then find and remove from loaded_models
        model_name_to_remove = None
        for name, model_info in app.state.loaded_models.items():
            if isinstance(model_info, dict) and model_info["id"] == model_id:
                model_name_to_remove = name
                break

        if model_name_to_remove:
            del app.state.loaded_models[model_name_to_remove]
            logger.info(f"Successfully removed {model_name_to_remove} from loaded models")
        else:
            logger.warning(f"Model ID {model_id} not found in loaded_models")

        return True, "success"

    except Exception as e:
        logger.error(f"Error in _unload_model: {str(e)}", exc_info=True)
        return False, str(e)


def create_llm_process(model_id: str) -> LLMProcess:
    llm_process = LLMProcess()
    app.state.llm_processes[model_id] = llm_process
    llm_process.start()
    return llm_process

@app.get("/webui", response_class=HTMLResponse)
async def webui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/management/processes")
async def get_management_processes():
    processes_info = {}
    for model_id, llm_process in app.state.llm_processes.items():
        model_name = llm_process.model_info["model_name"]
        model_entry = app.state.loaded_models.get(model_name, {})
        process_info = {
            "model_id": model_id,
            "model_name": llm_process.model_info["model_name"],
            "model_path": llm_process.model_info["model_path"],
            "model_type": llm_process.model_info["model_type"],
            "context_length": llm_process.model_info["context_length"],
            "process_id": llm_process.process.pid,
            "cpu_usage": llm_process.get_cpu_usage(),
            "memory_usage": llm_process.get_memory_usage(),
            "current_queue": llm_process.get_queues_info(),
            "auto_unload": app.state.loaded_models.get(llm_process.model_info["model_name"], {}).get("auto_unload", True),
            "priority": app.state.loaded_models.get(llm_process.model_info["model_name"], {}).get("priority", 0),
            "trust_remote_code": model_entry.get("trust_remote_code", False) 
        }
        processes_info[model_id] = process_info
    return {"processes": processes_info}

@app.post("/management/process/clean-up")
async def post_process_clean_up(params: ProcessCleanParams, model_id: str = Header(default="0", alias="X-Model-Id")):
    llm_process: LLMProcess = get_llm_process(model_id)
    llm_process.clean_queues(params.timeout)
    return {"process_clean_up": "success"}

@app.get("/management/debug/latest")
async def get_debug_info(model_id: str = Header(default="0", alias="X-Model-Id")):
    llm_process: LLMProcess = get_llm_process(model_id)
    status_code, response = await llm_process.task_request('debug_latest', params=None)
    if status_code == 200:
        return response
    else: 
        raise HTTPException(status_code=status_code, detail=response)


@app.get("/v1/models")
# OpenAI API compatible API endpoint.
# Return only loaded models.
# https://platform.openai.com/docs/api-reference/models/list
def get_v1_models():
    model_names = list(app.state.loaded_models.keys())
    model_names = sorted(model_names)
    data = []
    for model_name in model_names:
        data.append({"id": model_name, "object": "model", "owned_by": "organization-owner"})
    return {"object": "list", "data": data}


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def post_completion(request:Request, params: CompletionParams, model_id: str = Header(default="0", alias="X-Model-Id")):

    if request.url.path == "/v1/completions" and params.prompt == "":
        raise HTTPException(status_code=400, detail="/v1/completions needs prompt string")
    if request.url.path == "/v1/chat/completions" and params.messages == []:
        raise HTTPException(status_code=400, detail="/v1/chat/completions needs messages list")

    if request.url.path == "/v1/chat/completions":
        params.apply_chat_template = True

    if "X-Model-Id" not in request.headers and hasattr(params, "model") and params.model:
        if params.model in app.state.loaded_models:
            model_id = app.state.loaded_models[params.model]['id']
        else:
            if app.state.auto_load_enabled is False:
                raise HTTPException(status_code=400, detail=f"Model '{params.model}' not exist.")

    # Auto-load logic
    requested_model = None
    if hasattr(params, "model") and params.model:
        requested_model = params.model

    if requested_model and requested_model not in app.state.loaded_models:
        if not app.state.auto_load_enabled:
            raise HTTPException(status_code=400, detail=f"Model '{requested_model}' not loaded. Auto-load is disabled.")

        # Find an available model ID
        auto_load_id = _get_next_auto_load_id()

        # Create ModelLoadParams with auto-load defaults
        auto_load_params = ModelLoadParams(
            llm_model_name=requested_model,
            auto_unload=True,         # Always auto-unload for auto-loaded models
            priority=0,               # Default priority
            use_kv_cache=True,        # Enable KV cache by default
            trust_remote_code=False,  # Auto-loaded models should not trust remote code by default
            # Copy other parameters from request if needed
            temperature=params.temperature if hasattr(params, "temperature") else None,
            max_tokens=params.max_tokens if hasattr(params, "max_tokens") else None
        )

        # Load the model
        success, message = await _load_model(auto_load_params, auto_load_id)

        if not success:
            logger.error(f"Auto-load failed for {requested_model}: {message}")
            raise HTTPException(status_code=500, detail=f"Failed to auto-load model '{requested_model}'")

        # Update model_id to use the newly loaded model
        model_id = auto_load_id

    llm_process: LLMProcess = get_llm_process(model_id)

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
    success, message = _unload_model(model_id)
    if success:
        return {"unload": "success"}
    else:
        raise HTTPException(status_code=500, detail=f"Unload model failed: {message}")

@app.post("/v1/internal/model/load")
async def post_model_load(params: ModelLoadParams, model_id: str = Header(default="0", alias="X-Model-Id")):
    success, message = await _load_model(params, model_id)

    if success:
        return {"load": "success"}
    else:
        raise HTTPException(status_code=500, detail=f"Load model failed: {message}") 

@app.post("/kokoro/generate")
async def post_kokoro_generate(params: KokoroTtsParams):

    if not app.state.enable_kokoro_tts:
        raise HTTPException(status_code=400, detail="Kokoro TTS is not enabled")

    try:
        queue = Queue()
        process = Process(target=tts.kokoro_tts.run_process.run, args=(params.model_dump(), queue))
        process.start()
        
        audio_data = queue.get() # 結果をqueueから取得
        process.join()

        if isinstance(audio_data,str):
          try:
              # json文字列だった場合はパース
            audio_data = json.loads(audio_data)
            if "error" in audio_data:
                  raise HTTPException(status_code=500, detail=f"Error at kokoro-TTS: {audio_data['error']}")
          except Exception: # jsonじゃない場合はそのまま
              pass

        if isinstance(audio_data,dict):
            raise HTTPException(status_code=500, detail=f"Error at kokoro-TTS: {audio_data}")

        # バイト列で返す
        logger.debug(f"debug: kokoro response: {audio_data[:100] if isinstance(audio_data, bytes) else str(audio_data)[:100]} ...")
        return Response(
            content=audio_data,
            media_type="audio/mpeg"  # mp3形式で返す場合
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
    filename = getattr(file, 'filename', '') or ''
    if not filename.lower().endswith(('.wav', '.mp3', '.m4a', 'webm')):
        raise HTTPException(status_code=400, detail="Only WAV, MP3, M4A or WEBM files are allowed")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(filename)[1]
    new_filename = f"audio_{timestamp}{file_extension}"
    file_path = os.path.join("whisper_stt", "uploads", new_filename)
        
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

@app.get("/v1/internal/model/kv_cache/export/{kvcache_id}")
async def export_kv_cache(kvcache_id: str):
    """
    Export a KV Cache file with optional compression.
    """
    cache_dir = Path(os.environ["KV_CACHE_PATH"])
    file_path = cache_dir.joinpath(f"{kvcache_id}.safetensors")

    if not file_path.exists():
        raise HTTPException(404, "Cache file not found")

    response_path = file_path
    media_type = "application/octet-stream"
    filename = f"{kvcache_id}.safetensors"

    return FileResponse(
        path=response_path,
        media_type=media_type,
        filename=filename
    )


@app.post("/v1/internal/model/kv_cache/import")
async def import_kv_cache(file: UploadFile):
    """
    Import a KV Cache file.
    KV Cache file must be safetensors file or gz compressed safetensors.
    """
    upload_file_path = Path()
    temp_dir = Path()

    try:
        # Validate inputs
        filename = getattr(file, 'filename', '') or ''
        validate_filename(filename=filename)

        # Create temporary directory inside llm_process/kv_cache/tmp/
        temp_dir: Path = prepare_temp_dir()
        temporary_upload_file_path = temp_dir.joinpath(filename)

        # process uploaded file
        temporary_upload_file_path = await process_upload_file(file=file, file_path=temporary_upload_file_path)

        # Save to target location
        cache_dir = Path(os.environ["KV_CACHE_PATH"])
        cache_id = str(uuid.uuid4())
        target_path = cache_dir.joinpath(f"{cache_id}.safetensors")
        temporary_upload_file_path.rename(target_path)
        target_path.chmod(0o444)

        return {"status": "success", "message": f"Cache file imported", "cache_id": cache_id}

    except HTTPException:
        # Re-raise HTTPException without wrapping
        raise
    except Exception as e:
        # Clean up temporary files on error
        if upload_file_path.exists():
            try:
                upload_file_path.unlink()
            except OSError:
                pass
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                pass
        raise HTTPException(500, str(e)) from e

    finally:
        # Ensure temporary directory is removed
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                pass

@app.get("/v1/internal/adapter/list")
def get_v1_internal_adapter_list():
    """Return the scanned adapter list (only name and size; do not expose filesystem paths)."""
    adapters = getattr(app.state, "adapters", {}) or {}
    adapter_entries = list(app.state.adapters.values())
    adapter_entries = sorted(adapter_entries, key=lambda x: x["name"])
    return {"adapters": adapter_entries}


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAPI server arguments')
    parser.add_argument('-a', '--addr', type=str, default='127.0.0.1', help='ip addr to listen (default: 127.0.0.1)')
    parser.add_argument('-p', '--port', type=int, default=4000, help='port number to listen (default: 4000)')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help='Log level')
    parser.add_argument('--max-kv-size', type=int, default=10, help='max kv cache files size (GB)')
    parser.add_argument('--whisper-model', type=str, help='HuggingFace path or local filepath to Whisper model', default=None)
    parser.add_argument('--kokoro-config', type=str, help='Kokoro-82M config path', default=None)
    parser.add_argument('--max-memory-gb', type=int, default=0, help='Specify the memory size in GB. If total loaded model size exceeded this size, automactically unload the current largest model. If not speficy this argucment, auto unload will not be triggered.')
    parser.add_argument('--enable-auto-load', action='store_true', help='Enable automatic loading model. (JIT Loading)')
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

    if args.enable_auto_load:
        if args.max_memory_gb <= 0:
            raise argparse.ArgumentTypeError(
                f"max-memory-gb must be set for enables auto-load."
            )
        app.state.auto_load_enabled = True

    if args.max_memory_gb < 0:
        raise argparse.ArgumentTypeError(
            f"max-memory-gb must be at least 1 (current: {args.max_memory_gb})"
        )

    os.environ["MAX_MEMORY_GB"] = str(args.max_memory_gb)

    os.environ["MAX_KV_SIZE_GB"] = str(args.max_kv_size)
    os.environ["LOG_LEVEL"]      = args.log_level.upper()
    os.environ["KV_CACHE_PATH"]  = get_kv_cache_abs_path()

    uvicorn.run(app, host=args.addr, port=args.port, access_log=True, log_level=args.log_level)
