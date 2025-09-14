from typing import Union, Any
from multiprocessing import Queue
import os
from pathlib import Path
from mlx_lm.models.cache import load_prompt_cache

from .task_response import TaskResponse
from .llm_model import LLMModel
from .model_loader import ModelLoader
from .tokenizer_service import TokenizerService
from .generation_service import GenerationService
from .kv_cache_metadata import KVCacheMetadataStore
from .kv_cache_manager import KVCacheManager

from .logger_config import setup_logger
log_level = os.environ.get("LOG_LEVEL", "INFO")
logger = setup_logger(__name__, level=log_level)

async def start_llm_process(request_queue: Queue, response_queue: Queue):
    logger.info(f"start llm process. log_level={log_level}")
    model = LLMModel()

    # Create PERSISTENT metadata store (lives for process lifetime)
    metadata_store = KVCacheMetadataStore()
    for kv_cache_path in Path(KVCacheManager.get_cache_dir()).glob("*.safetensors"):
        try:
            _, metadata = load_prompt_cache(str(kv_cache_path), return_metadata=True)
            metadata_store.add(str(kv_cache_path), metadata)
        except:
            pass

    while True:
        item = request_queue.get()
        if item is None:
            logger.debug("llm_process received \"None\". process quitting.")
            break
        else:
            task, request_id, params = item
            logger.debug(f"debug: new request: {task=}, {request_id=}, {params=}")
        try:
            if task == 'load':
                loader = ModelLoader()
                result = loader.load(model, params)

            elif task == 'token-count':
                tokenizer_service = TokenizerService()
                result = tokenizer_service.count_tokens(model, params)

            elif task == 'completions_stream':
                generator = GenerationService(model, params, cache_manager=KVCacheManager(metadata_store))
                for response in generator.generate_completion():
                    task_response = TaskResponse.create(200, response)
                    response_queue.put((request_id, task_response.to_json()))
                continue  # ストリーミングのためcontinue

            assert_task_response(result)
            response_queue.put((request_id, result.to_json()))

        except Exception as e:
            error_response = TaskResponse.create(500, {"error": str(e)})
            response_queue.put((request_id, error_response.to_json()))


def assert_task_response(result: Union[TaskResponse, Any]):
    if not isinstance(result, TaskResponse):
        error_message = f"Expected TaskResponse, but got {type(result)}."
        logger.error(error_message)
        raise TypeError(error_message)