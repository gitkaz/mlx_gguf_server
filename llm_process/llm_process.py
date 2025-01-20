from typing import Union, Any
from multiprocessing import Queue, Manager
import os
from .llm_model import LLMModel
from .task_response import TaskResponse
from .logger_config import setup_logger


async def start_llm_process(request_queue: Queue, response_queue: Queue):
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    logger = setup_logger(__name__, level=log_level)
    logger.info(f"start llm process. log_level={log_level}")

    model = LLMModel()

    while True:
        item = request_queue.get()
        if item is None:
            logger.debug("llm_process receied \"None\". process quitting.")
            break
        else:
            task, request_id, params = item
            logger.debug(f"debug: new request: {task=}, {request_id=}, {params=}")

        if task == 'load':
            result = model.load_model(params)
        elif task == 'token-count':
            result = model.token_count(params)
        elif task == 'get_cache_memory':
            result = model.get_cache_memory()
        elif task == 'set_cache_liimt':
            result = model.set_cache_liimt(params)
        elif task == 'completions_stream':
            for result in model.completions_stream(params):
                assert_task_response(result)
                # logger.debug(f"completions_stream_result: {result.message=}")
                response_queue.put(( request_id, result.to_json() ))
            continue
        assert_task_response(result)
        response_queue.put(( request_id, result.to_json() ))


def assert_task_response(result: Union[TaskResponse, Any]):
    if not isinstance(result, TaskResponse):
        error_message = f"Expected TaskResponse, but got {type(result)}."
        logger.error(error_message)
        raise TypeError(error_message)