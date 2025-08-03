from typing import Union, Optional, Dict
from multiprocessing import Queue, Manager
import psutil
import time
import uuid
from fastapi import HTTPException
import json
import asyncio
from multiprocessing import Process, Queue, Manager

from llm_process.llm_process import start_llm_process
from schemas import CompletionParams, TokenCountParams, ModelLoadParams, ProcessCleanParams, CacheLimitParams, KokoroTtsParams, EmbeddingsParams
import logging

logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)


class LLMProcess:
    def __init__(self):
        self.manager = Manager()
        self.request_queue = self.manager.Queue()
        self.response_queue = self.manager.Queue()
        self.process = None
        self.start_time = time.time()
        self.last_activity_time = time.time()
        self.queues = {}
        self.model_info = {}

    def start(self):
        self.process = Process(target=self.run_llm_process, args=(self.request_queue, self.response_queue))
        self.process.start()

    def stop(self):
        self.request_queue.put(None)
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()

    async def task_request_streaming(self, task: str, params: Union[CompletionParams, TokenCountParams, ModelLoadParams, CacheLimitParams]):
        request_id = await self.add_request_to_queue(task, params)
        while True:
            try:
                response_id, response_message_json = await self.get_response_from_queue()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Unexpected error occurred at task_request_streaming: {str(e)}")

            if response_id == request_id:
                response = json.loads(response_message_json)
                status = response.get("status")
                message = response.get("message")

                if "stream_done" in message:
                    del self.queues[response_id]
                    break
                else:
                    yield message
            else:
                self.push_back_result(response_id, response_message_json)


    async def task_request(self, task: str, params: Union[CompletionParams, TokenCountParams, ModelLoadParams]):
        request_id = await self.add_request_to_queue(task, params)
        while True:
            try:
                response_id, response_message_json = await self.get_response_from_queue()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Unexpected error occurred at task_request: {str(e)}")

            if response_id == request_id:
                response = json.loads(response_message_json, strict=False)
                status = response.get("status")
                message = response.get("message")
                del self.queues[response_id]
                return status, message
            else:
                self.push_back_result(response_id, response_message_json)


    async def add_request_to_queue(self, task: str, params: Union[CompletionParams, TokenCountParams, ModelLoadParams, CacheLimitParams]) -> str:
        request_id = str(uuid.uuid4())
        current_time = time.time()
        await asyncio.get_event_loop().run_in_executor(None, self.request_queue.put, (task, request_id, params))
        self.queues[request_id] = {task: params, "start_time": current_time}
        return request_id
    
    async def get_response_from_queue(self):
        response_id, response_message_json = await asyncio.get_event_loop().run_in_executor(None, self.response_queue.get)
        return response_id, response_message_json

    def push_back_result(self, response_id: str, response_str: str):
        self.response_queue.put((response_id, response_str))

    @staticmethod
    def run_llm_process(request_queue: Queue, response_queue: Queue):
        asyncio.run(start_llm_process(request_queue, response_queue))

    def clean_queues(self, timeout: int):
        current_time = time.time()
        for request_id, task_info in list(self.queues.items()):
            start_time = task_info["start_time"]
            if current_time - start_time > timeout:
                del self.queues[request_id]
                logger.debug(f"Removed expired request: {request_id}")


    def print_queue_contents(self, queue):
        queue_contents = []
        while not queue.empty():
            queue_contents.append(queue.get())

        print(f"{queue_contents}")

        for item in queue_contents:
            queue.put(item)

    def get_queues_info(self):
        queue_info = {}
        queue_info["request_queue_size"]  = self.request_queue.qsize()
        queue_info["response_queue_size"] = self.response_queue.qsize()
        queue_info["queues"] = self.queues
        # for queue debbuging.
        # print("Queue debugging start")
        # self.print_queue_contents(self.response_queue)
        return queue_info

    def get_cpu_usage(self):
        process = psutil.Process(self.process.pid)
        cpu_usage = process.cpu_percent()
        return cpu_usage

    def get_memory_usage(self):
        process = psutil.Process(self.process.pid)
        memory_usage = process.memory_info().rss
        return memory_usage
