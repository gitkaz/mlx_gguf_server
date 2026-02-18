from typing import Union
import multiprocessing
import psutil
import time
import uuid
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from worker.llm_process import start_llm_process
from schemas import CompletionParams, TokenCountParams, ModelLoadParams
import logging

logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)


class LLMProcess:
    def __init__(self):
        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.pending_requests = {} 
        self.listener_task = None

        self.process = None
        self.start_time = time.time()
        self.last_activity_time = time.time()
        self.queues = {}
        self.model_info = {}

    async def start_listener(self):
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        while True:
            try:
                # 1. プロセス間通信のQueueからデータを取り出す
                response = await loop.run_in_executor(
                    executor, self.response_queue.get
                )

                if response is None: break

                response_id, response_data = response

                # 2. 辞書に登録されているか確認
                if response_id in self.pending_requests:
                    target = self.pending_requests[response_id]

                    # --- A. 通常リクエストの場合 (Future) ---
                    if isinstance(target, asyncio.Future):
                        # Futureは1回しかセットできないので、辞書から削除してセットする
                        del self.pending_requests[response_id]
                        if not target.done():
                            target.set_result(response_data)

                    # --- B. ストリーミングの場合 (asyncio.Queue) ---
                    elif isinstance(target, asyncio.Queue):
                        # Queueはずっと使い続けるので、辞書から削除しない（まだ次が来るかも）
                        # 内部Queueにデータを入れる
                        target.put_nowait(response_data)
                
                else:
                    # 既にキャンセルされたか、タイムアウトしたID
                    pass

            except Exception as e:
                print(f"Listener Error: {e}")
                break


    def start(self):
        # ここは親プロセスの「同期的な」起動処理なので、
        # args に渡すQueueが新しくなっていることに注意してください
        self.process = multiprocessing.Process(
            target=self.run_llm_process, 
            args=(self.request_queue, self.response_queue)
        )
        self.process.start()

    def stop(self):
        # 1. stop worker process. (For graceful stop, put "none" to queue)
        self.request_queue.put(None)
        # if worker process still exists, termninate the process by force.
        if self.process:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()

        #2. stop listener
        if self.listener_task:
            self.listener_task.cancel()

        #3. send "none" to response_queue to break start_listener.
        self.response_queue.put(None)


    async def task_request_streaming(self, task: str, params: Union[CompletionParams, TokenCountParams, ModelLoadParams]):
        request_id = str(uuid.uuid4())
        
        # 1. ストリーミング受取用の内部Queueを作成
        stream_queue = asyncio.Queue()
        
        # 2. 辞書に登録 (変数名を pending_requests に統一)
        self.pending_requests[request_id] = stream_queue
        
        # 3. リクエスト送信 (エラーハンドリングを追加)
        try:
            self.request_queue.put((task, request_id, params))
        except Exception as e:
            logger.error(f"Failed to put streaming task to queue. Error: {e}")
            # 送信失敗時は即座に辞書から削除
            del self.pending_requests[request_id]
            raise e
        
        try:
            # 4. ループでデータを取り出し続ける
            while True:
                # ディスパッチャーからデータを受け取る
                raw_data = await stream_queue.get()
                
                # A. データのパース (JSON文字列 -> 辞書)
                if isinstance(raw_data, str):
                    try:
                        response = json.loads(raw_data)
                    except json.JSONDecodeError:
                        yield f"Error: Failed to decode JSON response: {raw_data}"
                        break
                else:
                    response = raw_data

                # B. ステータスコードのチェック
                # 子プロセスで例外が発生した場合 (status 500など) を検知してストップする
                status = response.get("status", 200)
                message = response.get("message")

                if status != 200:
                    # エラーメッセージを返して終了
                    error_msg = f"Worker Error (Status {status}): {message}"
                    logger.error(error_msg)
                    # 必要に応じて例外を投げるか、エラー文字列をyieldするか選択してください
                    # ここでは呼び出し元にエラーを伝えて終了します
                    yield str(message) # または raise Exception(message)
                    break

                # C. 終了判定
                # メッセージ内容に "stream_done" が含まれていたら正常終了
                # (子プロセスの実装依存ですが、現状のルールに従います)
                if message and "stream_done" in str(message):
                    break
                
                # 正常なデータチャンクを返す
                yield message
                
        finally:
            # 5. 【重要】クリーンアップ
            # 正常終了でもエラー終了でも、必ず辞書からIDを削除する
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]


    async def task_request(self, task: str, params: Union[CompletionParams, TokenCountParams, ModelLoadParams]):
        request_id = str(uuid.uuid4())
        
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        # リスナーが正しい宛先を見つけられるように登録
        self.pending_requests[request_id] = future
        
        try:
            self.request_queue.put((task, request_id, params))
        except Exception as e:
            logger.error(f"Failed to put task to queue. Error: {e}")
            del self.pending_requests[request_id]
            raise e
        
        # 1. 子プロセスから送られてきた「JSON文字列」を受け取る
        json_str_response = await future
        
        # 2. 文字列を辞書にパースする
        try:
            response_dict = json.loads(json_str_response)
        except (json.JSONDecodeError, TypeError) as e:
            # 万が一JSONでないものが来た場合のエラーハンドリング
            return 500, f"Internal Error: Failed to decode response. {str(e)}"

        # 3. 呼び出し元 (_load_model) が期待する形式 (status_code, message) に分解する
        status = response_dict.get("status", 500)
        message = response_dict.get("message", "Unknown error")
        
        return status, message


    async def add_request_to_queue(self, task: str, params: Union[CompletionParams, TokenCountParams, ModelLoadParams]) -> str:
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
    def run_llm_process(request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
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
        """
        Return queue information in a JSON-serializable format.
        Does NOT expose internal asyncio.Future or asyncio.Queue objects.
        """
        queue_info = {}
        queue_info["pending_requests_count"] = len(self.pending_requests)

        # Return only request IDs and their types (serializable)
        queue_info["queues"] = {
            request_id: {
                "type": "stream" if isinstance(target, asyncio.Queue) else "request",
                "has_result": target.done() if isinstance(target, asyncio.Future) else None
            }
            for request_id, target in self.pending_requests.items()
        }

        return queue_info


    def get_cpu_usage(self):
        if self.process is None:
            raise psutil.NoSuchProcess(pid=None)
        process = psutil.Process(self.process.pid)
        cpu_usage = process.cpu_percent()
        return cpu_usage



    def get_memory_usage(self):
        if self.process is None:
            raise psutil.NoSuchProcess(pid=None)
        process = psutil.Process(self.process.pid)
        memory_usage = process.memory_info().rss
        return memory_usage