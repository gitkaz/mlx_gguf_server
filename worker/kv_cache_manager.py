import os
import time
from typing import Dict, Any, List
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
from .logger_config import setup_logger
logger = setup_logger(__name__, level="DEBUG")

KV_CACHE_DIR = "worker/kv_cache"

def load_kv_cache(model, messages: List):
    metadata = {}
    stats = {}
    for reversed_index, message in enumerate(reversed(messages)):
        expect_file_name = f'{message["message_id"]}.safetensors'
        expect_file_path = os.path.join(KV_CACHE_DIR, expect_file_name)
        if os.path.exists(expect_file_path):
            index = len(messages) - reversed_index
            logger.debug(f"kv cache hit. {expect_file_path}. index = {index}.")

            start_time = time.time()
            cache, metadata = load_prompt_cache(expect_file_path, return_metadata=True)
            load_time = time.time() - start_time
            os_stat = os.stat(expect_file_path)
            stats["filename"]  = expect_file_name
            stats["size"]      = os_stat.st_size
            stats["load_time"] = load_time
            return cache, metadata, index, stats

    logger.debug("kv cache not hit. create new cache.")
    metadata["token_count"] = 0
    cache = make_prompt_cache(model=model)
    return cache, metadata, None, stats

def save_kv_cache(message_id: str, kv_cache: List[any], metadata: Dict):
    clean_kv_cache()
    logger.debug(f"metadata={metadata}")
    filepath = os.path.join(KV_CACHE_DIR, f"{message_id}.safetensors")
    save_prompt_cache(file_name=filepath, cache=kv_cache, metadata=metadata)

def clean_kv_cache():
    """
    当 KV 缓存目录大小超过 max_kv_size（默认 10GB）时，删除最旧的文件。
    该操作会递归执行直到目录大小不再超过阈值。
    max_kv_size 通过启动参数 "--max-kv-size" 决定。
    """
    max_kv_size_gb = os.environ.get("MAX_KV_SIZE_GB", 10)
    max_kv_size = int(max_kv_size_gb) * (1024 ** 3)

    def get_dir_size(path):
        """递归计算目录大小"""
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    def get_oldest_file(path):
        """返回目录内最旧（按最后修改时间）的文件路径；若无则返回 None"""
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if not files:
            return None
        return min(files, key=os.path.getmtime) 
    
    dir_size = get_dir_size(KV_CACHE_DIR)

    if dir_size > max_kv_size:
        logger.debug(f"kv cache size overed threshold: {dir_size}")
        oldest_file = get_oldest_file(KV_CACHE_DIR)
        if oldest_file:
          os.remove(oldest_file)
          logger.debug(f"deleted kv cache: {oldest_file}")
        clean_kv_cache()