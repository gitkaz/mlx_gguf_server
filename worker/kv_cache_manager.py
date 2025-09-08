import os
import time
import json
from typing import Dict, Any, List
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
from .logger_config import setup_logger

logger = setup_logger(__name__, level="DEBUG")

KV_CACHE_DIR = "worker/kv_cache"

def load_kv_cache(model, prompt_tokens: List):
    """
    Find the best matching KV cache by comparing token sequences (not message IDs)

    Returns:
        (cache, metadata, start_index, stats) where:
        - cache: The KV cache to use
        - metadata: Associated metadata
        - start_index: Where to start processing in the tokenized prompt
        - stats: Performance/stats information
    """
    metadata = {}
    stats = {}

    start_time = time.time()
    cache_files = []
    for f in os.listdir(KV_CACHE_DIR):
        if f.endswith('.safetensors'):
            f_path = os.path.join(KV_CACHE_DIR, f)
            # Skip files that are too new (still being written)
            if time.time() - os.path.getmtime(f_path) < 1:
                continue
            cache_files.append((os.path.getmtime(f_path), f_path))
    # Sort by modification time, newest first
    cache_files.sort(key=lambda x: x[0], reverse=True)
    sort_time = time.time() - start_time
    logger.debug(f"{sort_time=}")

    best_match = {
        "prefix_len": 0,
        "cache": None,
        "metadata": None,
        "file_path": None
    }    

    # Check each cache file for the best token match
    for _, file_path in cache_files:
        try:
            # Load cache with metadata
            cache, metadata = load_prompt_cache(file_path, return_metadata=True)
            os_stat = os.stat(file_path)

            # Extract tokens from metadata
            try:
                cached_tokens = json.loads(metadata.get("tokens", "[]"))
            except:
                continue

            # Calculate common prefix length
            common_len = 0
            min_len = min(len(cached_tokens), len(prompt_tokens))
            for i in range(min_len):
                if cached_tokens[i] != prompt_tokens[i]:
                    break
                common_len = i + 1

            # Only consider caches where:
            # - We have a non-zero match
            # - Current prompt is longer than cached tokens
            # - This is the best match so far
            if (common_len > 0 and 
                len(prompt_tokens) > len(cached_tokens) and
                common_len > best_match["prefix_len"]):

                best_match.update({
                    "prefix_len": common_len,
                    "cache": cache,
                    "metadata": metadata,
                    "file_path": file_path,
                    "size": os_stat.st_size
                })

        except Exception as e:
            logger.warning(f"Error loading cache {file_path}: {str(e)}")
            continue

    # Process the best match
    if best_match["cache"] is not None:
        logger.debug(f"KV cache hit! Prefix length: {best_match['prefix_len']}, "
                    f"from cache: {os.path.basename(best_match['file_path'])}")
        
        # Return the cache, metadata, and where to start processing
        return (
            best_match["cache"],
            best_match["metadata"],
            best_match["prefix_len"],
            {
                "prefix_len": best_match["prefix_len"],
                "cached_tokens": len(json.loads(best_match["metadata"].get("tokens", "[]"))),
                "filename": os.path.basename(best_match["file_path"]),
                "size": best_match["size"],
                "load_time": 0
            }
        )

    # No suitable cache found
    logger.debug("No KV cache hit. Creating new cache.")
    return make_prompt_cache(model), {"token_count": 0}, 0, {}

def save_kv_cache(message_id: str, kv_cache: List[any], metadata: Dict):
    clean_kv_cache()
    filepath = os.path.join(KV_CACHE_DIR, f"{message_id}.safetensors")
    save_prompt_cache(file_name=filepath, cache=kv_cache, metadata=metadata)

def clean_kv_cache():
    """
    KVキャッシュディレクトリのサイズがmax_kv_size(デフォルト10GB)を超える場合に、
    最も古いファイルを削除する関数。条件を満たさなくなるまで再帰的に実行する。
    max_kv_size は、mlx_gguf_server 起動時の引数 "--max-kv-size" で決定する。
    """
    max_kv_size_gb = os.environ.get("MAX_KV_SIZE_GB", 10)
    max_kv_size = int(max_kv_size_gb) * (1024 ** 3)

    def get_dir_size(path):
        """ディレクトリのサイズを再帰的に計算する"""
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    def get_oldest_file(path):
        """ディレクトリ内の最も古い(最終変更時刻)ファイルのパスを返す。ファイルがない場合はNoneを返す"""
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