import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache, can_trim_prompt_cache, trim_prompt_cache
from ..utils import get_dir_size, get_oldest_file
from ..logger_config import setup_logger
from .kv_cache_metadata import KVCacheMetadataStore

log_level = os.environ.get("LOG_LEVEL", "INFO")
logger = setup_logger(__name__, level=log_level)

class KVCacheManager:
    """
    handle KV Cache related operations.
    - create new KV Cache
    - search appropriate Cache file
    - calculate start position of non-cached prompt begins
    - load KV Cache
    - save KV Cache
    - delete KV Cache files
    """

    def __init__(self, metadata_store: KVCacheMetadataStore):
        self.cache_dir = self.get_cache_dir()
        self.metadata_store = metadata_store

    @staticmethod
    def get_cache_dir():
        """
        return absolute path of kv_cache directory.
        """
        return os.environ["KV_CACHE_PATH"]

    def _get_cache_files_list(self):
        """
        gather kv_cache files (.safetensor files in worker/kv_cache/data directory).
          then sort the file list by modification time. newest first.
          then return sorted list.
        """
        cache_files = []
        for f in os.listdir(self.cache_dir):
            if f.endswith('.safetensors'):
                f_path = os.path.join(self.cache_dir, f)
                # Skip files that are too new (still being written)
                if time.time() - os.path.getmtime(f_path) < 1:
                    continue
                cache_files.append((os.path.getmtime(f_path), f_path))
        # Sort by modification time, newest first
        cache_files.sort(key=lambda x: x[0], reverse=True)
        return cache_files

    def _calc_common_length(self, cached_tokens, prompt_tokens) -> int :
        """
        calculate length of common tokens between cache_tokens and prompt_tokens.
        """
        common_len = 0
        min_len = min(len(cached_tokens), len(prompt_tokens))
        for i in range(min_len):
            if cached_tokens[i] != prompt_tokens[i]:
                break
            common_len = i + 1
        
        return common_len
    

    def _load_best_match_cache(self, file_path: str, common_len: int):
        cache, metadata = load_prompt_cache(file_path, return_metadata=True)

        try:
            cached_tokens = json.loads(metadata.get("tokens", "[]"))
        except:
            logger.error(f"KV Cache {os.path.basename(file_path)} failed to load metadata.")
            cached_tokens = []

        # if common_len < lan(cached_tokens), cache is needed to be trimmed.
        if 0 < common_len and common_len < len(cached_tokens):
            logger.debug(f"{common_len=}, {len(cached_tokens)=}. Try to trim cache file: {Path(file_path).name}")
            if can_trim_prompt_cache(cache):
                # 2025/09/10. mlx-0.29.0 and mlx_lm-0.27.0
                # as far as I tested, trim_len need to be add 1... (not sure why... orz)
                trim_len = len(cached_tokens) - common_len +1
                trim_prompt_cache(cache, trim_len)
                logger.debug(f"{Path(file_path).name} was trimmed {trim_len} tokens.")
            else:
                logger.warning(f"Cache file {Path(file_path).name} is not able to trim.")

        return cache, metadata


    def _find_best_match_cache(self, model_name: str, all_metadata: Dict[str, Dict], prompt_tokens:List):
        """

        """
        best_match = {
            "common_len": 0,
            "file_path": None,
        }    

        # Check each cache file for the best token match
        for file_path, metadata in all_metadata.items():
            try:
                #compare by model_name
                if model_name != metadata["model_name"]:
                    continue

                # Load cache with metadata
                cached_tokens = json.loads(metadata["tokens"])
                if cached_tokens is None:
                    continue

                # Calculate common prefix length
                common_len = self._calc_common_length(cached_tokens, prompt_tokens)

                # If common tokens exist but it is shorter than cached_tokens, then cache needs to be trimmed.
                if 0 < common_len and common_len < len(cached_tokens):
                    if metadata.get("trimmable") and metadata["trimmable"] == "False":
                        continue

                if (best_match["common_len"] < common_len and common_len < len(prompt_tokens)):
                    best_match.update({
                        "common_len": common_len,
                        "file_path": file_path,
                    })

            except Exception as e:
                logger.warning(f"Error loading cache {file_path}: {str(e)}")
                continue

        return best_match


    def save_kv_cache(self, message_id: str, kv_cache: List[any], metadata: Dict):
        self.clean_kv_cache()
        if can_trim_prompt_cache(kv_cache):
            metadata["trimmable"] = "True"
        else:
            metadata["trimmable"] = "False"

        filepath = os.path.join(self.cache_dir, f"{message_id}.safetensors")
        logger.info(f"saving {message_id}.safetensors")
        save_prompt_cache(file_name=filepath, cache=kv_cache, metadata=metadata)
        # Update metadata store
        self.metadata_store.add(filepath, metadata)
        # set file as read-only
        os.chmod(filepath, 0o444)

    def clean_kv_cache(self):
        """
        KVキャッシュディレクトリのサイズがmax_kv_size(デフォルト10GB)を超える場合に、
        最も古いファイルを削除する関数。条件を満たさなくなるまで再帰的に実行する。
        max_kv_size は、mlx_gguf_server 起動時の引数 "--max-kv-size" で決定する。
        """
        max_kv_size_gb = os.environ.get("MAX_KV_SIZE_GB", 10)
        max_kv_size = int(max_kv_size_gb) * (1024 ** 3)
        
        dir_size = get_dir_size(self.cache_dir)

        if dir_size > max_kv_size:
            logger.info(f"kv cache size overed threshold: {dir_size}")
            oldest_file = get_oldest_file(self.cache_dir)
            if oldest_file:
                self.metadata_store.remove(oldest_file)
                os.remove(oldest_file)
                logger.info(f"deleted kv cache: {oldest_file}")
            self.clean_kv_cache()

    def load_kv_cache_by_message_id(self, model_name: str, messages: List):
        """
        Find the best matching KV cache based on message id.

        Returns:
            (cache, start_index, stats) where:
            - cache: The KV cache to use
            - index: The nummber of array of messages whether message_id matches.
            - stats: loaded kv_cacshe's info (filename, cached token length, filesize, and time spent load)
            - metadata tokens: for debugging purpose
        """

        all_metadata = self.metadata_store.get_all_metadata()

        for reversed_index, message in enumerate(reversed(messages)):
            if "message_id" not in message:
                continue

            expect_file_name = f'{message["message_id"]}.safetensors'
            expect_file_path = Path(self.get_cache_dir()).joinpath(expect_file_name)

            if str(expect_file_path) in all_metadata.keys():

                # check model_name
                if model_name != all_metadata[str(expect_file_path)]["model_name"]:
                    continue

                index = len(messages) - reversed_index
                logger.info(f"kv cache hit by message_id based load_kv_cache. {str(expect_file_path)}. index = {index}.")

                start_time = time.time()
                cache, metadata = load_prompt_cache(expect_file_path, return_metadata=True)
                load_time = time.time() - start_time
                kv_load_stats = {"filename": expect_file_name,
                                "size": os.stat(expect_file_path).st_size,
                                "load_time": load_time
                                }
                return cache, index, kv_load_stats, json.loads(metadata["tokens"])

        # No suitable cache found
        return None, 0, {}, []

    def make_new_cache(self, model):
        dummy_stats =  {"filename": "new cache", "cached_tokens": 0, "size": 0, "load_time": 0}
        return make_prompt_cache(model), dummy_stats


    def load_kv_cache(self, model_name: str, prompt_tokens: List):
        """
        Find the best matching KV cache by comparing token sequences

        Returns:
            (cache, start_index, stats) where:
            - cache: The KV cache to use
            - start_index: Where to start processing in the tokenized prompt
            - stats: loaded kv_cacshe's info (filename, cached token length, filesize, and time spent load)
        """

        # 1. Get metadata from metadata store
        all_metadata = self.metadata_store.get_all_metadata()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("check all_metadata")
            for fp, md in all_metadata.items():
                logger.debug(f"filename = {fp}")
                tk = json.loads(md["tokens"])
                logger.debug(f"length of tokens={len(tk)}")
                logger.debug(f"tokens = {tk}")

        # find best maatch (longest match) kv cache file for prompt_tokens
        best_match = self._find_best_match_cache(model_name, all_metadata, prompt_tokens)

        # Process the best match
        if best_match["file_path"] is not None:
            logger.info(f"KV cache hit by token_based load_kv_cache. Common length: {best_match['common_len']}, "
                        f"from cache: {os.path.basename(best_match['file_path'])}")

            # load cache data to best_match["cache"]
            load_start_time = time.time()
            cache, metadata = self._load_best_match_cache(best_match["file_path"], best_match["common_len"])
            load_time = time.time() - load_start_time
            
            kv_load_stats = {"filename": os.path.basename(best_match["file_path"]),
                             "size": os.stat(best_match["file_path"]).st_size,
                             "load_time": load_time
                             }

            logger.debug(f"{kv_load_stats=}")
            return (cache, best_match["common_len"], kv_load_stats)

        # No suitable cache found
        return None, 0, {}

