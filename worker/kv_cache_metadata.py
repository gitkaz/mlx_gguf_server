from threading import RLock
from typing import Dict, Any

class KVCacheMetadataStore:
    """
    KVCacheMetadataStore is a persistent datastore of KV Cache files.
    This is created when llm_prosss started and maintain each KV Cache files path and 
    it's metadata information while process terminated.

    These data is storead in the single dict (self._entries).
    * key = file_pass of a KV Cache file.
    * value = dict of KV Cache file's metadata.
    """

    def __init__(self):
        self._entries: Dict[str, Dict[str, Any]] = {}  # file_path -> metadata dict
        self._lock = RLock()

    def add(self, file_path: str, metadata: Dict[str, Any]):
        """Add metadata entry (called after cache save)"""
        with self._lock:
            self._entries[file_path] = metadata

    def remove(self, file_path: str):
        """Remove metadata entry (called after cache delete)"""
        with self._lock:
            if file_path in self._entries:
                del self._entries[file_path]

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return all metadata (for cache matching)"""
        with self._lock:
            return self._entries.copy()