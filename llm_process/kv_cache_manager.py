import sys
import time
from typing import Dict, Any

class KVCacheManager:
    cache_table: Dict[int, Dict[str, Any]] = {}

    @classmethod
    def set_cache(cls, session_id: int, value: Any) -> None:
        if session_id not in cls.cache_table:
            cls.cache_table[session_id] = {}
        cls.cache_table[session_id]['cache'] = value
        cls.cache_table[session_id]['last_updated'] = int(time.time())

    @classmethod
    def has_cache(cls, session_id: int) -> bool:
        return session_id in cls.cache_table and 'cache' in cls.cache_table[session_id]

    @classmethod
    def get_cache(cls, session_id: int) -> Any:
        return cls.cache_table[session_id].get('cache', None)

    @classmethod
    def get_last_updated(cls, session_id: int) -> int:
        return cls.cache_table[session_id].get('last_updated', 0)

    @classmethod
    def remove_cache(cls, session_id: int) -> None:
        cls.cache_table.pop(session_id, None)

    @classmethod
    def remove_old_caches(cls, seconds: int) -> None:
        current_time = int(time.time())
        threshold = current_time - seconds

        sessions_to_remove = [
            session_id for session_id, session_data in cls.cache_table.items()
            if session_data.get('last_updated', 0) < threshold
        ]
        for session_id in sessions_to_remove:
            cls.remove_cache(session_id)

    @classmethod
    def get_all_sessions_with_size(cls) -> Dict[int, int]:
        return {
            session_id: cls.get_object_size(session_data.get('cache'))
            for session_id, session_data in cls.cache_table.items()
            if 'cache' in session_data
        }

    @staticmethod
    def get_object_size(obj: Any) -> int:
        return sys.getsizeof(obj)
    
    @classmethod
    def get_total_cache_size(cls) -> int:
        return sum(cls.get_all_sessions_with_size().values())