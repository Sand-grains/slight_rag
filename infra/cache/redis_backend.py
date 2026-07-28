"""Redis 缓存后端：redis-py 封装，支持 SCAN 批量删除。

核心特性：
    - from_url 连接 + key_prefix 命名空间隔离
    - set 使用 setex 原子设置值 + TTL
    - delete_pattern 使用 SCAN 批量删除（非 KEYS，生产安全）
    - exists 基于 EXISTS 命令，O(1)

用法示例::

    from infra.cache.redis_backend import RedisBackend
    backend = RedisBackend("redis://localhost:6379/0", key_prefix="slight_rag")
    backend.set("key", "value", ttl_seconds=3600)

公共接口：
    - RedisBackend: Redis 缓存后端（一个实例持有一个连接池）
"""

import redis
from infra.cache.backend import CacheBackend


class RedisBackend(CacheBackend):
    """Redis 缓存后端。一个实例持有一个连接池。"""

    def __init__(self, redis_url: str, key_prefix: str = "slight_rag"):
        self._client = redis.Redis.from_url(redis_url)
        self._prefix = key_prefix

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def get(self, key: str) -> str | None:
        value = self._client.get(self._full_key(key))
        return value.decode("utf-8") if value else None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._client.setex(self._full_key(key), ttl_seconds, value)

    def delete_pattern(self, pattern: str) -> int:
        full_pattern = f"{self._prefix}:{pattern}"
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match=full_pattern, count=100)
            if keys:
                deleted += self._client.delete(*keys)
            if cursor == 0:
                break
        return deleted

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(self._full_key(key)))
