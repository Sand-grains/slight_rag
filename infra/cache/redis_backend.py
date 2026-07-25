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
