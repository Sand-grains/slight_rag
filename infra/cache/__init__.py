import logging
from infra.cache.backend import CacheBackend
from infra.cache.noop_backend import NoopBackend

_cache: CacheBackend | None = None


def get_cache() -> CacheBackend:
    """模块级单例。Redis 不可用时自动降级为 NoopBackend。"""
    global _cache
    if _cache is None:
        try:
            from infra.cache.redis_backend import RedisBackend
            from infra.config import REDIS_URL, REDIS_KEY_PREFIX
            backend = RedisBackend(REDIS_URL, key_prefix=REDIS_KEY_PREFIX)
            backend.set("__health_check__", "ok", ttl_seconds=10)
            _cache = backend
            logging.info("Redis 缓存后端已连接: %s", REDIS_URL)
        except Exception as e:
            logging.warning("Redis 不可用 (%s)，降级为 NoopBackend", e)
            _cache = NoopBackend()
    return _cache
