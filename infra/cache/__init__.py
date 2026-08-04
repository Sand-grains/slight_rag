"""缓存工厂：RedisBackend + NoopBackend 降级。

核心特性：
    - get_cache() 模块级单例，双重检查锁（threading.Lock）防止并行区域多线程重复创建
    - 首次调用时尝试创建 RedisBackend 并做健康检查，失败则自动降级为 NoopBackend
    - NoopBackend 的 get 永远返回 None，eval 流程不中断，缓存命中率显示 '—'

用法示例::

    from infra.cache import get_cache
    cache = get_cache()
    cache.set("key", "value", ttl_seconds=3600)

公共接口：
    - get_cache: 获取缓存后端单例（RedisBackend | NoopBackend）
"""

import logging
import threading
from infra.cache.backend import CacheBackend
from infra.cache.noop_backend import NoopBackend

_cache: CacheBackend | None = None
_lock = threading.Lock()


def get_cache() -> CacheBackend:
    """获取缓存后端单例：Redis 可用则用 Redis，否则降级为 NoopBackend。

    模块级单例 + 双重检查锁，保证并发下只创建一次后端实例。

    Returns:
        CacheBackend：进程内唯一的缓存后端实例（RedisBackend 或 NoopBackend）。
    """
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                try:
                    from infra.cache.redis_backend import RedisBackend
                    from infra.config import REDIS_CONNECTION_URL, REDIS_KEY_PREFIX
                    backend = RedisBackend(REDIS_CONNECTION_URL, key_prefix=REDIS_KEY_PREFIX)
                    backend.set("__health_check__", "ok", ttl_seconds=10)
                    _cache = backend
                    logging.info("Redis 缓存后端已连接：%s", REDIS_CONNECTION_URL)
                except Exception as error:
                    logging.warning("Redis 不可用，降级为 NoopBackend。错误：%s", error)
                    _cache = NoopBackend()
    return _cache
