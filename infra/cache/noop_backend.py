"""Noop 缓存后端：吞掉所有操作，Redis 不可用时的降级方案。

核心特性：
    - get 永远返回 None，set 和 delete_by_pattern 无操作
    - 保证 eval 流程不因 Redis 不可用而中断
    - LivePanel 检测到 NoopBackend 时显示 'Redis: unavailable (no cache)'，缓存命中率显示 '—'

用法示例::

    from infra.cache.noop_backend import NoopBackend
    backend = NoopBackend()
    assert backend.get("any_key") is None  # 永远返回 None

公共接口：
    - NoopBackend: 吞掉所有操作的缓存后端
"""

from infra.cache.backend import CacheBackend


class NoopBackend(CacheBackend):
    """吞掉所有操作的缓存后端。Redis 不可用时降级为此类，保证 eval 流程不中断。"""

    def get(self, key: str) -> str | None:
        """永远返回 None（即无缓存命中）。"""
        return None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """无操作：吞掉对缓存的写入。"""
        pass

    def delete_by_pattern(self, pattern: str) -> int:
        """无操作：返回 0。"""
        return 0
