from infra.cache.backend import CacheBackend


class NoopBackend(CacheBackend):
    """吞掉所有操作的缓存后端。Redis 不可用时降级为此类，保证 eval 流程不中断。"""

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        pass

    def delete_pattern(self, pattern: str) -> int:
        return 0
