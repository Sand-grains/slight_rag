from abc import ABC, abstractmethod


class CacheBackend(ABC):
    """缓存后端抽象接口。方法签名对齐 Redis 语义但不绑定 Redis。"""

    @abstractmethod
    def get(self, key: str) -> str | None: ...

    @abstractmethod
    def set(self, key: str, value: str, ttl_seconds: int) -> None: ...

    @abstractmethod
    def delete_pattern(self, pattern: str) -> int: ...

    def exists(self, key: str) -> bool:
        return self.get(key) is not None
