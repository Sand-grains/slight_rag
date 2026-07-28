"""缓存后端抽象接口（ABC）。方法签名对齐 Redis 语义但不绑定 Redis。

核心特性：
    - get / set / delete_pattern / exists 四方法接口
    - 子类实现不限于 Redis——NoopBackend 吞掉所有操作，未来可扩展 FileBackend 等

用法示例::

    from infra.cache.backend import CacheBackend
    class MyBackend(CacheBackend):
        def get(self, key): ...
        def set(self, key, value, ttl_seconds): ...
        def delete_pattern(self, pattern): ...

公共接口：
    - CacheBackend: 抽象基类（get / set / delete_pattern / exists）
"""

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
