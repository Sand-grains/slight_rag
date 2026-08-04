"""缓存后端抽象接口（ABC）。方法签名对齐 Redis 语义但不绑定 Redis。

核心特性：
    - get / set / delete_by_pattern / exists 四方法接口
    - 子类实现不限于 Redis——NoopBackend 吞掉所有操作，未来可扩展 FileBackend 等

用法示例::

    from infra.cache.backend import CacheBackend
    class MyBackend(CacheBackend):
        def get(self, key): ...
        def set(self, key, value, ttl_seconds): ...
        def delete_by_pattern(self, pattern): ...

公共接口：
    - CacheBackend: 抽象基类（get / set / delete_by_pattern / exists）
"""

from abc import ABC, abstractmethod


class CacheBackend(ABC):
    """缓存后端抽象接口。方法签名对齐 Redis 语义但不绑定 Redis。"""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """读取缓存值。

        Args:
            key: 缓存键（不含命名空间前缀）。

        Returns:
            str | None：缓存值；未命中或已过期时返回 None。
        """
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """写入缓存值(同时设置 TTL)。

        Args:
            key: 缓存键（不含命名空间前缀）。
            value: 缓存值。
            ttl_seconds: 存活时间（秒），到期自动过期。
        """
        pass

    @abstractmethod
    def delete_by_pattern(self, pattern: str) -> int:
        """按模式批量删除缓存键。

        Args:
            pattern: 键匹配模式（如 "judge:*"）。

        Returns:
            int：删除的键数量。
        """
        pass

    def is_exists(self, key: str) -> bool:
        """判断缓存键是否存在（基于 get 是否命中）。

        Args:
            key: 缓存键（不含命名空间前缀）。

        Returns:
            bool：存在返回 True；未命中或已过期返回 False。
        """
        return self.get(key) is not None
