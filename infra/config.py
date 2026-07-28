"""Infra 层配置：Redis 连接参数。

核心特性：
    - REDIS_URL / REDIS_KEY_PREFIX / REDIS_DEFAULT_TTL 均从 .env 读取，带默认值
    - _PROJECT_ROOT 从根 config.py 导入
    - REDIS_DEFAULT_TTL 默认 259200（72h），匹配当前 1-3 天一个优化迭代的开发节奏

用法示例::

    from infra.config import REDIS_URL, REDIS_DEFAULT_TTL
"""

import os
from config import _PROJECT_ROOT

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "slight_rag")
REDIS_DEFAULT_TTL = int(os.getenv("REDIS_DEFAULT_TTL", "259200"))  # 72h
