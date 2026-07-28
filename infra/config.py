import os
from config import _PROJECT_ROOT

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "slight_rag")
REDIS_DEFAULT_TTL = int(os.getenv("REDIS_DEFAULT_TTL", "259200"))  # 72h
