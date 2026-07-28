"""双层缓存键构造 + Judge 缓存 get/set。Generator 缓存在 runner.py 中直接操作。"""

import hashlib
from infra.cache import get_cache
from infra.config import REDIS_DEFAULT_TTL
from config import GENERATOR_CONFIG_HASH


def _judge_cache_key(query_id: str, context_str: str,
                     prompt_version: str, model_id: str) -> str:
    """Judge 缓存键（不含 answer_hash，改用 GENERATOR_CONFIG_HASH 作为 Generator 配置变更信号）。"""
    context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
    return f"judge:{query_id}:{context_hash}:{GENERATOR_CONFIG_HASH}:{prompt_version}:{model_id}"


def _generator_cache_key(query_id: str, context_str: str) -> str:
    """Generator 缓存键。"""
    context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
    return f"generator:{query_id}:{context_hash}:{GENERATOR_CONFIG_HASH}"


def get_cached_result(cache_key: str) -> "JudgeResult | None":
    """查 Judge 缓存，命中则记录 metric 并返回反序列化的 JudgeResult。"""
    from eval.core.llm_as_judge import JudgeResult
    from eval.core.monitor_metrics import get_metrics
    cache = get_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        get_metrics().record_judge_cache_hit()
        return JudgeResult.from_json(cached)
    get_metrics().record_judge_cache_miss()
    return None


def set_cached_result(cache_key: str, result: "JudgeResult"):
    """将 JudgeResult 写入缓存。"""
    cache = get_cache()
    cache.set(cache_key, result.to_json(), ttl_seconds=REDIS_DEFAULT_TTL)
