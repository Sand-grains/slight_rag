"""Judge 结果缓存：内容寻址键构造 + get/set 编排。"""

import hashlib
from infra.cache import get_cache
from infra.config import REDIS_DEFAULT_TTL


def _judge_cache_key(query_id: str, context_str: str, answer: str,
                     prompt_version: str, model_id: str) -> str:
    """内容寻址缓存键。query_id 保留用于 Redis 中可读性（调试可追踪）。"""
    context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
    answer_hash = hashlib.sha256(answer.encode()).hexdigest()[:16]
    return f"judge:{query_id}:{context_hash}:{answer_hash}:{prompt_version}:{model_id}"


def get_cached_result(cache_key: str) -> "JudgeResult | None":
    """查缓存，命中则记录 metric 并返回反序列化的 JudgeResult。"""
    from eval.core.llm_as_judge import JudgeResult
    from eval.core.monitor_metrics import get_metrics
    cache = get_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        get_metrics().record_cache_hit()
        return JudgeResult.from_json(cached)
    get_metrics().record_cache_miss()
    return None


def set_cached_result(cache_key: str, result: "JudgeResult"):
    """将 JudgeResult 写入缓存。"""
    cache = get_cache()
    cache.set(cache_key, result.to_json(), ttl_seconds=REDIS_DEFAULT_TTL)
