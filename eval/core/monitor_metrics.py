"""Eval 成本与质量监控。模块级单例，采集 token 计数、缓存命中率、错误率。"""

import logging
from dataclasses import dataclass, field


@dataclass
class MonitorMetrics:
    # Token 计数
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    input_tokens_list: list[int] = field(default_factory=list)
    output_tokens_list: list[int] = field(default_factory=list)

    # 延迟（TTFT 预留）
    ttft_list: list[float] = field(default_factory=list)
    total_latency_list: list[float] = field(default_factory=list)

    # 缓存
    cache_hits: int = 0
    cache_misses: int = 0

    # 错误
    error_count: int = 0
    error_types: dict[str, int] = field(default_factory=dict)

    # 调用计数（错误率分母 —— 独立于缓存命中/未命中）
    total_llm_calls: int = 0

    # 成本估算（env 可调）
    input_price_per_1k: float = 0.0003
    output_price_per_1k: float = 0.0012

    def record_tokens(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.input_tokens_list.append(input_tokens)
        self.output_tokens_list.append(output_tokens)

    def record_latency(self, total_seconds: float):
        self.total_latency_list.append(total_seconds)

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def record_llm_call(self):
        self.total_llm_calls += 1

    def record_error(self, error_type: str):
        self.error_count += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.total_llm_calls if self.total_llm_calls > 0 else 0.0

    @property
    def estimated_cost_usd(self) -> float:
        input_cost = (self.total_input_tokens / 1000) * self.input_price_per_1k
        output_cost = (self.total_output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost

    @staticmethod
    def _p95(values: list) -> float:
        if not values:
            return 0.0
        import numpy
        return float(numpy.percentile(values, 95))

    @property
    def input_tokens_p95(self) -> float:
        return self._p95(self.input_tokens_list)

    @property
    def output_tokens_p95(self) -> float:
        return self._p95(self.output_tokens_list)

    def summary_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "input_tokens_p95": self.input_tokens_p95,
            "output_tokens_p95": self.output_tokens_p95,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "errors_by_type": dict(self.error_types),
        }

    def print_summary(self):
        logging.info(
            "cost: $%.6f | cache hits: %.0f%% | errors: %d | input P95: %d tok | output P95: %d tok",
            self.estimated_cost_usd,
            self.cache_hit_rate * 100,
            self.error_count,
            int(self.input_tokens_p95),
            int(self.output_tokens_p95),
        )


_metrics: MonitorMetrics | None = None


def get_metrics() -> MonitorMetrics:
    """模块级单例。"""
    global _metrics
    if _metrics is None:
        from config import COST_INPUT_PRICE_PER_1K, COST_OUTPUT_PRICE_PER_1K
        _metrics = MonitorMetrics(
            input_price_per_1k=COST_INPUT_PRICE_PER_1K,
            output_price_per_1k=COST_OUTPUT_PRICE_PER_1K,
        )
    return _metrics


def reset_metrics():
    """重置指标（每次 cmd_full 开头调用）。"""
    global _metrics
    _metrics = None
