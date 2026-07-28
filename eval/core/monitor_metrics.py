"""Eval 运行时监控。模块级单例，MonitorMetrics 为唯一真源——LivePanel 和 reporter 均为只读消费者。"""

import logging
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MonitorMetrics:
    # == Token ==
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    input_tokens_list: list[int] = field(default_factory=list)
    output_tokens_list: list[int] = field(default_factory=list)

    # == 阶段延迟（per-query 毫秒列表）==
    stage_retrieve_ms: list[float] = field(default_factory=list)
    stage_generate_ms: list[float] = field(default_factory=list)
    stage_judge_faithfulness_ms: list[float] = field(default_factory=list)
    stage_judge_quality_ms: list[float] = field(default_factory=list)

    # == 缓存（拆分为 Generator / Judge 独立计数）==
    generator_cache_hits: int = 0
    generator_cache_misses: int = 0
    judge_cache_hits: int = 0
    judge_cache_misses: int = 0

    # == 错误与重试 ==
    error_count: int = 0
    error_types: dict[str, int] = field(default_factory=dict)
    retry_count: int = 0
    parse_error_count: int = 0

    # == LLM 调用计数（按类型拆分）==
    generator_llm_calls: int = 0
    judge_faithfulness_calls: int = 0
    judge_quality_calls: int = 0

    # == Per-query 原始结果（唯一真源，裸 list 类型避免 import 依赖）==
    layer1_results: list = field(default_factory=list)   # list[RetrievalEvalResult]
    layer2_results: list = field(default_factory=list)   # list[JudgeResult]

    # == 成本 ==
    input_price_per_1k: float = 0.0003
    output_price_per_1k: float = 0.0012

    # === 记录方法 ===
    def record_tokens(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.input_tokens_list.append(input_tokens)
        self.output_tokens_list.append(output_tokens)

    def record_stage(self, stage: str, ms: float):
        getattr(self, f"stage_{stage}_ms").append(ms)

    def record_generator_cache_hit(self):
        self.generator_cache_hits += 1

    def record_generator_cache_miss(self):
        self.generator_cache_misses += 1

    def record_judge_cache_hit(self):
        self.judge_cache_hits += 1

    def record_judge_cache_miss(self):
        self.judge_cache_misses += 1

    def record_llm_call(self, call_type: str):
        """call_type: 'generator' | 'judge_faithfulness' | 'judge_quality'"""
        if call_type == "generator":
            self.generator_llm_calls += 1
        elif call_type == "judge_faithfulness":
            self.judge_faithfulness_calls += 1
        elif call_type == "judge_quality":
            self.judge_quality_calls += 1

    def record_retry(self):
        self.retry_count += 1

    def record_parse_error(self):
        self.parse_error_count += 1

    def record_error(self, error_type: str):
        self.error_count += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

    # === 计算属性 ===
    @property
    def total_llm_calls(self) -> int:
        return self.generator_llm_calls + self.judge_faithfulness_calls + self.judge_quality_calls

    @property
    def generator_cache_hit_rate(self) -> float:
        total = self.generator_cache_hits + self.generator_cache_misses
        return self.generator_cache_hits / total if total > 0 else 0.0

    @property
    def judge_cache_hit_rate(self) -> float:
        total = self.judge_cache_hits + self.judge_cache_misses
        return self.judge_cache_hits / total if total > 0 else 0.0

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

    # === 聚合方法（LivePanel 面板和 render_final 共用）===
    def layer1_means(self) -> dict[str, float]:
        results = list(self.layer1_results)
        if not results:
            return {}
        n = len(results)
        return {
            "recall_at_k": sum(r.recall_at_k for r in results) / n,
            "precision_at_k": sum(r.precision_at_k for r in results) / n,
            "hit_at_k": sum(r.hit_at_k for r in results) / n,
            "mrr": sum(r.mrr for r in results) / n,
            "map_at_k": sum(r.map_at_k for r in results) / n,
            "ndcg_at_k": sum(r.ndcg_at_k for r in results) / n,
        }

    def layer2_means(self) -> dict[str, Optional[float]]:
        results = list(self.layer2_results)
        valid = [r for r in results if r.faithfulness is not None]
        if not valid:
            return {}
        n = len(valid)
        return {
            "faithfulness": sum(r.faithfulness or 0 for r in valid) / n,
            "answer_relevancy": sum(r.answer_relevancy or 0 for r in valid) / n,
            "context_precision": sum(r.context_precision or 0 for r in valid) / n,
            "context_recall": sum(r.context_recall or 0 for r in valid) / n,
            "answer_correctness": sum(r.answer_correctness or 0 for r in valid) / n,
        }

    def verdict_distribution(self) -> dict[str, int]:
        counts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
        for r in self.layer2_results:
            v = r.verdict if r.verdict in counts else "error"
            counts[v] += 1
        return counts

    def stage_percentiles(self) -> dict[str, dict[str, float]]:
        stages = {
            "retrieve": list(self.stage_retrieve_ms),
            "generate": list(self.stage_generate_ms),
            "judge_faithfulness": list(self.stage_judge_faithfulness_ms),
            "judge_quality": list(self.stage_judge_quality_ms),
        }
        import numpy
        result = {}
        for name, values in stages.items():
            if not values:
                result[name] = {"p50": 0.0, "p75": 0.0, "p95": 0.0}
            else:
                result[name] = {
                    "p50": float(numpy.percentile(values, 50)),
                    "p75": float(numpy.percentile(values, 75)),
                    "p95": float(numpy.percentile(values, 95)),
                }
        return result

    # === 序列化 ===
    def summary_dict(self) -> dict:
        sp = self.stage_percentiles()
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "input_tokens_p95": self.input_tokens_p95,
            "output_tokens_p95": self.output_tokens_p95,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "generator_cache_hit_rate": round(self.generator_cache_hit_rate, 4),
            "judge_cache_hit_rate": round(self.judge_cache_hit_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "errors_by_type": dict(self.error_types),
            "retry_count": self.retry_count,
            "parse_error_count": self.parse_error_count,
            "generator_llm_calls": self.generator_llm_calls,
            "judge_faithfulness_calls": self.judge_faithfulness_calls,
            "judge_quality_calls": self.judge_quality_calls,
            "stage_retrieve_p50": sp.get("retrieve", {}).get("p50", 0.0),
            "stage_retrieve_p75": sp.get("retrieve", {}).get("p75", 0.0),
            "stage_retrieve_p95": sp.get("retrieve", {}).get("p95", 0.0),
            "stage_generate_p50": sp.get("generate", {}).get("p50", 0.0),
            "stage_generate_p75": sp.get("generate", {}).get("p75", 0.0),
            "stage_generate_p95": sp.get("generate", {}).get("p95", 0.0),
            "stage_judge_faithfulness_p50": sp.get("judge_faithfulness", {}).get("p50", 0.0),
            "stage_judge_faithfulness_p75": sp.get("judge_faithfulness", {}).get("p75", 0.0),
            "stage_judge_faithfulness_p95": sp.get("judge_faithfulness", {}).get("p95", 0.0),
            "stage_judge_quality_p50": sp.get("judge_quality", {}).get("p50", 0.0),
            "stage_judge_quality_p75": sp.get("judge_quality", {}).get("p75", 0.0),
            "stage_judge_quality_p95": sp.get("judge_quality", {}).get("p95", 0.0),
        }

    def print_summary(self):
        logging.info(
            "cost: $%.6f | Gen cache: %.0f%% | Judge cache: %.0f%% | errors: %d | "
            "retries: %d | parse err: %d | input P95: %d tok | output P95: %d tok",
            self.estimated_cost_usd,
            self.generator_cache_hit_rate * 100,
            self.judge_cache_hit_rate * 100,
            self.error_count,
            self.retry_count,
            self.parse_error_count,
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
    """重置指标（每次 run_full_mode 开头调用）。"""
    global _metrics
    _metrics = None
