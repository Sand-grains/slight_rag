"""评估报告生成：JSON 文件写入 + history.jsonl 追加。

核心特性：
    - generate_report() 一站式写入：per_query.json / failures.json / summary.json / run_info.json / history.jsonl
    - per_query.json 自动合并 Layer 1（检索指标 + 诊断）+ Layer 2（Judge 分数 + verdict + 阶段延迟）
    - _build_history() 向 eval/results/check/history.jsonl 追加一行运行摘要（跨运行索引）
    - _build_layer2_summary() 计算 Layer 2 聚合（5 项均值 + verdict 分布 + judge/generator errors）
    - 终端输出职责已移交 LivePanel，reporter 仅做文件 I/O
    - build_run_info() 含 git commit 快照 + 配置快照

用法示例::

    from eval.reporter import generate_report, build_run_info
    run_info = build_run_info("bench.json", run_mode="full")
    generate_report(layer1_output, results_dir, run_info, judge_results=judge_results, metrics_summary=metrics.summary_dict())

公共接口：
    - generate_report: 生成完整 eval 报告（5 个文件）
    - build_run_info: 构建运行配置快照 dict
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from eval.core.retrieval_layer import LayerOutput, RetrievalEvalResult


def generate_report(output: LayerOutput, results_dir: str, run_info: dict,
                    judge_results: list | None = None,
                    metrics_summary: dict | None = None) -> str:
    """生成完整 eval 报告：写入 summary.json / per_query.json / failures.json / run_info.json。

    Args:
        output: Layer 1 评估完整输出
        results_dir: 结果输出目录
        run_info: 运行配置快照
        judge_results: Layer 2 JudgeResult 列表(可选, full mode 时传入)
        metrics_summary: MonitorMetrics.summary_dict()(可选, full mode 时传入)
    """
    os.makedirs(results_dir, exist_ok=True)

    # per_query.json: query_id → 完整 trace
    per_query = {r.query_id: _serialize_result(r) for r in output.results}
    _write_json(os.path.join(results_dir, "per_query.json"), per_query)

    # failures.json: diagnosis != "accept" 的子集
    failures = {r.query_id: _serialize_result(r) for r in output.results if r.diagnosis != "accept"}
    _write_json(os.path.join(results_dir, "failures.json"), failures)

    # summary.json: 聚合 + 分组 + Layer 2 + cost
    summary = {
        "aggregate": output.aggregate,
        "by_category": output.by_category,
        "by_difficulty": output.by_difficulty,
    }
    if judge_results is not None:
        summary["layer2"] = _build_layer2_summary(judge_results)
    if metrics_summary is not None:
        summary["cost"] = metrics_summary
    _write_json(os.path.join(results_dir, "summary.json"), summary)

    # run_info.json: 配置快照
    _write_json(os.path.join(results_dir, "run_info.json"), run_info)

    # per_query.json: 合并 Layer 2 数据
    if judge_results is not None:
        for jr in judge_results:
            if jr.query_id in per_query:
                per_query[jr.query_id].update(_serialize_judge_result(jr))
        _write_json(os.path.join(results_dir, "per_query.json"), per_query)

    # history.jsonl: 追加一行
    _build_history(results_dir, output, run_info, judge_results, metrics_summary)

    return results_dir


def _serialize_result(r: RetrievalEvalResult) -> dict:
    return {
        "query_id": r.query_id,
        "query": r.query,
        "category": r.category,
        "difficulty": r.difficulty,
        "recall_at_k": r.recall_at_k,
        "precision_at_k": r.precision_at_k,
        "hit_at_k": r.hit_at_k,
        "mrr": r.mrr,
        "map_at_k": r.map_at_k,
        "ndcg_at_k": r.ndcg_at_k,
        "diagnosis": r.diagnosis,
        "final_chunk_ids": r.final_chunk_ids,
        "candidate_chunk_ids": r.candidate_chunk_ids,
        "retrieved_files": r.retrieved_files,
        "query_rewritten": r.query_rewritten,
        "query_rewritten_flag": r.query_rewritten_flag,
        "final_context_text": r.final_context_text,
    }


def _serialize_judge_result(jr) -> dict:
    """序列化 JudgeResult 中需写入 per_query.json 的字段。"""
    return {
        "faithfulness": jr.faithfulness,
        "answer_relevancy": jr.answer_relevancy,
        "context_precision": jr.context_precision,
        "context_recall": jr.context_recall,
        "answer_correctness": jr.answer_correctness,
        "verdict": jr.verdict,
        "parse_error": jr.parse_error,
        "judge_error": jr.judge_error,
        "generator_error": jr.generator_error,
        "retrieve_ms": jr.retrieve_ms,
        "generate_ms": jr.generate_ms,
        "judge_faithfulness_ms": jr.judge_faithfulness_ms,
        "judge_quality_ms": jr.judge_quality_ms,
    }


def _build_history(results_dir: str, output: LayerOutput, run_info: dict,
                    judge_results: list | None = None,
                    metrics_summary: dict | None = None):
    """向 history.jsonl 追加一行运行摘要。"""
    from config import _PROJECT_ROOT
    history_dir = str(_PROJECT_ROOT / "eval" / "results" / "check")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, "history.jsonl")

    agg = output.aggregate
    run_id = os.path.basename(results_dir)
    line = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "test_mode": run_info.get("test_mode", "retrieval"),
        "benchmark": run_info.get("benchmark", ""),
        "num_queries": agg.get("num_queries", 0),
        "recall_at_k": round(agg.get("recall_at_k", 0.0), 4),
        "precision_at_k": round(agg.get("precision_at_k", 0.0), 4),
        "hit_at_k": round(agg.get("hit_at_k", 0.0), 4),
        "mrr": round(agg.get("mrr", 0.0), 4),
        "map_at_k": round(agg.get("map_at_k", 0.0), 4),
        "ndcg_at_k": round(agg.get("ndcg_at_k", 0.0), 4),
    }
    if judge_results is not None:
        l2 = _build_layer2_summary(judge_results)
        line["faithfulness_avg"] = l2.get("faithfulness_avg")
        line["answer_relevancy_avg"] = l2.get("answer_relevancy_avg")
        line["context_precision_avg"] = l2.get("context_precision_avg")
        line["context_recall_avg"] = l2.get("context_recall_avg")
        line["answer_correctness_avg"] = l2.get("answer_correctness_avg")
    if metrics_summary is not None:
        line["generator_cache_hit_rate"] = metrics_summary.get("generator_cache_hit_rate")
        line["judge_cache_hit_rate"] = metrics_summary.get("judge_cache_hit_rate")
        line["estimated_cost_usd"] = metrics_summary.get("estimated_cost_usd")
        line["retry_count"] = metrics_summary.get("retry_count")
        line["parse_error_count"] = metrics_summary.get("parse_error_count")
        line["generator_llm_calls"] = metrics_summary.get("generator_llm_calls")
        line["judge_faithfulness_calls"] = metrics_summary.get("judge_faithfulness_calls")
        line["judge_quality_calls"] = metrics_summary.get("judge_quality_calls")
        line["stage_retrieve_p50"] = metrics_summary.get("stage_retrieve_p50")
        line["stage_retrieve_p95"] = metrics_summary.get("stage_retrieve_p95")
        line["stage_generate_p50"] = metrics_summary.get("stage_generate_p50")
        line["stage_generate_p95"] = metrics_summary.get("stage_generate_p95")
        line["stage_judge_faithfulness_p50"] = metrics_summary.get("stage_judge_faithfulness_p50")
        line["stage_judge_faithfulness_p95"] = metrics_summary.get("stage_judge_faithfulness_p95")
        line["stage_judge_quality_p50"] = metrics_summary.get("stage_judge_quality_p50")
        line["stage_judge_quality_p95"] = metrics_summary.get("stage_judge_quality_p95")
        line["stage_end_to_end_p50"] = metrics_summary.get("stage_end_to_end_p50")
        line["stage_end_to_end_p95"] = metrics_summary.get("stage_end_to_end_p95")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _build_layer2_summary(judge_results: list) -> dict:
    valid = [jr for jr in judge_results if jr.faithfulness is not None]
    n = len(valid) if valid else 0
    return {
        "num_valid": n,
        "num_total": len(judge_results),
        "faithfulness_avg": sum(j.faithfulness or 0 for j in valid) / n if n else None,
        "answer_relevancy_avg": sum(j.answer_relevancy or 0 for j in valid) / n if n else None,
        "context_precision_avg": sum(j.context_precision or 0 for j in valid) / n if n else None,
        "context_recall_avg": sum(j.context_recall or 0 for j in valid) / n if n else None,
        "answer_correctness_avg": sum(j.answer_correctness or 0 for j in valid) / n if n else None,
        "verdict_distribution": _count_verdicts(judge_results),
        "judge_errors": sum(1 for jr in judge_results if jr.judge_error),
        "generator_errors": sum(1 for jr in judge_results if jr.generator_error),
    }


def _count_verdicts(judge_results: list) -> dict:
    counts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
    for jr in judge_results:
        v = jr.verdict if jr.verdict in counts else "error"
        counts[v] += 1
    return counts


def _get_git_commit() -> str:
    """获取当前 git commit 短哈希。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def build_run_info(benchmark_path: str, run_mode: str = "retrieval") -> dict:
    """构建 run_info 配置快照。"""
    from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K as tk, LLM_MODEL_ID
    return {
        "timestamp": datetime.now().isoformat(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": tk,
        "model_id": LLM_MODEL_ID,
        "git_commit": _get_git_commit(),
        "benchmark": benchmark_path,
        "test_mode": run_mode,
    }


def _write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
