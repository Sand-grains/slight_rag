"""评估报告生成：JSON 文件写入 + history.jsonl 追加。

核心特性：
    - generate_report() 一站式写入：per_query.json / failures.json / summary.json / run_info.json / history.jsonl
    - per_query.json 自动合并 Layer 1（检索指标 + 诊断）+ Layer 2（Judge 分数 + verdict + 阶段延迟）
    - _build_history() 向 eval/results/check/history.jsonl 追加一行运行摘要（跨运行索引）
    - _build_layer2_summary() 计算 Layer 2 聚合（5 项均值 + verdict 分布 + judge/generator errors）
    - 终端输出职责已移交 MonitorPanel，reporter 仅做文件 I/O
    - build_run_info() 含 git commit 快照 + 配置快照

用法示例::

    from eval.reporter import generate_report, build_run_info
    run_info = build_run_info("bench.json", run_mode="full")
    generate_report(layer1_output, results_dir, run_info, judge_results=judge_results, metrics_summary=metrics.summary_dict())

公共接口：
    - generate_report: 生成完整 eval 报告（5 个文件）
    - build_run_info: 构建运行配置快照 dict
"""

import os
from datetime import datetime

from eval.core.retrieval.retrieval_layer import LayerOutput, RetrievalEvalResult
from eval.utils import append_jsonl, get_git_commit, write_json


def generate_report(output: LayerOutput, results_dir: str, run_info: dict,
                    judge_results: list | None = None,
                    metrics_summary: dict | None = None) -> str:
    """生成完整 eval 报告：写入 summary.json / per_query.json / failures.json / run_info.json / history.jsonl。

    Args:
        output: Layer 1 评估完整输出。
        results_dir: 结果输出目录。
        run_info: 运行配置快照。
        judge_results: Layer 2 JudgeResult 列表（可选，full mode 时传入）。
        metrics_summary: MonitorMetrics.summary_dict()（可选，full mode 时传入）。

    Returns:
        str：结果输出目录。
    """
    os.makedirs(results_dir, exist_ok=True)

    # per_query.json: query_id → 完整 trace
    per_query = {result.query_id: _serialize_result(result) for result in output.results}
    write_json(os.path.join(results_dir, "per_query.json"), per_query)

    # failures.json: diagnosis != "accept" 的子集
    failures = {result.query_id: _serialize_result(result) for result in output.results if result.diagnosis != "accept"}
    write_json(os.path.join(results_dir, "failures.json"), failures)

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
    write_json(os.path.join(results_dir, "summary.json"), summary)

    # run_info.json: 配置快照
    write_json(os.path.join(results_dir, "run_info.json"), run_info)

    # per_query.json: 合并 Layer 2 数据
    if judge_results is not None:
        for judge_result in judge_results:
            if judge_result.query_id in per_query:
                per_query[judge_result.query_id].update(_serialize_judge_result(judge_result))
        write_json(os.path.join(results_dir, "per_query.json"), per_query)

    # history.jsonl: 追加一行
    _build_history(results_dir, output, run_info, judge_results, metrics_summary)

    return results_dir


def _serialize_result(result: RetrievalEvalResult) -> dict:
    """将单条 Layer 1 结果序列化为可写入 JSON 的 dict。"""
    return {
        "query_id": result.query_id,
        "query": result.query,
        "category": result.category,
        "difficulty": result.difficulty,
        "recall_at_k": result.recall_at_k,
        "precision_at_k": result.precision_at_k,
        "hit_at_k": result.hit_at_k,
        "mrr": result.mrr,
        "map_at_k": result.map_at_k,
        "ndcg_at_k": result.ndcg_at_k,
        "child_hit_at_k": result.child_hit_at_k,
        "child_recall_at_k": result.child_recall_at_k,
        "child_annotated": result.child_annotated,
        "diagnosis": result.diagnosis,
        "final_chunk_ids": result.final_chunk_ids,
        "candidate_chunk_ids": result.candidate_chunk_ids,
        "retrieved_files": result.retrieved_files,
        "query_rewritten": result.query_rewritten,
        "query_rewritten_flag": result.query_rewritten_flag,
        "final_context_text": result.final_context_text,
    }


def _serialize_judge_result(judge_result) -> dict:
    """序列化 JudgeResult 中需写入 per_query.json 的字段。"""
    return {
        "faithfulness": judge_result.faithfulness,
        "answer_relevancy": judge_result.answer_relevancy,
        "context_precision": judge_result.context_precision,
        "context_recall": judge_result.context_recall,
        "answer_correctness": judge_result.answer_correctness,
        "verdict": judge_result.verdict,
        "parse_error": judge_result.parse_error,
        "judge_error": judge_result.judge_error,
        "generator_error": judge_result.generator_error,
        "retrieve_ms": judge_result.retrieve_ms,
        "generate_ms": judge_result.generate_ms,
        "judge_faithfulness_ms": judge_result.judge_faithfulness_ms,
        "judge_quality_ms": judge_result.judge_quality_ms,
    }


def _build_history(results_dir: str, output: LayerOutput, run_info: dict,
                    judge_results: list | None = None,
                    metrics_summary: dict | None = None) -> None:
    """向 history.jsonl 追加一行运行摘要。

    Args:
        results_dir: 本次运行结果目录。
        output: Layer 1 评估完整输出。
        run_info: 运行配置快照。
        judge_results: Layer 2 JudgeResult 列表（可选）。
        metrics_summary: MonitorMetrics.summary_dict()（可选）。
    """
    from config import _PROJECT_ROOT
    history_dir = str(_PROJECT_ROOT / "eval" / "results" / "check")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, "history.jsonl")

    aggregate = output.aggregate
    run_id = os.path.basename(results_dir)
    line = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "test_mode": run_info.get("test_mode", "retrieval"),
        "benchmark": run_info.get("benchmark", ""),
        "num_queries": aggregate.get("num_queries", 0),
        "recall_at_k": round(aggregate.get("recall_at_k", 0.0), 4),
        "precision_at_k": round(aggregate.get("precision_at_k", 0.0), 4),
        "hit_at_k": round(aggregate.get("hit_at_k", 0.0), 4),
        "mrr": round(aggregate.get("mrr", 0.0), 4),
        "map_at_k": round(aggregate.get("map_at_k", 0.0), 4),
        "ndcg_at_k": round(aggregate.get("ndcg_at_k", 0.0), 4),
    }
    if aggregate.get("num_child_annotated", 0) > 0:
        line["child_hit_at_k"] = round(aggregate.get("child_hit_at_k", 0.0), 4)
        line["child_recall_at_k"] = round(aggregate.get("child_recall_at_k", 0.0), 4)
        line["num_child_annotated"] = aggregate.get("num_child_annotated", 0)
    if judge_results is not None:
        layer2_summary = _build_layer2_summary(judge_results)
        line["faithfulness_avg"] = layer2_summary.get("faithfulness_avg")
        line["answer_relevancy_avg"] = layer2_summary.get("answer_relevancy_avg")
        line["context_precision_avg"] = layer2_summary.get("context_precision_avg")
        line["context_recall_avg"] = layer2_summary.get("context_recall_avg")
        line["answer_correctness_avg"] = layer2_summary.get("answer_correctness_avg")
    if metrics_summary is not None:
        line["generator_cache_hit_rate"] = metrics_summary.get("generator_cache_hit_rate")
        line["judge_cache_hit_rate"] = metrics_summary.get("judge_cache_hit_rate")
        line["estimated_cost"] = metrics_summary.get("estimated_cost")
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
    append_jsonl(history_path, line)


def _build_layer2_summary(judge_results: list) -> dict:
    """计算 Layer 2 聚合：5 项均值 + verdict 分布 + judge/generator errors。"""
    valid = [judge_result for judge_result in judge_results if judge_result.faithfulness is not None]
    count = len(valid) if valid else 0
    return {
        "num_valid": count,
        "num_total": len(judge_results),
        "faithfulness_avg": sum(judge_result.faithfulness or 0 for judge_result in valid) / count if count else None,
        "answer_relevancy_avg": sum(judge_result.answer_relevancy or 0 for judge_result in valid) / count if count else None,
        "context_precision_avg": sum(judge_result.context_precision or 0 for judge_result in valid) / count if count else None,
        "context_recall_avg": sum(judge_result.context_recall or 0 for judge_result in valid) / count if count else None,
        "answer_correctness_avg": sum(judge_result.answer_correctness or 0 for judge_result in valid) / count if count else None,
        "verdict_distribution": _count_verdicts(judge_results),
        "judge_errors": sum(1 for judge_result in judge_results if judge_result.judge_error),
        "generator_errors": sum(1 for judge_result in judge_results if judge_result.generator_error),
    }


def _count_verdicts(judge_results: list) -> dict:
    """统计 pass / partial / fail / error 各 verdict 数量。"""
    counts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
    for judge_result in judge_results:
        verdict = judge_result.verdict if judge_result.verdict in counts else "error"
        counts[verdict] += 1
    return counts


def build_run_info(benchmark_path: str, run_mode: str = "retrieval") -> dict:
    """构建 run_info 配置快照。

    Args:
        benchmark_path: benchmark 文件路径。
        run_mode: 运行模式（retrieval / full）。

    Returns:
        dict：含分块参数、模型 id、git commit、benchmark 路径的配置快照。
    """
    from config import CHILD_CHUNK_SIZE, CHILD_OVERLAP, TOP_K, LLM_MODEL_ID
    return {
        "timestamp": datetime.now().isoformat(),
        "chunk_size": CHILD_CHUNK_SIZE,
        "chunk_overlap": CHILD_OVERLAP,
        "top_k": TOP_K,
        "model_id": LLM_MODEL_ID,
        "git_commit": get_git_commit(),
        "benchmark": benchmark_path,
        "test_mode": run_mode,
    }


