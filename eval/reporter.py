"""聚合、分组、诊断分类、JSON 输出。"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from eval.core.retrieval_layer import Layer1Output, RetrievalEvalResult


def generate_report(output: Layer1Output, results_dir: str, run_info: dict) -> str:
    """生成完整eval报告：写入 summary.json / per_query.json / failures.json / run_info.json。

    Args:
        output: Layer 1 评估完整输出
        results_dir: 结果输出目录（如 eval/results/2026-07-23_143000）
        run_info: 运行配置快照（chunk_size, top_k, model, git_commit 等）

    Returns:
        结果目录路径
    """
    os.makedirs(results_dir, exist_ok=True)

    # per_query.json: query_id → 完整 trace
    per_query = {r.query_id: _serialize_result(r) for r in output.results}
    _write_json(os.path.join(results_dir, "per_query.json"), per_query)

    # failures.json: diagnosis != "accept" 的子集
    failures = {r.query_id: _serialize_result(r) for r in output.results if r.diagnosis != "accept"}
    _write_json(os.path.join(results_dir, "failures.json"), failures)

    # summary.json: 聚合 + 分组
    summary = {
        "aggregate": output.aggregate,
        "by_category": output.by_category,
        "by_difficulty": output.by_difficulty,
    }
    _write_json(os.path.join(results_dir, "summary.json"), summary)

    # run_info.json: 配置快照
    _write_json(os.path.join(results_dir, "run_info.json"), run_info)

    # history.jsonl: 追加一行
    _append_history(results_dir, output)

    # 终端输出
    _print_summary(output)

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


def _print_summary(output: Layer1Output):
    """终端打印聚合结果。"""
    agg = output.aggregate
    if not agg:
        print("无评估结果。")
        return

    print("\n" + "=" * 60)
    print("Layer 1 检索评估结果")
    print("=" * 60)
    print(f"  Query 数:        {agg['num_queries']}")
    print(f"  Recall@{_k()}:      {agg['recall_at_k']:.4f}")
    print(f"  Precision@{_k()}:   {agg['precision_at_k']:.4f}")
    print(f"  Hit@{_k()}:         {agg['hit_at_k']:.4f}")
    print(f"  MRR:             {agg['mrr']:.4f}")
    print(f"  MAP@{_k()}:         {agg['map_at_k']:.4f}")
    print(f"  NDCG@{_k()}:        {agg['ndcg_at_k']:.4f}")
    print(f"\n  诊断分布:")
    for label, count in agg["diagnosis_distribution"].items():
        print(f"    {label}: {count}")
    print("=" * 60 + "\n")


def _append_history(results_dir: str, output: Layer1Output):
    """向 history.jsonl 追加一行运行摘要。"""
    from config import TOP_K
    history_dir = os.path.join(os.path.dirname(results_dir), "check")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, "history.jsonl")

    agg = output.aggregate
    run_id = os.path.basename(results_dir)
    line = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "test_mode": "retrieval",
        "benchmark": "benchmark_private.json",
        "num_queries": agg.get("num_queries", 0),
        "recall_at_k": round(agg.get("recall_at_k", 0.0), 4),
        "precision_at_k": round(agg.get("precision_at_k", 0.0), 4),
        "hit_at_k": round(agg.get("hit_at_k", 0.0), 4),
        "mrr": round(agg.get("mrr", 0.0), 4),
        "map_at_k": round(agg.get("map_at_k", 0.0), 4),
        "ndcg_at_k": round(agg.get("ndcg_at_k", 0.0), 4),
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _get_git_commit() -> str:
    """获取当前 git commit 短哈希。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def build_run_info(benchmark_path: str) -> dict:
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
    }


def _k() -> int:
    from config import TOP_K
    return TOP_K


def _write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
