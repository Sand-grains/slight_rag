"""Layer 1 检索评估编排：两轮评估 + 诊断分类，纯数学指标，不调 LLM。

核心特性：
    - 第一轮：计算 6 项 IR 指标 + 诊断分类（accept / recall_miss / file_miss / ranking_miss / low_precision）
    - 第二轮：对 low_precision 样本追加检测（判定是检索问题还是标注问题）
    - CANDIDATE_K = TOP_K * 2，候选池比最终输出大，用于 ranking_miss 检测
    - 聚合输出 LayerOutput（逐 query 结果 + 聚合指标 + 按 category/difficulty 分组）

用法示例::

    from eval.core.retrieval_layer import run_retrieval_eval
    output = run_retrieval_eval(retriever, benchmark_items)
    print(output.aggregate["recall_at_k"])  # → 0.8256

公共接口：
    - RetrievalEvalResult: 单条 query 的检索评估结果
    - LayerOutput: 完整 Layer 1 输出（results + aggregate + by_category + by_difficulty）
    - run_retrieval_eval: 执行完整两轮检索评估
"""

import statistics
from dataclasses import dataclass, field

from eval.core.benchmark import BenchmarkItem
from eval.core.calculator.metrics import (
    recall_at_k, precision_at_k, hit_at_k,
    mrr, avg_precision, ndcg_at_k,
)
from retrieval.retriever import Retriever
from retrieval.generator import _build_context
from config import TOP_K


@dataclass
class RetrievalEvalResult:
    """单条 query 的 Layer 1 评估结果。"""
    query_id: str
    query: str
    category: str
    difficulty: str
    recall_at_k: float
    precision_at_k: float
    hit_at_k: int
    mrr: float
    map_at_k: float
    ndcg_at_k: float
    diagnosis: str  # recall_miss / file_miss / ranking_miss / low_precision / accept
    final_chunk_ids: list[str] = field(default_factory=list)
    candidate_chunk_ids: list[str] = field(default_factory=list)
    retrieved_files: list[str] = field(default_factory=list)
    query_rewritten: str = ""
    query_rewritten_flag: str = "原"  # "原" 或 "rewritten"
    final_context_text: str = ""


@dataclass
class LayerOutput:
    """Layer 1 完整输出。"""
    results: list[RetrievalEvalResult]
    aggregate: dict  # 全局聚合指标
    by_category: dict[str, dict]  # category → 聚合指标
    by_difficulty: dict[str, dict]  # difficulty → 聚合指标


CANDIDATE_K = TOP_K * 2


def run_retrieval_eval(retriever: Retriever, items: list[BenchmarkItem]) -> LayerOutput:
    """执行 Layer 1 检索评估，含两轮 low_precision 判定。

    Returns:
        LayerOutput: 包含逐 query 结果、聚合指标、分组统计。
    """
    results = _first_pass(retriever, items)
    results = _second_pass_low_precision(results)
    aggregate = _aggregate(results)
    by_category = _group_by(results, key=lambda r: r.category)
    by_difficulty = _group_by(results, key=lambda r: r.difficulty)
    return LayerOutput(
        results=results,
        aggregate=aggregate,
        by_category=by_category,
        by_difficulty=by_difficulty,
    )


def _first_pass(retriever: Retriever, items: list[BenchmarkItem]) -> list[RetrievalEvalResult]:
    """第一轮：计算指标。ranking_miss 通过 candidate 级检索判定。"""
    results = []
    for item in items:
        # 用 top_k * 2 做候选检索，前 TOP_K 为 final，剩余为 candidate-only
        all_chunks = retriever.retrieve(item.query, top_k=CANDIDATE_K)
        final_chunks = all_chunks[:TOP_K]
        candidate_only = all_chunks[TOP_K:]

        relevant_ids = set(item.expected_chunk_ids)
        retrieved_ids = [c.chunk_id for c in final_chunks]
        candidate_ids = [c.chunk_id for c in all_chunks]
        retrieved_files = list({c.origin_metadata.title + c.origin_metadata.doc_type for c in final_chunks})

        has_intersection = any(rid in relevant_ids for rid in retrieved_ids)
        has_candidate_intersection = any(rid in relevant_ids for rid in candidate_ids)
        expected_files_stems = {_stem(f) for f in item.expected_files}
        retrieved_files_stems = {_stem(f) for f in retrieved_files}
        file_miss_flag = not expected_files_stems & retrieved_files_stems if expected_files_stems else False

        if not has_intersection:
            if has_candidate_intersection:
                diagnosis = "ranking_miss"
            elif file_miss_flag:
                diagnosis = "file_miss"
            else:
                diagnosis = "recall_miss"
        else:
            diagnosis = "pending"  # 等第二轮 low_precision 判定

        results.append(RetrievalEvalResult(
            query_id=item.query_id,
            query=item.query,
            category=item.category,
            difficulty=item.difficulty,
            recall_at_k=recall_at_k(retrieved_ids, relevant_ids, TOP_K),
            precision_at_k=precision_at_k(retrieved_ids, relevant_ids, TOP_K),
            hit_at_k=hit_at_k(retrieved_ids, relevant_ids, TOP_K),
            mrr=mrr(retrieved_ids, relevant_ids),
            map_at_k=avg_precision(retrieved_ids, relevant_ids, TOP_K),
            ndcg_at_k=ndcg_at_k(retrieved_ids, item.relevance, TOP_K),
            diagnosis=diagnosis,
            final_chunk_ids=retrieved_ids,
            candidate_chunk_ids=candidate_ids,
            retrieved_files=retrieved_files,
            query_rewritten=item.query,
            query_rewritten_flag="原",
            final_context_text=_build_context(final_chunks),
        ))
    return results


def _second_pass_low_precision(results: list[RetrievalEvalResult]) -> list[RetrievalEvalResult]:
    """第二轮：按 category 分组计算噪声比基线，标记 low_precision。"""
    # 按 category 分组收集噪声比（1 - precision）
    by_cat: dict[str, list[float]] = {}
    for r in results:
        noise_ratio = 1.0 - r.precision_at_k
        by_cat.setdefault(r.category, []).append(noise_ratio)

    # 计算每组的噪声比中位数作为基线
    baseline = {cat: statistics.median(vals) if vals else 0.0 for cat, vals in by_cat.items()}

    for r in results:
        if r.diagnosis != "pending":
            continue
        cat_baseline = baseline.get(r.category, 0.0)
        noise_ratio = 1.0 - r.precision_at_k
        if cat_baseline > 0 and noise_ratio > cat_baseline * 1.5:
            r.diagnosis = "low_precision"
        elif r.hit_at_k:
            r.diagnosis = "accept"
        else:
            r.diagnosis = "recall_miss"

    return results


def _aggregate(results: list[RetrievalEvalResult]) -> dict:
    """计算全局聚合指标（均值）。"""
    if not results:
        return {}
    return {
        "num_queries": len(results),
        "recall_at_k": statistics.mean(r.recall_at_k for r in results),
        "precision_at_k": statistics.mean(r.precision_at_k for r in results),
        "hit_at_k": statistics.mean(r.hit_at_k for r in results),
        "mrr": statistics.mean(r.mrr for r in results),
        "map_at_k": statistics.mean(r.map_at_k for r in results),
        "ndcg_at_k": statistics.mean(r.ndcg_at_k for r in results),
        "diagnosis_distribution": _diagnosis_distribution(results),
    }


def _group_by(results: list[RetrievalEvalResult], key) -> dict[str, dict]:
    """按给定 key 函数分组并计算每组聚合指标。"""
    groups: dict[str, list[RetrievalEvalResult]] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)
    return {k: _aggregate(v) for k, v in groups.items()}


def _diagnosis_distribution(results: list[RetrievalEvalResult]) -> dict[str, int]:
    """统计五种诊断分类的计数。"""
    dist = {"recall_miss": 0, "file_miss": 0, "ranking_miss": 0, "low_precision": 0, "accept": 0}
    for r in results:
        if r.diagnosis in dist:
            dist[r.diagnosis] += 1
    return dist


def _stem(filepath: str) -> str:
    """提取文件名主干（不含扩展名），用于文件级匹配。"""
    import os
    return os.path.splitext(os.path.basename(filepath))[0].lower()
