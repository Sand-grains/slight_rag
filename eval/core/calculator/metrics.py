"""IR 评估指标：Recall、Precision、Hit、MRR、MAP、DCG、NDCG。

核心特性：
    - 全部为纯函数，零外部依赖（仅 math 标准库）
    - 每个函数输入 retrieved_ids + relevant_ids / relevance_map + k，返回 float 或 int
    - 处理空输入的边界情况（空列表 / 空集合返回 0.0）

用法示例::

    from eval.core.calculator.metrics import recall_at_k, ndcg_at_k
    r = recall_at_k(["c1", "c2", "c3"], {"c1", "c4"}, k=3)  # → 0.5

公共接口：
    - recall_at_k: |retrieved[:k] ∩ relevant| / |relevant|
    - precision_at_k: |retrieved[:k] ∩ relevant| / k
    - hit_at_k: 二值，前 k 个是否至少命中一个
    - mrr: 1 / 第一个相关 chunk 的排名
    - average_precision: 各相关位置处 Precision 的均值
    - dcg_at_k: 折损累积增益（多级相关度 + 位置折损）
    - ndcg_at_k: 归一化 DCG（DCG / IDCG）
"""

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """召回率：前 K 个检索结果中命中相关 chunk 的比例。

    |retrieved[:k] ∩ relevant| / |relevant|
    """
    if not relevant_ids:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """精确率：前 K 个检索结果中相关 chunk 的占比。

    |retrieved[:k] ∩ relevant| / k
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / k


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> int:
    """命中率：前 K 个检索结果是否至少命中一个相关 chunk（二值）。

    1 if any retrieved[:k] in relevant, else 0.
    """
    if not retrieved_ids or not relevant_ids:
        return 0
    return 1 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """平均倒数排名：第一个相关 chunk 排名的倒数，找不到返回 0。

    1 / rank_of_first_relevant, 0 if none found. Rank starts at 1.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """平均精度：各相关 chunk 召回位置处 Precision 的均值。

    Mean of P@i at each rank where a relevant item appears.
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_ids:
            hits += 1
            precision_sum += hits / i
    if hits == 0:
        return 0.0
    return precision_sum / len(relevant_ids)


def dcg_at_k(ids: list[str], relevance_map: dict[str, int], k: int) -> float:
    """折损累积增益：考虑排序位置和多级相关度的累积得分。

    Σ (2^rel_i - 1) / log2(i+1) for i from 1..k.
    """
    dcg = 0.0
    for i, rid in enumerate(ids[:k], start=1):
        rel = relevance_map.get(rid, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, int], k: int) -> float:
    """归一化折损累积增益：DCG 除以理想排序下的 DCG，消除 query 难度差异。

    DCG@K / IDCG@K. IDCG computed from ideal ordering (descending relevance).
    """
    dcg = dcg_at_k(retrieved_ids, relevance_map, k)
    if dcg == 0.0:
        return 0.0
    ideal_rels = sorted(relevance_map.values(), reverse=True)
    ideal_retrieved = [f"ideal_{i}" for i in range(len(ideal_rels))]
    ideal_map = {f"ideal_{i}": rel for i, rel in enumerate(ideal_rels)}
    idcg = dcg_at_k(ideal_retrieved, ideal_map, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
