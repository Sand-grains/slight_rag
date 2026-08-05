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
    - avg_precision: 各相关位置处 Precision 的均值
    - dcg_at_k: 折损累积增益（多级相关度 + 位置折损）
    - ndcg_at_k: 归一化 DCG（DCG / IDCG）
"""

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """召回率（查全率）：前 K 个结果中被命中的相关 chunk 占全部相关 chunk 的比例。

    分母是"全部相关 chunk 数"（|relevant|），衡量"该查到的查到了几分"——
    与 precision_at_k 的区别在分母：recall 除以相关总数，precision 除以返回数 K。

    |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved_ids: 检索返回的 chunk_id 列表（按得分降序）。
        relevant_ids: 相关 chunk_id 集合。
        k: 仅将前 K 个纳入计算。

    Returns:
        float：召回率；相关集合为空时返回 0.0。
    """
    if not relevant_ids:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """精确率（查准率）：返回结果中相关 chunk 占返回总数 K 的比例。

    分母是"返回条数 K"（非相关总数），衡量"查到的里面相关占几分"——
    与 recall 的关键区别在分母：precision 除以返回数 K，recall 除以相关总数。

    |retrieved[:k] ∩ relevant| / k

    Args:
        retrieved_ids: 检索返回的 chunk_id 列表（按得分降序）。
        relevant_ids: 相关 chunk_id 集合。
        k: 仅将前 K 个纳入计算。

    Returns:
        float：精确率；k <= 0 或检索结果为空时返回 0.0。
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / k


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> int:
    """命中率：前 K 个检索结果是否至少命中一个相关 chunk（二值）。

    1 if any retrieved[:k] in relevant, else 0.

    Args:
        retrieved_ids: 检索返回的 chunk_id 列表（按得分降序）。
        relevant_ids: 相关 chunk_id 集合。
        k: 只考察前 K 个。

    Returns:
        int：前 K 个有命中返回 1，否则 0。
    """
    if not retrieved_ids or not relevant_ids:
        return 0
    return 1 if any(retrieved_id in relevant_ids for retrieved_id in retrieved_ids[:k]) else 0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """平均倒数排名：第一个相关 chunk 排名的倒数，找不到返回 0。

    1 / rank_of_first_relevant, 0 if none found. Rank starts at 1.

    Args:
        retrieved_ids: 检索返回的 chunk_id 列表（按得分降序）。
        relevant_ids: 相关 chunk_id 集合。

    Returns:
        float：MRR 值；无命中返回 0.0。
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0
    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def avg_precision(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """平均精度：各相关 chunk 召回位置处 Precision 的均值。

    Mean of P@i at each rank where a relevant item appears.

    Args:
        retrieved_ids: 检索返回的 chunk_id 列表（按得分降序）。
        relevant_ids: 相关 chunk_id 集合。
        k: 只考察前 K 个。

    Returns:
        float：平均精度；无相关命中时返回 0.0。
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, retrieved_id in enumerate(retrieved_ids[:k], start=1):
        if retrieved_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    if hits == 0:
        return 0.0
    return precision_sum / len(relevant_ids)


def dcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, int], k: int) -> float:
    """折损累积增益：考虑排序位置和多级相关度的累积得分。

    Σ (2^rel_i - 1) / log2(i+1) for i from 1..k.

    Args:
        retrieved_ids: 检索返回的 chunk_id 列表（按得分降序）。
        relevance_map: chunk_id → 相关度分数（0/1/2/3）。
        k: 只考察前 K 个。

    Returns:
        float：DCG 值。
    """
    dcg = 0.0
    for rank, retrieved_id in enumerate(retrieved_ids[:k], start=1):
        relevance = relevance_map.get(retrieved_id, 0)
        if relevance > 0:
            dcg += (2 ** relevance - 1) / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, int], k: int) -> float:
    """归一化折损累积增益：DCG 除以理想排序下的 DCG，消除 query 难度差异。

    DCG@K / IDCG@K. IDCG computed from ideal ordering (descending relevance).

    Args:
        retrieved_ids: 检索返回的 chunk_id 列表（按得分降序）。
        relevance_map: chunk_id → 相关度分数（0/1/2/3）。
        k: 只考察前 K 个。

    Returns:
        float：NDCG 值；DCG 或 IDCG 为 0 时返回 0.0。
    """
    dcg = dcg_at_k(retrieved_ids, relevance_map, k)
    if dcg == 0.0:
        return 0.0
    ideal_relevances = sorted(relevance_map.values(), reverse=True)
    ideal_retrieved = [f"ideal_{index}" for index in range(len(ideal_relevances))]
    ideal_map = {f"ideal_{index}": relevance for index, relevance in enumerate(ideal_relevances)}
    idcg = dcg_at_k(ideal_retrieved, ideal_map, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
