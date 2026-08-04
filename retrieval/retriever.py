"""双路检索 + RRF 融合（父块层）。

核心特性：
    - 稠密检索：BGE-M3 embedding → 余弦相似度，返回**子块**（含 parent_id）
    - 稀疏检索：jieba/IK 分词 → BM25 打分，返回**父块**
    - RRF (Reciprocal Rank Fusion, k=60) 在**父块层**融合：key = parent_id or chunk_id
    - 融合后 index_store.get_parents(top_ids) 统一落到父块（检索/benchmark 单元）
    - 各路多取 candidate_k = top_k * 2 给 RRF 留余量

用法示例::

    from retrieval import Retriever
    from indexing.index_store import IndexStore
    index_store = IndexStore.vector_restore()
    retriever = Retriever(index_store)
    chunks = retriever.retrieve("什么是 RAG", top_k=5)

公共接口：
    - Retriever: 双路检索器，持有 IndexStore 引用
"""
from config import RRF_K, TOP_K
from indexing.chunk import Chunk
from indexing.index_store import IndexStore
from retrieval.embedding import embed


def _parent_key(chunk: Chunk) -> str:
    """RRF 融合键：子块按 parent_id 归并到父块，flat_simple 无 parent_id → 自父。

    Args:
        chunk: 待归并的检索结果 Chunk（子块或父块）。

    Returns:
        str：父块 chunk_id。子块取 metadata.parent_id；无 parent_id（flat_simple）
        时取自身 chunk_id。
    """
    return chunk.metadata.get("parent_id") or chunk.chunk_id


class Retriever:
    """检索层：双路检索（稠密子块 + 稀疏父块） → 父块层 RRF 融合 → 返回 top_k 父块。"""

    def __init__(self, store: IndexStore):
        self.index_store = store

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[Chunk]:
        """单次检索：返回 top_k 个父块 Chunk（RRF 融合后的最终结果）。

        Args:
            query: 查询文本。
            top_k: 返回的父块最大条数。

        Returns:
            list[Chunk]：RRF 融合后按分数降序的父块列表。
        """
        parents, _ = self.retrieve_with_dense_child(query, top_k)
        return parents

    def retrieve_with_dense_child(self, query: str, top_k: int = TOP_K) -> tuple[list[Chunk], list[Chunk]]:
        """混合检索 → RRF 融合 → 返回(父块 top_k, dense 子块候选)。
        dense 子块候选供 child 层指标/诊断使用，检索本身只用父块 top_k。

        这个方法主要用于eval模块,
        保留dense_child是因为测评时要同时衡量两套不同粒度的指标, 评测的子块层指标必须要这份RRF融合前数据

        Args:
            query: 查询文本。
            top_k: 返回的父块最大条数。

        Returns:
            tuple[list[Chunk], list[Chunk]]：(父块 top_k, dense 子块候选列表)。
            子块候选长度 = candidate_k（top_k * 2），是融合前多取的余量。
        """
        query_vector = embed([query])[0]
        candidate_k = top_k * 2  # 各路多取候选给 RRF 留余量

        dense_results = self.index_store.search_dense(query_vector, top_k=candidate_k)
        sparse_results = self.index_store.search_sparse(query, top_k=candidate_k)

        # RRF 融合：score(id) = 1/(k + rank_dense) + 1/(k + rank_sparse)，key 统一到父块
        rrf_scores = {}
        for rank, chunk in enumerate(dense_results, start=1):
            key = _parent_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        for rank, chunk in enumerate(sparse_results, start=1):
            key = _parent_key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)

        # 按 RRF 分数降序取 top_k 父块
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return self.index_store.get_parents(sorted_ids), dense_results
