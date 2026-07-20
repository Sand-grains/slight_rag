from typing import List
from indexing.chunk import Chunk
from retrieval.embedding import embed
from retrieval.store import VectorStore
from config import TOP_K

RRF_K = 60  # RRF 平滑常量，对排名变化不敏感


class Retriever:
    """检索层：双路检索（稠密 + 稀疏） → RRF 融合 → 返回 top_k chunks"""

    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        query_vector = embed([query])[0]
        candidate_k = top_k * 2  # 各路多取候选给 RRF 留余量

        # 稠密检索
        dense_results = self.store.search_dense(query_vector, top_k=candidate_k)
        # 稀疏检索
        sparse_results = self.store.search_sparse(query, top_k=candidate_k)

        # RRF 融合: score(d) = 1/(k + rank_d) + 1/(k + rank_s)
        rrf_scores = {}
        for rank, c in enumerate(dense_results, start=1):
            rrf_scores[c.chunk_id] = rrf_scores.get(c.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, c in enumerate(sparse_results, start=1):
            rrf_scores[c.chunk_id] = rrf_scores.get(c.chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        # 按 RRF 分数降序，取 top_k
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        # 保留密集检索结果中的 Chunk 对象用于最终返回
        id_to_chunk = {c.chunk_id: c for c in dense_results + sparse_results}
        return [id_to_chunk[cid] for cid in sorted_ids]
