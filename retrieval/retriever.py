from typing import List
from indexing.chunk import Chunk
from retrieval.embedding import embed
from retrieval.store import VectorStore
from config import TOP_K


class Retriever:
    """检索层：封装 query 向量化 + 向量库检索，输入自然语言问题，返回相关 chunk 列表"""

    def __init__(self, store: VectorStore):
        self.store = store                                                    # 持有已索引完成的向量库引用

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        """query 向量化 → 余弦相似度检索 → 返回 top_k 个最相关 chunk"""
        query_vector = embed([query])[0]                                      # 用户问题向量化
        return self.store.search(query_vector, top_k)                         # 委托 VectorStore 做相似度检索
