"""Elasticsearch 客户端：索引管理 + BM25 全文检索（IK 分词器）。

核心特性：
    - create_index 建 IK 分词映射（ik_max_word 索引 / ik_smart 搜索）
    - insert_batch 以 chunk_id 为文档 _id（幂等），bulk 批量写入
    - search_bm25 全文检索，返回 (chunk_id, score) 对，供 IndexStore 落父块缓存
    - delete_by_doc_id 按 doc_id 级联删除（rollback_doc 的一环）

公共接口：
    - ESClient: Elasticsearch 客户端（父块索引 + BM25 稀疏检索）
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from elasticsearch import Elasticsearch
from infra.config import ES_CONNECTION_URI, ES_INDEX

if TYPE_CHECKING:
    from indexing.chunk import Chunk


class ESClient:
    """Elasticsearch 客户端：父块全文索引 + BM25 稀疏检索。"""

    def __init__(self, es_url: str = ES_CONNECTION_URI, index_name: str = ES_INDEX):
        self._es = Elasticsearch(es_url)
        self._index = index_name

    def create_index(self) -> None:
        """建索引（chunk_id/doc_id 为 keyword，content 走 IK 分词），已存在则忽略。"""
        body = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    },
                }
            },
        }
        self._es.indices.create(index=self._index, body=body, ignore=400)

    def delete_index(self) -> None:
        """删除索引（用于重建/清理）。"""
        self._es.indices.delete(index=self._index, ignore=[400, 404])

    def insert_batch(self, chunks: list[Chunk]) -> None:
        """批量索引 chunk，chunk_id 为文档 _id（幂等）。

        Args:
            chunks: 父块列表（父块是 BM25 检索单元）。
        """
        from elasticsearch.helpers import bulk
        actions = [
            {
                "_index": self._index,
                "_id": chunk.chunk_id,
                "_source": {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "content": chunk.content},
            }
            for chunk in chunks
        ]
        if actions:
            bulk(self._es, actions, refresh=True)

    def search_bm25(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """BM25 全文检索，返回命中父块及其分数。

        Args:
            query: 查询文本（ES 用 ik_smart 粗粒度分词）。
            top_k: 返回的最大条数。

        Returns:
            list[(chunk_id, score)]：按 BM25 分数降序命中的父块。
        """
        result = self._es.search(
            index=self._index,
            query={"match": {"content": query}},
            size=top_k,
            source=["chunk_id"],
        )
        return [(hit["_source"]["chunk_id"], hit["_score"]) for hit in result["hits"]["hits"]]

    def delete_by_doc_id(self, doc_id: str) -> None:
        """按 doc_id 删除全部文档（rollback_doc 的一环）。"""
        self._es.delete_by_query(
            index=self._index,
            body={"query": {"term": {"doc_id": doc_id}}},
            ignore=[400, 404],
        )
