"""Elasticsearch 客户端：索引管理 + BM25 全文检索。"""
from elasticsearch import Elasticsearch
from config import ES_CONNECTION_URI, ES_INDEX


class ESClient:
    def __init__(self, es_url=ES_CONNECTION_URI, index_name=ES_INDEX):
        self._es = Elasticsearch(es_url)
        self._index = index_name

    def create_index(self):
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

    def delete_index(self):
        self._es.indices.delete(index=self._index, ignore=[400, 404])

    def insert_batch(self, chunks):
        """批量索引 chunk，chunk_id 为文档 _id（幂等）。"""
        from elasticsearch.helpers import bulk
        actions = [
            {
                "_index": self._index,
                "_id": c.chunk_id,
                "_source": {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "content": c.content},
            }
            for c in chunks
        ]
        if actions:
            bulk(self._es, actions, refresh=True)

    def search_bm25(self, query, top_k=10):
        """BM25 全文检索，返回 [(chunk_id, score), ...]."""
        result = self._es.search(
            index=self._index,
            query={"match": {"content": query}},
            size=top_k,
            source=["chunk_id"],
        )
        return [(hit["_source"]["chunk_id"], hit["_score"]) for hit in result["hits"]["hits"]]

    def delete_by_doc_id(self, doc_id):
        self._es.delete_by_query(
            index=self._index,
            body={"query": {"term": {"doc_id": doc_id}}},
            ignore=[400, 404],
        )
