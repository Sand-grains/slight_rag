"""Milvus 客户端：collection 管理 + Dense 向量检索（IP 度量 = 余弦相似度）。"""
from pymilvus import MilvusClient
from config import MILVUS_CONNECTION_URI, MILVUS_COLLECTION, MILVUS_HNSW_EF


class MilvusClientWrapper:
    def __init__(self, uri=MILVUS_CONNECTION_URI, collection_name=MILVUS_COLLECTION):
        self._client = MilvusClient(uri=uri)
        self._collection = collection_name

    def ensure_collection(self):
        if self._client.has_collection(self._collection):
            self._client.load_collection(self._collection)
            return
        self._client.create_collection(
            collection_name=self._collection,
            dimension=1024,
            metric_type="IP",
            primary_field_name="chunk_id",
            id_type="string",
            max_length=256,
        )
        if "vector" not in self._client.list_indexes(self._collection):
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_name="vector_hnsw",
                index_type="HNSW",
                metric_type="IP",
                params={"M": 16, "efConstruction": 200},
            )
            self._client.create_index(
                collection_name=self._collection,
                index_params=index_params,
            )
        self._client.load_collection(self._collection)

    def drop_collection(self):
        if self._client.has_collection(self._collection):
            self._client.drop_collection(self._collection)

    def insert_batch(self, chunks, vectors):
        data = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "content": c.content,
                "chunk_idx": i,
                "vector": vectors[i].tolist(),
            }
            for i, c in enumerate(chunks)
        ]
        self._client.insert(collection_name=self._collection, data=data)

    def search_dense(self, query_vector, top_k=10):
        """Dense 向量检索，返回 [(chunk_id, score), ...]。score 为 IP 距离（越大越相似）。"""
        import numpy as np
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        results = self._client.search(
            collection_name=self._collection,
            data=[query_vector],
            anns_field="vector",
            search_params={"metric_type": "IP", "params": {"ef": MILVUS_HNSW_EF}},
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "content"],
        )
        return [(hit["entity"]["chunk_id"], hit["distance"]) for hit in results[0]]

    def delete_by_doc_id(self, doc_id):
        self._client.delete(collection_name=self._collection, filter=f'doc_id == "{doc_id}"')

    def delete_by_chunk_ids(self, chunk_ids):
        expr = "chunk_id in [" + ",".join(f'"{cid}"' for cid in chunk_ids) + "]"
        self._client.delete(collection_name=self._collection, filter=expr)
