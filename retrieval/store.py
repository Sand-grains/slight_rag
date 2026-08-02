"""向量存储门面：按 STORAGE_BACKEND 分流 memory（v5 内存）或 external（PgSQL+ES+Milvus）。

外部模式（external）委托三库客户端，内存模式（memory）保留 v5 行为。对外接口统一。
"""
import numpy
import pickle
import jieba
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi
from indexing.chunk import Chunk
from config import TOP_K, VECTOR_CACHE_DIR, STORAGE_BACKEND
from indexing.chunk_ingest import ingest_chunks, cleanup_suspending

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class IndexStore:
    """向量存储门面：对外统一接口，内部按 STORAGE_BACKEND 分流。"""

    def __init__(self):
        self._backend = STORAGE_BACKEND
        if self._backend == "external":
            self._init_external()
        else:
            self._init_memory()

    # ---- memory 模式----

    def _init_memory(self):
        self._chunks: List[Chunk] = []
        self._vectors: numpy.ndarray | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_tokenized: List[List[str]] = []

    # ---- external 模式（Phase 1 新增）----

    def _init_external(self):
        from infra.db.postgres import PgSQLClient
        from infra.search.es import ESClient
        from infra.vector.milvus import MilvusClientWrapper
        self._pgsql = PgSQLClient()
        self._es = ESClient()
        self._milvus = MilvusClientWrapper()
        self._pgsql.ensure_table()
        self._es.create_index()
        self._milvus.ensure_collection()
        cleanup_suspending(self._pgsql, self._es, self._milvus)
        self._chunks_cache: dict[str, Chunk] = {}
        self._load_cache_from_pgsql()

    def _load_cache_from_pgsql(self):
        """从 PgSQL 回填 chunks_cache，用于 vector_restore 后的检索映射。"""
        cur = self._pgsql._conn.cursor()
        cur.execute("SELECT chunk_id, doc_id, content FROM chunks WHERE status = 'indexed'")
        for chunk_id, doc_id, content in cur.fetchall():
            self._chunks_cache[chunk_id] = Chunk(
                chunk_id=chunk_id, doc_id=doc_id, content=content,
            )
        cur.close()

    # ---- 公共属性 ----

    @property
    def chunk_ids(self) -> set[str]:
        if self._backend == "external":
            return set(self._chunks_cache.keys())
        return {c.chunk_id for c in self._chunks}

    @property
    def chunks(self) -> List[Chunk]:
        if self._backend == "external":
            return list(self._chunks_cache.values())
        return list(self._chunks)

    # ---- 公共方法 ----

    def batch_add(self, documents: List[Chunk], vectors: List[List[float]]):
        if self._backend == "external":
            self._batch_add_external(documents, vectors)
        else:
            self._batch_add_memory(documents, vectors)

    def search_dense(self, query_vector: List[float], top_k: int = TOP_K) -> List[Chunk]:
        if self._backend == "external":
            return self._search_dense_external(query_vector, top_k)
        return self._search_dense_memory(query_vector, top_k)

    def search_sparse(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        if self._backend == "external":
            return self._search_sparse_external(query, top_k)
        return self._search_sparse_memory(query, top_k)

    def vector_persistence(self, cache_dir: str = VECTOR_CACHE_DIR):
        if self._backend == "external":
            return  # 外部模式无需 pickle/npy 持久化，数据已在数据库中
        self._persistence_memory(cache_dir)

    @classmethod
    def vector_restore(cls, cache_dir: str = VECTOR_CACHE_DIR) -> "IndexStore | None":
        if STORAGE_BACKEND == "external":
            store = cls()  # __init__ 自动连接外部服务
            return store
        return cls._restore_memory(cache_dir)

    # ---- memory 实现 ----

    def _batch_add_memory(self, documents: List[Chunk], vectors: List[List[float]]):
        self._chunks.extend(documents)
        new_vectors = numpy.array(vectors)
        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = numpy.vstack([self._vectors, new_vectors])
        self._bm25_tokenized = [list(jieba.cut(c.content)) for c in self._chunks]
        self._bm25 = BM25Okapi(self._bm25_tokenized)

    def _search_dense_memory(self, query_vector: List[float], top_k: int = TOP_K) -> List[Chunk]:
        if self._vectors is None:
            return []
        query = numpy.array(query_vector)
        scores = numpy.dot(self._vectors, query)
        top_k = min(top_k, len(scores))
        kth = len(scores) - top_k
        partitioned = numpy.argpartition(scores, kth)
        top_indices = partitioned[kth:]
        top_scores = scores[top_indices]
        sorted_order = numpy.argsort(top_scores)
        top_indices = top_indices[numpy.flip(sorted_order)]
        return [self._chunks[i] for i in top_indices]

    def _search_sparse_memory(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        if self._bm25 is None:
            return []
        tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        top_k = min(top_k, len(scores))
        top_indices = numpy.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[numpy.argsort(scores[top_indices])[::-1]]
        return [self._chunks[i] for i in top_indices]

    def _persistence_memory(self, cache_dir: str = VECTOR_CACHE_DIR):
        path = Path(cache_dir)
        path.mkdir(exist_ok=True)
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        numpy.save(path / "vectors.npy", self._vectors)

    @classmethod
    def _restore_memory(cls, cache_dir: str = VECTOR_CACHE_DIR) -> "IndexStore | None":
        path = Path(cache_dir)
        chunks_file = path / "chunks.pkl"
        vectors_file = path / "vectors.npy"
        if not chunks_file.exists() or not vectors_file.exists():
            return None
        store = cls()
        with open(chunks_file, "rb") as f:
            store._chunks = pickle.load(f)
        store._vectors = numpy.load(vectors_file)
        store._bm25_tokenized = [list(jieba.cut(c.content)) for c in store._chunks]
        store._bm25 = BM25Okapi(store._bm25_tokenized)
        return store

    # ---- external 实现 ----

    def _batch_add_external(self, documents: List[Chunk], vectors: List[List[float]]):
        ingest_chunks(documents, numpy.array(vectors), self._pgsql, self._es, self._milvus)
        for c in documents:
            self._chunks_cache[c.chunk_id] = c

    def _search_dense_external(self, query_vector: List[float], top_k: int = TOP_K) -> List[Chunk]:
        results = self._milvus.search_dense(query_vector, top_k)
        return [self._chunks_cache.get(cid) for cid, _ in results if cid in self._chunks_cache]

    def _search_sparse_external(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        results = self._es.search_bm25(query, top_k)
        return [self._chunks_cache.get(cid) for cid, _ in results if cid in self._chunks_cache]
