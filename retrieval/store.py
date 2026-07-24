import numpy
import pickle
import jieba
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi
from indexing.chunk import Chunk
from config import TOP_K


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR_NAME = str(_PROJECT_ROOT / ".vector_cache")


class VectorStore:
    """内存向量存储：numpy 稠密向量 + BM25 稀疏索引，双路检索"""

    def __init__(self):
        self._chunks: List[Chunk] = []                # 存储所有 chunk 的 Chunk 对象
        self._vectors: numpy.ndarray | None = None       # 二维数组，形状 (N, dim)，行对应 chunk，列对应向量维度
        self._bm25: BM25Okapi | None = None               # BM25 稀疏索引
        self._bm25_tokenized: List[List[str]] = []        # 已分词的 chunk 文本，与 _chunks 下标对齐

    @property
    def chunk_ids(self) -> set[str]:
        """返回库中所有 chunk_id 的集合。"""
        return {c.chunk_id for c in self._chunks}

    @property
    def chunks(self) -> List[Chunk]:
        """返回库中全部 chunk 的只读视图。"""
        return list(self._chunks)

    def add(self, documents: List[Chunk], vectors: List[List[float]]):
        """将 chunk 及其向量追加入库，并重建 BM25 索引"""
        self._chunks.extend(documents)                   # 追加 chunk 到列表末尾
        new_vectors = numpy.array(vectors)               # List[List[float]] → 高效的二维 numpy 数组, 方便后续计算
        if self._vectors is None:
            self._vectors = new_vectors                  # 首次入库，直接赋值
        else:
            self._vectors = numpy.vstack([self._vectors, new_vectors])  # 非首次，沿行方向拼接
        # 重建 BM25 索引（每次 add 全量重建，542 chunk 级别开销可忽略）
        self._bm25_tokenized = [list(jieba.cut(c.retrieval_text or c.content)) for c in self._chunks]
        self._bm25 = BM25Okapi(self._bm25_tokenized)

    def search_dense(self, query_vector: List[float], top_k: int = TOP_K) -> List[Chunk]:
        """余弦相似度检索：query_vector 与库中所有向量做点积，返回 top_k 个最相关的 chunk"""
        if self._vectors is None:
            return []                                    # 库为空，直接返回空列表
        query = numpy.array(query_vector)                # 查询向量转 numpy 数组
        scores = numpy.dot(self._vectors, query)         # 查询向量与"内存向量数据库"作点积, 得到余弦相似度([0, 1])的分数数组
        top_k = min(top_k, len(scores))                  # 防止库中数量不足 top_k

        # 第一步：粗筛 —— 从 N 个分数中捞出最大的 top_k 个（不会全排序）
        kth = len(scores) - top_k                        # 第 top_k 大的元素在升序数组中的位置
        partitioned = numpy.argpartition(scores, kth)    # 以第 kth 个为界划分，右边全是 ≥ 它的, 左边全是 <= 它的
        top_indices = partitioned[kth:]                  # 从第top_k大的元素开始, 截取右侧所有元素得到一个索引数组(右侧全是>=它的)

        # 第二步：精排 —— 对上一步的 top_k 个按分数降序排列
        top_scores = scores[top_indices]                 # 用将一个索引数组传入[], 取出这 top_k 个对应的分数
        sorted_order = numpy.argsort(top_scores)         # 升序排列，返回排序后的位置映射
        top_indices = top_indices[numpy.flip(sorted_order)]  # flip 反转得到降序
        return [self._chunks[i] for i in top_indices]    # 按得分从高到低返回对应 chunk

    def search_sparse(self, query: str, top_k: int = TOP_K) -> List[Chunk]:
        """BM25 稀疏检索：query 分词 → BM25 打分 → top_k chunks"""
        if self._bm25 is None:
            return []
        tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)               # (N,) numpy array
        top_k = min(top_k, len(scores))
        top_indices = numpy.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[numpy.argsort(scores[top_indices])[::-1]]
        return [self._chunks[i] for i in top_indices]

    def vector_persistence(self, cache_dir: str = CACHE_DIR_NAME):
        """将当前 chunks 和 vectors 持久化到磁盘"""
        path = Path(cache_dir)
        path.mkdir(exist_ok=True)
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        numpy.save(path / "vectors.npy", self._vectors)

    @classmethod
    def vector_restore(cls, cache_dir: str = CACHE_DIR_NAME) -> "VectorStore | None":
        """从磁盘恢复 VectorStore，缓存不存在时返回 None"""
        path = Path(cache_dir)
        chunks_file = path / "chunks.pkl"
        vectors_file = path / "vectors.npy"
        if not chunks_file.exists() or not vectors_file.exists():
            return None
        store = cls()
        with open(chunks_file, "rb") as f:
            store._chunks = pickle.load(f)
        store._vectors = numpy.load(vectors_file)
        # 从 chunks 重建 BM25 索引
        store._bm25_tokenized = [list(jieba.cut(c.retrieval_text or c.content)) for c in store._chunks]
        store._bm25 = BM25Okapi(store._bm25_tokenized)
        return store
