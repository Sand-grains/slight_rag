import numpy
from typing import List
from indexing.loader import Document
from config import TOP_K


class VectorStore:
    """内存向量存储：用列表存 chunk，用二维 numpy 数组存向量，纯内存运算检索"""

    def __init__(self):
        self._chunks: List[Document] = []                # 存储所有 chunk 的 Document 对象
        self._vectors: numpy.ndarray | None = None       # 二维数组，形状 (N, dim)，行对应 chunk，列对应向量维度

    def add(self, documents: List[Document], vectors: List[List[float]]):
        """将 chunk 及其向量追加入库，支持多次调用（增量索引）"""
        self._chunks.extend(documents)                   # 追加 chunk 到列表末尾
        new_vectors = numpy.array(vectors)               # List[List[float]] → 高效的二维 numpy 数组, 方便后续计算
        if self._vectors is None:
            self._vectors = new_vectors                  # 首次入库，直接赋值
        else:
            self._vectors = numpy.vstack([self._vectors, new_vectors])  # 非首次，沿行方向拼接

    def search(self, query_vector: List[float], top_k: int = TOP_K) -> List[Document]:
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
