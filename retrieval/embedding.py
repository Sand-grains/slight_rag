from typing import List
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None  # 模块级单例，整个进程只加载一次模型


def _get_model() -> SentenceTransformer:
    """延迟加载模型：首次调用时从本地缓存加载，后续直接复用，避免重复加载"""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)  # 首次加载模型到内存
    return _model


def embed(texts: List[str]) -> List[List[float]]:  # 返回向量化后的高维数组
    """将文本列表转为向量列表，每条向量已 L2 归一化（模长=1），点积即余弦相似度"""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)  # model.encode() 直接从本地缓存加载，normalize 确保向量模长为 1
    return embeddings.tolist()  # numpy 数组 → Python 原生 list，方便后续 JSON 序列化等操作
