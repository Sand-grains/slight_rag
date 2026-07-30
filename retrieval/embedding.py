"""文本向量化：Sentence-Transformer 模型加载 + 批量编码。

核心特性：
    - 模块级单例延迟加载 BGE-M3 模型（1024 维），整个进程只加载一次
    - 输出向量已 L2 归一化（模长=1），点积即余弦相似度
    - 模型从本地路径加载（D:/Model），离线可用

用法示例::

    from retrieval.embedding import embed
    vectors = embed(["查询文本", "另一段文本"])  # → [[0.1, -0.3, ...], [...]]

公共接口：
    - embed: 文本列表 → 归一化向量列表
"""

from typing import List
from config import EMBEDDING_MODEL_PATH          # 先加载配置（触发 load_dotenv()，设置 HF_ENDPOINT 等环境变量）
from sentence_transformers import SentenceTransformer  # 后导入模型库（此时环境变量已就绪）

_model: SentenceTransformer | None = None  # 模块级单例，整个进程只加载一次模型


def _get_model() -> SentenceTransformer:
    """延迟加载模型：首次调用时从本地缓存加载，后续直接复用，避免重复加载"""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_PATH)  # 首次加载模型到内存
    return _model


def embed(texts: List[str]) -> List[List[float]]:  # 返回向量化后的高维数组
    """将文本列表转为向量列表，每条向量已 L2 归一化（模长=1），点积即余弦相似度"""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)  # model.encode() 直接从本地缓存加载，normalize 确保向量模长为 1
    return embeddings.tolist()  # numpy 数组 → Python 原生 list，方便后续 JSON 序列化等操作
