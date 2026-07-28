"""检索管线：稠密+稀疏双路检索 → RRF 融合 → LLM 生成回答。

公共接口：
    - Retriever: 双路检索器（dense + sparse → RRF → top_k Chunks）
    - Generator: LLM 生成器（Prompt 构造 + OpenAI API 调用 → 自然语言回答）
"""

from .retriever import Retriever
from .generator import Generator
