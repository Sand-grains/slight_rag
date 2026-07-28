from typing import List
from openai import OpenAI
from indexing.chunk import Chunk
from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL, PROMPT_TEMPLATE, GENERATOR_TEMPERATURE


def _build_context(chunks: List[Chunk]) -> str:
    """将检索到的 chunk 列表拼成带来源标记的上下文字符串"""
    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.doc_meta.title or chunk.doc_meta.source or "未知来源"
        parts.append(f"[来源{i + 1}] 文档: {source}\n{chunk.content}")  # 每个 chunk 标注序号和来源
    return "\n\n---\n\n".join(parts)                         # 用分隔线拼接多个 chunk


class Generator:
    """生成层：封装 Prompt 构造 + LLM 调用，输入问题与检索到的上下文，输出自然语言回答"""

    def __init__(self, model: str = LLM_MODEL_ID):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)  # 复用同一个 client 实例
        self.model = model                                                  # 模型 ID，评估时可传入低成本模型

    def generate(self, query: str, context_chunks: List[Chunk], temperature: float = GENERATOR_TEMPERATURE) -> str:
        """构造 prompt 并调用 API 生成回答"""
        context_str = _build_context(context_chunks)             # chunk 列表 → 带来源标记的上下文字符串
        prompt = PROMPT_TEMPLATE.format(
            context_str=context_str, query_str=query             # 将上下文和用户问题填入模板
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],      # 单轮对话，System Prompt已写在模板里的Role段
            temperature=temperature,
        )
        return response.choices[0].message.content               # 提取 LLM 返回的文本
