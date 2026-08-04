"""LLM 生成: 将问题 + 检索上下文组装为 Prompt，调用 OpenAI 兼容 API 产出自然语言回答。

核心特性：
    - _build_context 把检索到的 chunk 拼成带 [来源i] 标记的上下文字符串，供 LLM 引用溯源
    - generate() 用 GENERATOR_PROMPT_TEMPLATE.format() 填入上下文与问题，单轮 user 消息调用
    - temperature 默认取 GENERATOR_TEMPERATURE（eval 场景为 0，保证输出确定性，是内容寻址缓存生效的前提）
    - 被上层消费：agent_pipeline（Agent 问答）与 eval Layer 2（构造参考上下文）均调用 Generator

用法示例::

    from retrieval.generator import Generator
    generator = Generator()
    answer = generator.generate("什么是 RAG", context_chunks)

公共接口：
    - _build_context: chunk 列表 → 带来源标记的上下文字符串
    - Generator.generate: 问题 + 上下文 → LLM 自然语言回答
"""
from openai import OpenAI

from config import GENERATOR_PROMPT_TEMPLATE, GENERATOR_TEMPERATURE, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_ID
from indexing.chunk import Chunk


def _build_context(chunks: list[Chunk]) -> str:
    """将检索到的最终 chunk 列表拼接(带来源标记), 作为生成前的上下文

    Args:
        chunks: 检索层返回的 chunk 列表（父块）。

    Returns:
        str：按 [来源i] 序号逐段展开、用分隔线拼接的上下文字符串，供 LLM 引用溯源。
    """
    parts = []
    for index, chunk in enumerate(chunks):
        source = chunk.origin_metadata.title or chunk.origin_metadata.source or "[未知来源]"
        parts.append(f"[来源{index + 1}] 文档: {source}\n{chunk.content}")  # 每个 chunk 标注序号和来源
    return "\n\n---\n\n".join(parts)  # 用分隔线拼接多个 chunk


class Generator:
    """生成层：封装 Prompt 构造 + LLM 调用，输入问题与检索上下文，输出自然语言回答。"""

    def __init__(self, model: str = LLM_MODEL_ID):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)  # 复用同一个 client 实例
        self.model = model  # 模型 ID

    def generate(self, query: str, context_chunks: list[Chunk], temperature: float = GENERATOR_TEMPERATURE) -> str:
        """构造 prompt 并调用 API 生成回答。

        Args:
            query: 用户问题。
            context_chunks: 检索到的上下文 chunk 列表。
            temperature: 采样温度，默认取 GENERATOR_TEMPERATURE
            （eval 下为 0，保证输出确定性，是内容寻址缓存生效的前提）

        Returns:
            str：LLM 返回的自然语言回答文本。
        """
        context = _build_context(context_chunks)  # chunk 列表 → 带来源标记的上下文字符串
        prompt = GENERATOR_PROMPT_TEMPLATE.format(
            context_str=context,
            query_str=query,  # 将上下文和用户问题填入模板
        )
        # LLM调用
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],  # 单轮对话，System Prompt 已写在模板里的 Role 段
            temperature=temperature,
        )
        return response.choices[0].message.content  # 返回 LLM 生成的文本
