from typing import List
from openai import OpenAI
from indexing.chunk import Chunk
from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL

PROMPT_TEMPLATE = """
## Role
你是一个专业的知识库问答助手, 你的任务是严格根据提供的【参考文档】回答用户的问题

## Rules(关键)
1.必须**仅依赖**下方的【参考文档】进行回答, 不要使用你内部的训练知识
2.如果【参考文档】中没有包含回答问题所需的信息, 请直接回答: "知识库中未找到相关信息". **严禁编造**
3.回答需要简洁, 逻辑清晰, 准确, 有条理, 分点描述
4.引用来源时标注[来源X], 在回答的末尾注明引用的文档名称

## Context(检索到的片段)
以下是参考文档片段:
<context>
{context_str}
</context>

## User Question
用户问题是:
{query_str}

## 回答
请开始回答:"""


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

    def generate(self, query: str, context_chunks: List[Chunk]) -> str:
        """构造 prompt 并调用 DeepSeek API 生成回答"""
        context_str = _build_context(context_chunks)             # chunk 列表 → 带来源标记的上下文字符串
        prompt = PROMPT_TEMPLATE.format(
            context_str=context_str, query_str=query             # 将上下文和用户问题填入模板
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],      # 单轮对话，System Prompt已写在模板里的Role段
        )
        return response.choices[0].message.content               # 提取 LLM 返回的文本
