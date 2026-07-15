from typing import List
from openai import OpenAI
from indexing.loader import Document
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


def _build_context(chunks: List[Document]) -> str:
    """将检索到的 chunk 列表拼成带来源标记的上下文字符串"""
    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "未知来源")     # 取 chunk 的来源文件名，取不到就用默认值
        parts.append(f"[来源{i + 1}] 文档: {source}\n{chunk.content}")  # 每个 chunk 标注序号和来源
    return "\n\n---\n\n".join(parts)                         # 用分隔线拼接多个 chunk


def generate(query: str, context_chunks: List[Document]) -> str:
    """构造 prompt 并调用 DeepSeek API 生成回答"""
    context_str = _build_context(context_chunks)             # chunk 列表 → 带来源标记的上下文字符串
    prompt = PROMPT_TEMPLATE.format(
        context_str=context_str, query_str=query             # 将上下文和用户问题填入模板
    )

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)  # 用兼容接口连 DeepSeek
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],      # 单轮对话，System Prompt已写在模板里的Role段
    )
    return response.choices[0].message.content               # 提取 LLM 返回的文本
