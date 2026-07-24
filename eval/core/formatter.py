"""Context 格式化 + Prompt 模板加载与填充。"""

import os
from pathlib import Path
from typing import List
from indexing.chunk import Chunk

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str:
    """加载 .md prompt 模板文件"""
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt 模板不存在: {path}")
    return path.read_text(encoding="utf-8")


def _fill(template: str, **kwargs) -> str:
    """用简单字符串替换填充模板（避免 .format() 的 brace 转义问题）"""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def build_judge_context(chunks: List[Chunk]) -> str:
    """构建带 chunk_id 的上下文文本，供 Judge 引用

    格式: [来源X | chunk_id: <id> | 文档: <name>] content...
    """
    if not chunks:
        return "（无检索结果）"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.doc_meta.title or chunk.doc_meta.source or "未知来源"
        parts.append(
            f"[来源{i} | chunk_id: {chunk.chunk_id} | 文档: {source}]\n{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)


class Formatter:
    """Prompt 格式化器: 加载模板并缓存，按需填充。"""

    def __init__(self):
        self._faithfulness_tpl = _load_prompt("faithfulness.md")
        self._quality_tpl = _load_prompt("quality.md")

    def build_faithfulness_prompt(self, query: str, context_str: str, answer: str) -> str:
        """构建 faithfulness 评估 prompt（调用 1）。"""
        return _fill(
            self._faithfulness_tpl,
            query=query,
            context=context_str,
            answer=answer,
        )

    def build_quality_prompt(
        self, query: str, context_str: str, answer: str, reference_facts: str
    ) -> str:
        """构建质量评估 prompt（调用 2, 含 reference_facts）"""
        return _fill(
            self._quality_tpl,
            query=query,
            context=context_str,
            answer=answer,
            reference_facts=reference_facts if reference_facts else "（未提供参考事实）",
        )
