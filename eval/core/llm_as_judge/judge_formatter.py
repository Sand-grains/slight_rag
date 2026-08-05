"""Judge 上下文字符串构建 + Prompt 模板加载与填充。

核心特性：
    - Formatter 类加载并缓存 faithfulness.md / quality.md 两个 Judge prompt 模板
    - build_judge_context() 将 chunk 列表拼成带 [来源X] 标记的结构化上下文字符串
    - prompt_version_hash 由模板文件内容 SHA256 计算，Judge 缓存键使用此版本号
    - 模块级 get_formatter() 单例，首次调用加载模板

用法示例::

    from eval.core.llm_as_judge.judge_formatter import get_formatter, build_judge_context
    context_str = build_judge_context(chunks)
    formatter = get_formatter()
    prompt = formatter.build_faithfulness_prompt(query, context_str, answer)

公共接口：
    - Formatter: 模板加载器 + prompt 构造函数
    - build_judge_context: chunk 列表 → Judge 用上下文字符串
    - get_formatter: 模块级 Formatter 单例
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from eval.utils import fill

if TYPE_CHECKING:
    from indexing.chunk import Chunk

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str:
    """加载 .md prompt 模板文件。

    Args:
        filename: prompts 目录下的模板文件名（如 faithfulness.md）。

    Returns:
        str：模板文件内容。

    Raises:
        FileNotFoundError: 模板文件不存在。
    """
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt 模板不存在: {path}")
    return path.read_text(encoding="utf-8")


def build_judge_context(chunks: list[Chunk]) -> str:
    """构建带 chunk_id 的上下文文本，供 Judge 引用。

    格式: [来源X | chunk_id: <id> | 文档: <name>] content...

    Args:
        chunks: 检索返回的 chunk 列表。

    Returns:
        str：结构化上下文；空列表返回 "（无检索结果）"。
    """
    if not chunks:
        return "（无检索结果）"

    parts = []
    for source_index, chunk in enumerate(chunks, start=1):
        source = chunk.origin_metadata.title or chunk.origin_metadata.source or "未知来源"
        parts.append(
            f"[来源{source_index} | chunk_id: {chunk.chunk_id} | 文档: {source}]\n{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)


class Formatter:
    """Prompt 格式化器：加载模板并缓存，按需填充。"""

    def __init__(self):
        self._faithfulness_prompt_template = _load_prompt("faithfulness.md")
        self._quality_prompt_template = _load_prompt("quality.md")
        faithfulness_hash = hashlib.sha256(self._faithfulness_prompt_template.encode()).hexdigest()[:8]
        quality_hash = hashlib.sha256(self._quality_prompt_template.encode()).hexdigest()[:8]
        self.prompt_version_hash = f"{faithfulness_hash}/{quality_hash}"

    def build_faithfulness_prompt(self, query: str, context_str: str, answer: str) -> str:
        """构建 faithfulness 评估 prompt（调用 1）。

        Args:
            query: 用户问题。
            context_str: 结构化检索上下文。
            answer: 生成器回答。

        Returns:
            str：填充后的 faithfulness prompt。
        """
        return fill(
            self._faithfulness_prompt_template,
            query=query,
            context=context_str,
            answer=answer,
        )

    def build_quality_prompt(
        self, query: str, context_str: str, answer: str, reference_facts: str
    ) -> str:
        """构建质量评估 prompt（调用 2，含 reference_facts）。

        Args:
            query: 用户问题。
            context_str: 结构化检索上下文。
            answer: 生成器回答。
            reference_facts: 参考答案事实（为空则填占位提示）。

        Returns:
            str：填充后的质量评估 prompt。
        """
        return fill(
            self._quality_prompt_template,
            query=query,
            context=context_str,
            answer=answer,
            reference_facts=reference_facts if reference_facts else "（未提供参考事实）",
        )


_formatter_instance: Formatter | None = None


def get_formatter() -> Formatter:
    """模块级单例，保证 prompt_version_hash 日志只打印一次。

    Returns:
        Formatter：全局唯一实例。
    """
    global _formatter_instance
    if _formatter_instance is None:
        _formatter_instance = Formatter()
        logging.info("Judge prompt version: %s", _formatter_instance.prompt_version_hash)
    return _formatter_instance
