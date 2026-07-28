"""Judge 上下文字符串构建 + Prompt 模板加载与填充。

核心特性：
    - Formatter 类加载并缓存 faithfulness.md / quality.md 两个 Judge prompt 模板
    - build_judge_context() 将 chunk 列表拼成带 [来源X] 标记的结构化上下文字符串
    - prompt_version 由模板文件内容 SHA256 计算，Judge 缓存键使用此版本号
    - 模块级 get_formatter() 单例，首次调用加载模板

用法示例::

    from eval.core.formatter import get_formatter, build_judge_context
    context_str = build_judge_context(chunks)
    formatter = get_formatter()
    prompt = formatter.build_faithfulness_prompt(query, context_str, answer)

公共接口：
    - Formatter: 模板加载器 + prompt 构造函数
    - build_judge_context: chunk 列表 → Judge 用上下文字符串
    - get_formatter: 模块级 Formatter 单例
"""

import hashlib
import logging
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
    """Prompt 格式化器：加载模板并缓存，按需填充。"""

    def __init__(self):
        self._faithfulness_tpl = _load_prompt("faithfulness.md")
        self._quality_tpl = _load_prompt("quality.md")
        f_prompt_hash = hashlib.sha256(self._faithfulness_tpl.encode()).hexdigest()[:8]
        q_prompt_hash = hashlib.sha256(self._quality_tpl.encode()).hexdigest()[:8]
        self.prompt_version = f"{f_prompt_hash}/{q_prompt_hash}"

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


_formatter_instance: Formatter | None = None


def get_formatter() -> Formatter:
    """模块级单例, 保证 prompt_version 日志只打印一次。"""
    global _formatter_instance
    if _formatter_instance is None:
        _formatter_instance = Formatter()
        logging.info("Judge prompt version: %s", _formatter_instance.prompt_version)
    return _formatter_instance
