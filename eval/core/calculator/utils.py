"""Eval 工具函数：分数钳制 + LLM JSON 输出的三层兜底解析。

核心特性：
    - _clamp_score: 将 LLM 输出分数钳制到 [0, 1]
    - _extract_json: 三层兜底解析（直接 json.loads → markdown fence → 正则 brace 计数），应对 LLM 输出格式的不确定性

用法示例::

    from eval.core.calculator.utils import _extract_json
    result = _extract_json('{"faithfulness": 0.85}')

公共接口：
    - _clamp_score: 分数钳制到 [lo, hi]
    - _extract_json: 从任意文本中提取 JSON dict
"""

import json
import re


def _clamp_score(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """将数值钳制到特定的 [lo, hi] 区间内。

    钳制是幂等操作：区间内的值原样返回，越界值被拉回最近的边界
    （小于 lo 抬升到 lo，大于 hi 压回 hi），不涉及缩放。

    Args:
        value: 待钳制的数值（可能越界，如 LLM 输出的原始分数）。
        lo: 区间下界。
        hi: 区间上界。

    Returns:
        float：钳制后的数值，恒位于 [lo, hi] 区间内。
    """
    return max(lo, min(hi, value))


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON，三层兜底策略。

    Tier 1: 直接 json.loads。
    Tier 2: 从 ```json ... ``` markdown fence 中提取。
    Tier 3: 正则匹配首对完整花括号（计 brace 深度）。

    Args:
        text: LLM 原始输出文本（可能带说明文字 / markdown 代码块）。

    Returns:
        dict：解析出的 JSON 对象。

    Raises:
        ValueError: 文本中不存在可解析的 JSON 对象或花括号不闭合。
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    matches = re.findall(fence_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError(f"Cannot extract JSON from text: {text[:100]}...")

    depth = 0
    for index in range(brace_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[brace_start : index + 1]
                return json.loads(candidate)

    raise ValueError(f"Unmatched braces in text: {text[:100]}...")
