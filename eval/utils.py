"""eval 模块工具函数。纯函数，零外部依赖。"""

import json
import re
import os


def _clamp_score(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """将数值钳制到 [lo, hi] 区间内。"""
    return max(lo, min(hi, value))


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON，三层兜底策略。

    Tier 1: 直接 json.loads。
    Tier 2: 从 ```json ... ``` markdown fence 中提取。
    Tier 3: 正则匹配首对完整花括号（计 brace 深度）。
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
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[brace_start : i + 1]
                return json.loads(candidate)

    raise ValueError(f"Unmatched braces in text: {text[:100]}...")


def _ensure_list(value) -> list:
    """将单值包装为列表，已是列表则原样返回，None 返回空列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _filename_stem_match(path_a: str, path_b: str) -> bool:
    """比较两个路径的文件名主干（忽略扩展名和目录），不区分大小写。"""
    stem_a = os.path.splitext(os.path.basename(path_a))[0].lower()
    stem_b = os.path.splitext(os.path.basename(path_b))[0].lower()
    return stem_a == stem_b
