"""Eval 通用工具函数：文件 I/O、统计、字符串/路径、LLM 输出解析、运维。

核心特性：
    - 全部为纯函数（无模块级可变状态），零业务耦合，被 eval 下多包共享
    - 引用方一览（集中维护，避免工具函数散落各业务模块造成混淆）：
        - clamp_score / extract_json: eval/core/llm_as_judge/judge.py
        - read_json: eval/core/benchmark.py、eval/runner.py
        - write_json / append_jsonl / get_git_commit: eval/reporter.py
        - stem: eval/core/retrieval/retrieval_layer.py
        - mean_of / p95: eval/monitor/monitor_metrics.py
        - format_time: eval/monitor/monitor_panel.py
        - fill: eval/core/llm_as_judge/judge_formatter.py

用法示例::

    from eval.utils import extract_json, write_json
    data = extract_json('{"faithfulness": 0.85}')
    write_json("out.json", data)

公共接口：
    - 文件 I/O: read_json / write_json / append_jsonl
    - 统计: mean_of / p95
    - 字符串与路径: stem / fill
    - 时间: format_time
    - 数值与 LLM: clamp_score / extract_json
    - 运维: get_git_commit
"""

import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path


# ---- 文件 I/O ----

def read_json(path: str | Path) -> dict | list | None:
    """读取 JSON 文件；文件不存在或内容损坏时返回 None。

    引用方: eval/core/benchmark.py、eval/runner.py。

    Args:
        path: JSON 文件路径（str 或 Path）。

    Returns:
        dict | list | None：解析结果；文件不存在/解析失败时返回 None。
    """
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    try:
        with open(path_obj, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except (json.JSONDecodeError, OSError):
        return None


def write_json(path: str | Path, data: dict, default: Callable | None = None) -> None:
    """将 dict 写入 JSON 文件（ensure_ascii=False + 2 空格缩进）。

    引用方: eval/reporter.py、eval/core/llm_as_judge/judge_calibrate.py。

    Args:
        path: 输出文件路径（str 或 Path）。
        data: 待写入的 dict。
        default: 非 JSON 可序列化值的转换函数（同 json.dump 的 default 参数）。
    """
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, ensure_ascii=False, indent=2, default=default)


def append_jsonl(path: str, line: dict) -> None:
    """向 JSONL 文件追加一行记录（每个 dict 独立成行）。

    引用方: eval/reporter.py（history.jsonl 追加）。

    Args:
        path: JSONL 文件路径。
        line: 待追加的一行记录。
    """
    with open(path, "a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---- 统计 ----

def mean_of(values: list[float]) -> float | None:
    """非 None 值均值；全为 None 时返回 None。

    引用方: eval/monitor/monitor_metrics.py。

    Args:
        values: 数值列表（可含 None）。

    Returns:
        float | None：非 None 值的均值；全 None 时返回 None。
    """
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def percentile(values: list[float], percentile: float) -> float:
    """计算数值列表的第 percentile 百分位；空列表返回 0.0。

    引用方: eval/monitor/monitor_metrics.py、eval/utils.p95。

    Args:
        values: 数值列表。
        percentile: 百分位（0-100，如 50/75/95）。

    Returns:
        float：对应百分位值；空列表时返回 0.0。
    """
    if not values:
        return 0.0
    import numpy
    return float(numpy.percentile(values, percentile))


def p95(values: list[float]) -> float:
    """第 95 百分位；空列表返回 0.0。

    引用方: eval/monitor/monitor_metrics.py、eval/monitor/monitor_panel.py。

    Args:
        values: 数值列表。

    Returns:
        float：P95 值；空列表时返回 0.0。
    """
    return percentile(values, 95)


# ---- 字符串与路径 ----

def stem(filepath: str) -> str:
    """提取文件名主干（小写、不含扩展名），用于文件级匹配。

    引用方: eval/core/retrieval/retrieval_layer.py。

    Args:
        filepath: 完整文件路径。

    Returns:
        str：小写文件名主干。
    """
    return os.path.splitext(os.path.basename(filepath))[0].lower()


def fill(template: str, **kwargs) -> str:
    """用简单字符串替换填充模板（避免 .format() 的 brace 转义问题）。

    引用方: eval/core/llm_as_judge/judge_formatter.py。

    Args:
        template: 含 {key} 占位符的模板文本。
        **kwargs: 占位符名 → 替换值。

    Returns:
        str：占位符全部替换后的模板。
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


# ---- 时间 ----

def format_time(seconds: float) -> str:
    """秒数格式化为 HH:MM:SS。

    引用方: eval/monitor/monitor_panel.py。

    Args:
        seconds: 经过的秒数。

    Returns:
        str：HH:MM:SS 格式。
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


# ---- 数值与 LLM ----

def clamp_score(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """将数值钳制到 [lo, hi] 区间内。

    钳制是幂等操作：区间内的值原样返回，越界值被拉回最近的边界
    （小于 lo 抬升到 lo，大于 hi 压回 hi），不涉及缩放。

    引用方: eval/core/llm_as_judge/judge.py。

    Args:
        value: 待钳制的数值（可能越界，如 LLM 输出的原始分数）。
        lo: 区间下界。
        hi: 区间上界。

    Returns:
        float：钳制后的数值，恒位于 [lo, hi] 区间内。
    """
    return max(lo, min(hi, value))


def extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON，三层兜底策略。

    Tier 1: 直接 json.loads。
    Tier 2: 从 ```json ... ``` markdown fence 中提取。
    Tier 3: 正则匹配首对完整花括号（计 brace 深度）。

    引用方: eval/core/llm_as_judge/judge.py。

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


# ---- 运维 ----

def get_git_commit() -> str:
    """获取当前 git commit 短哈希。

    引用方: eval/reporter.py（run_info 快照）。

    Returns:
        str：7 位短哈希；git 不可用时返回 "unknown"。
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"
