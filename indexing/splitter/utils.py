"""Splitter 共用工具：token 估算、fenced code block 范围检测、标题正则。"""
import re

_HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_REGEX = re.compile(r"^```", re.MULTILINE)


def find_fenced_block_ranges(text: str) -> list[tuple[int, int]]:
    """返回 fenced code block 的 [开始索引start, 结束索引end): 未闭合则延伸到文末"""

    ranges = [] # 返回列表, 每个元素表示文档中一个代码块的字符区间
    in_code = False # 表示"当前扫描位置是否在某个代码块内部"
    start = 0 # 暂存"当前正在扫描的代码块的起始索引"
    for match in _FENCE_REGEX.finditer(text): # m是finditer返回的迭代器，每次迭代返回一个match对象
        if in_code:
            ranges.append((start, match.end()))
            in_code = False
        else:
            start = match.start()
            in_code = True
    if in_code:
        ranges.append((start, len(text)))
    return ranges


def inside_code(pos: int, ranges: list[tuple[int, int]]) -> bool:
    # 判断单个位置 pos 是否落在任一代码块区间内
    return any(s <= pos < e for s, e in ranges)


def overlaps_code(a: int, b: int, ranges: list[tuple[int, int]]) -> bool:
    # 判断区间 [a, b) 是否与任一代码块区间有交集
    return any(a < e and b > s for s, e in ranges)


def token_estimate(text: str) -> int:
    """token 估算(伪精确, 仅用于诊断而非监控数据)"""
    return len(text) // 2