"""Markdown 结构静态分析：标题解析、层级连续性、标题密度、section token 统计、文本占比、编码异味。

纯诊断用（Router 前置），阈值常量见 config.py。分析结果由 md_diagnosis.diagnose 聚合为 DocQualityReport。
"""
import re
from dataclasses import dataclass
from statistics import median

from config import HEADING_DENSITY_MIN_OK, HEADING_DENSITY_THRESHOLD
from indexing.splitter.utils import _HEADING_REGEX, token_estimate, find_fenced_block_ranges, inside_code

_MOJIBAKE_RE = re.compile(r"[�]|Ã[\x80-\xbf]|â€|ï¼")


@dataclass
class Heading:
    level: int # 标题级别
    text: str # 标题字面文本
    line_number: int # 标题所在原文行号
    char_start: int # 标题所在行起始位置
    char_end: int # 标题所在行结束位置


def parse_headings(text: str) -> list[Heading]:
    """解析 Markdown 标题行 (跳过 fenced code block 内的 # 行)
    返回整篇文档中所有Headings的信息
    """
    headings = []
    code_ranges = find_fenced_block_ranges(text)
    position = 0
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), 1):
        # text.splitlines 把整篇文本按行切开
        # keepends=True使 raw_line 是一行含换行符的原始文本
        line = raw_line.rstrip("\r\n") # line是去掉行尾换行符的纯行内容
        if not inside_code(position, code_ranges): # 表示不匹配code block内的标题
            match = _HEADING_REGEX.match(line)
            if match:
                headings.append(
                    Heading(
                        level=len(match.group(1)),
                        text=match.group(2).strip(),
                        line_number=line_number,
                        char_start=position,
                        char_end=position + len(line),))
        position += len(raw_line)
    return headings


def check_heading_continuity(headings: list[Heading]) -> bool:
    """标题层级连续性诊断: 拦截向下钻时一次跨两级以上 (层级回退, 如4->2 不会拦截)"""
    if not headings:
        return False
    current_level = headings[0].level
    for next_heading in headings[1:]:
        if next_heading.level > current_level + 1:
            return False
        current_level = next_heading.level
    return True


def heading_density_ok(headings: list[Heading], text: str) -> bool:
    """文章标题密度指标: 平均每 HEADING_DENSITY_THRESHOLD 字符至少一个标题
    另外有 h1 且标题总数达标可豁免"""
    if not headings:
        return False
    if any(heading.level == 1 for heading in headings) and len(headings) >= HEADING_DENSITY_MIN_OK:
        return True
    return  len(text) / len(headings) <= HEADING_DENSITY_THRESHOLD


def section_token_statistics(text: str, headings: list[Heading]) -> tuple[int, int]:
    """按标题切分 section，统计各 section token 的 max 与 median   无标题时整篇为一节"""
    if not headings:
        tokens = token_estimate(text)
        return tokens, tokens
    bounds = [heading.char_start for heading in headings] + [len(text)]
    tokens = [token_estimate(text[start:bounds[i + 1]]) for i, start in enumerate(bounds[:-1])]
    return max(tokens), median(tokens)


def text_ratio(text: str) -> float:
    """剔除代码块后的纯文本占比   非空白字符 / 总字符"""
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    code_ranges = find_fenced_block_ranges(text)
    non_whitespace = 0 # 非空白计数
    cursor = 0
    for start, end in code_ranges:
        # 每个符合条件的字符，产出 1, 然后再sum求和
        non_whitespace += sum(1 for ch in text[cursor:start] if not ch.isspace())
        cursor = end
    non_whitespace += sum(1 for ch in text[cursor:] if not ch.isspace())
    return non_whitespace / total_chars


def detect_encoding_issues(text: str) -> bool:
    """编码异味: UTF-8 replacement char 或常见 mojibake 序列。"""
    return bool(_MOJIBAKE_RE.search(text))
