"""定向验证：guide/fix_code_block.md 的 6 个修复变体 × 6 个测试输入（非侵入，不改源码）。

判定标准（对应文档"纯文本逐字节不变 / 正文干净 / 无垃圾尾巴"三条声明）：
  1. 垃圾尾巴消失（无 stray-fence：chunk 中部出现 ```；无 micro-chunk：≤3 字符）
  2. 代码块后段落边界保留（不把紧跟代码块的短段落硬切碎片化）
  3. 纯文本（含病态输入）逐字节不变（对照 OLD 的 chunk 内容序列）

6 个变体：
  A swallow+merge   = v2 原案（变更1 全局吞尾随换行 + 变更2 + 变更3 全局 merge）
  B only-overlap    = 仅变更2（不跨代码块回填，窗口 clamp）
  C hard-split-local = 变更2 + 收窄变更1（只在 _hard_split 拉边界时吞尾随换行，保 \n\n 段落边界）
  D swallow-only    = 变更2 + 变更1 全局吞（隔离变更3 的独立贡献）
  E C+全局空白合并    = C + 把任意纯空白微块并入前一片
  F C+仅代码邻接空白  = C + 仅把紧跟代码块（start-code_range_end≤2）的纯空白微块并入前一片 ← 定稿推荐

用法：uv run python scripts/verify_splitter_regression.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CHILD_CHUNK_SIZE, CHILD_OVERLAP
from indexing.splitter.recursive_splitter import RecursiveCharacterTextSplitter
from indexing.splitter.utils import find_fenced_block_ranges, overlaps_code

_FENCE_REGEX = re.compile(r"^```", re.MULTILINE)


def find_fenced_block_ranges_swallow(text: str) -> list[tuple[int, int]]:
    """变更1：代码块区间 end 从闭合 ``` 之后延伸到紧随的连续 \\n 之后。"""
    ranges = []
    in_code = False
    start = 0
    for match in _FENCE_REGEX.finditer(text):
        if in_code:
            end = match.end()
            while end < len(text) and text[end] == "\n":
                end += 1
            ranges.append((start, end))
            in_code = False
        else:
            start = match.start()
            in_code = True
    if in_code:
        ranges.append((start, len(text)))
    return ranges


class VariantBase(RecursiveCharacterTextSplitter):
    """模拟各变体的基类：按类属性开关组合变更，不改任何源文件。"""

    SWALLOW_FENCES = False      # 变更1：find_fenced_block_ranges 吞尾随换行
    GLOBAL_MERGE = False        # 变更3：split() 内全局 _merge_good
    HARD_SPLIT_SWALLOW = False  # 收窄变更1：仅 _hard_split 拉边界时吞尾随换行
    MERGE_WS_MICRO = False      # 变更3'：仅把纯空白微块并入前一片（替代全局 merge，避免粘段落）
    MERGE_WS_MICRO_ADJACENT_CODE = False  # 变更3''：仅把"紧跟代码块"的纯空白微块并入前一片

    def _fence_ranges(self, text: str) -> list[tuple[int, int]]:
        if self.SWALLOW_FENCES:
            return find_fenced_block_ranges_swallow(text)
        return find_fenced_block_ranges(text)

    def split(self, text: str, metadata: dict):
        code_ranges = self._fence_ranges(text)
        pieces = self._split_text(text, code_ranges)
        pieces = self._merge_isolated_headings(pieces)
        if self.GLOBAL_MERGE:
            pieces = self._merge_good(pieces)
        if self.MERGE_WS_MICRO:
            pieces = _merge_ws_micro(pieces)
        if self.MERGE_WS_MICRO_ADJACENT_CODE:
            pieces = _merge_ws_micro_adjacent_code(pieces, code_ranges)
        pieces = self._apply_overlap(pieces, code_ranges)
        chunks = []
        for index, (content, start) in enumerate(pieces):
            chunk = self._package_to_chunk(content, metadata, start_char_index=start)
            chunk.chunk_id = f"{metadata['doc_id']}:{index}"
            chunk.origin_metadata.chunk_level = self._chunk_level
            chunks.append(chunk)
        return chunks

    def _apply_overlap(self, pieces: list[tuple[str, int]],
                       code_ranges: list[tuple[int, int]]) -> list[tuple[str, int]]:
        """变更2：窗口 [start-min(overlap,len(prev)), start)，与代码区间重叠则跳过；不改 start。"""
        if self._chunk_overlap <= 0:
            return pieces
        result = []
        for content, start in pieces:
            if result:
                prev_content, _ = result[-1]
                window_start = start - min(self._chunk_overlap, len(prev_content))
                if not overlaps_code(window_start, start, code_ranges):
                    content = prev_content[-self._chunk_overlap:] + content
            result.append((content, start))
        return result

    def _hard_split(self, segment: str, segment_start: int,
                    code_ranges: list[tuple[int, int]]) -> list[tuple[str, int]]:
        """收窄变更1：拉边界到 code_range_end 时，把其后连续 \\n 一并吞入该片（保 \\n\\n 段落边界）。"""
        if not self.HARD_SPLIT_SWALLOW:
            return super()._hard_split(segment, segment_start, code_ranges)
        pieces = []
        length = len(segment)
        cursor_index = 0
        while cursor_index < length:
            end_index = min(cursor_index + self._chunk_size, length)
            for code_range_start, code_range_end in code_ranges:
                if code_range_start <= segment_start + end_index < code_range_end:
                    end_index = min(code_range_end, segment_start + length) - segment_start
                    while end_index < length and segment[end_index] == "\n":
                        end_index += 1
                    break
            if end_index <= cursor_index:
                end_index = min(cursor_index + self._chunk_size, length)
            pieces.append((segment[cursor_index:end_index], segment_start + cursor_index))
            cursor_index = end_index
        return pieces


class VariantA(VariantBase):
    SWALLOW_FENCES = True
    GLOBAL_MERGE = True


class VariantB(VariantBase):
    pass


class VariantC(VariantBase):
    HARD_SPLIT_SWALLOW = True


class VariantD(VariantBase):
    SWALLOW_FENCES = True


class VariantE(VariantBase):
    HARD_SPLIT_SWALLOW = True
    MERGE_WS_MICRO = True


class VariantF(VariantBase):
    HARD_SPLIT_SWALLOW = True
    MERGE_WS_MICRO_ADJACENT_CODE = True


# ---- 测试输入 ----

def _long_code_block() -> str:
    code = "\n".join(f"print({index})  # line {index:03d}" for index in range(20))
    return "```\n" + code + "\n```"


def _short_code_block() -> str:
    return "```\ncode()\n```"


def _merge_ws_micro(pieces: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """变更3'：把纯空白（len≤2）微块并入前一片；首片才并入后一片。start 取被并入片的原值。"""
    result = list(pieces)
    index = 0
    while index < len(result):
        content, start = result[index]
        if content.strip() == "" and len(content) <= 2:
            if index > 0:
                prev_content, prev_start = result[index - 1]
                result[index - 1] = (prev_content + content, prev_start)
            elif index + 1 < len(result):
                next_content, next_start = result[index + 1]
                result[index + 1] = (content + next_content, next_start)
            result.pop(index)
        else:
            index += 1
    return result


def _merge_ws_micro_adjacent_code(pieces: list[tuple[str, int]],
                                  code_ranges: list[tuple[int, int]]) -> list[tuple[str, int]]:
    """变更3''：仅把"紧跟代码块"（start == 某 code_range_end）的纯空白微块并入前一片。

    相比 _merge_ws_micro 的"无差别合并"，用 code_range_end 判定收窄到代码块邻接场景，
    保证纯文本（无代码块 → 无命中）逐字节不变。
    """
    result = list(pieces)
    index = 0
    while index < len(result):
        content, start = result[index]
        adjacent = any(0 <= start - code_range_end <= 2 for _, code_range_end in code_ranges)
        if content.strip() == "" and len(content) <= 2 and adjacent:
            if index > 0:
                prev_content, prev_start = result[index - 1]
                result[index - 1] = (prev_content + content, prev_start)
            elif index + 1 < len(result):
                next_content, next_start = result[index + 1]
                result[index + 1] = (content + next_content, next_start)
            result.pop(index)
        else:
            index += 1
    return result


def build_cases() -> list[tuple[str, str]]:
    long_block = _long_code_block()
    short_block = _short_code_block()
    preamble = "开头说明段落，介绍下文内容。" + "背景信息，" * 10
    after_prose = "代码块之后的第一段正文。" + "这是后续正文内容，" * 20
    return [
        ("1-原始bug-超长代码块+正文", preamble + "\n\n" + long_block + "\n\n" + after_prose),
        ("2-反例A-纯文本连续串+短文", "一" * 200 + "\n\n" + "H" * 400 + "\n\n" + "二" * 50),
        ("3-反例B-短代码块+300字长段", short_block + "\n\n" + "X" * 300),
        ("4-文末代码块", preamble + "\n\n" + long_block + "\n"),
        ("5-代码块后跟heading", short_block + "\n\n## Usage\n\n用法说明正文。"),
        ("6-短代码块+短文", short_block + "\n\n短文内容，合并场景。"),
    ]


# ---- 诊断 ----

def diagnose(chunks) -> list[str]:
    issues = []
    for chunk in chunks:
        content = chunk.content
        if len(content) <= 3:
            issues.append(f"micro-len={len(content)} {content!r}")
        fence_index = content.find("```")
        if fence_index > 0:
            issues.append(f"stray-fence@{fence_index} {content[:30]!r}")
    return issues


def render(chunks) -> list[str]:
    return [f"{chunk.chunk_id}({len(chunk.content)}):{ascii(chunk.content[:40])}" for chunk in chunks]


def main() -> None:
    variants = {
        "OLD": RecursiveCharacterTextSplitter(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
        "A swallow+merge": VariantA(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
        "B only-overlap": VariantB(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
        "C hard-split-local": VariantC(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
        "D swallow-only": VariantD(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
        "E C+ws-micro-merge": VariantE(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
        "F C+code-adjacent-ws": VariantF(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
    }
    for case_name, text in build_cases():
        print(f"\n=== {case_name} (len={len(text)}) ===")
        old_contents = None
        for label, splitter in variants.items():
            chunks = splitter.split(text, {"doc_id": "t"})
            contents = [chunk.content for chunk in chunks]
            if label == "OLD":
                old_contents = contents
            issues = diagnose(chunks)
            same = "  [SAME-OLD]" if contents == old_contents else ""
            issue_mark = f"  [issues: {'; '.join(issues)}]" if issues else ""
            print(f"  {label:<18} n={len(chunks)}{same}{issue_mark}")
            for line in render(chunks):
                print(f"      {line}")


if __name__ == "__main__":
    main()
