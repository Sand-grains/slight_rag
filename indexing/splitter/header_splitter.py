"""标题切分器：按 Markdown 标题树把文档切分为 section（父块粒度）。

本模块实现 HeadingSplitter：解析标题层级生成 section_path 路径栈，把
文档切分为若干 section（父块粒度）；section 超过 max_section_chars 时
内部用 RecursiveCharacterTextSplitter 再切为多个父块。标题解析跳过
fenced code block 内的 "#" 行，孤立标题并入前一个 section。
"""
from indexing.chunk import Chunk
from .base import BaseSplitter
from .recursive_splitter import RecursiveCharacterTextSplitter
from .utils import _HEADING_REGEX, find_fenced_block_ranges, inside_code


class HeadingSplitter(BaseSplitter):
    """按 Markdown 标题切分为 section（父块粒度）, section 超 max_section_chars 时按上限切为多个父块。"""

    def __init__(self, rules: list[tuple[str, str]], max_section_chars: int):
        super().__init__()
        self._rules = rules
        self._max_section_chars = max_section_chars
        self._overflow_splitter = RecursiveCharacterTextSplitter(max_section_chars, 0, chunk_level="parent")

    def split(self, text: str, metadata: dict) -> list[Chunk]:
        """把整段文档按标题树切分为父块 Chunk 列表。

        Args:
            text: 待切分的整段文档文本。
            metadata: 文档元数据，须含 doc_id；section_path 会写入
                每个 chunk 的 chunk_meta。

        Returns:
            list[Chunk]：父块列表，chunk_level 固定为 "parent"。常规
            section 的 chunk_id 形如 f"{doc_id}:p{i}"；超长 section 由
            内部 _overflow_splitter 再切分（沿用其自身编号）。
        """
        code_ranges = find_fenced_block_ranges(text)
        sections = self._build_sections(text, code_ranges)
        chunks = []
        for section_index, (path, content, start) in enumerate(sections):
            if len(content) <= self._max_section_chars:
                chunk = self._package_to_chunk(content, {**metadata, "chunk_meta": {"section_path": path}},
                                               start_char_index=start)
                chunk.chunk_id = f"{metadata['doc_id']}:p{section_index}"
                chunk.origin_metadata.chunk_level = "parent"
                chunks.append(chunk)
            else:
                sub = self._overflow_splitter.split(content, {**metadata, "chunk_meta": {"section_path": path}})
                chunks.extend(sub)
        return chunks

    # ---- section 构建 ----

    def _build_sections(self, text: str, code_ranges: list[tuple[int, int]]) -> list[tuple[list[str], str, int]]:
        """按标题解析结果把文档切分为 section，并合并孤立标题。

        Args:
            text: 待切分的整段文档文本。
            code_ranges: fenced code block 的全局 [(start, end), ...] 区间列表。

        Returns:
            list[(section_path, content, start)]：每个 section 的标题路径、
            内容与全局起始偏移；无标题时整篇作为单个空路径 section。
        """
        items = self._parse_headings_with_paths(text, code_ranges)
        if not items:
            return [([], text, 0)]
        sections = []
        if items[0][1] > 0:
            preamble = text[:items[0][1]]
            if preamble.strip():
                sections.append(([], preamble, 0))
        for heading_index, (path, start) in enumerate(items):
            end = items[heading_index + 1][1] if heading_index + 1 < len(items) else len(text)
            sections.append((path, text[start:end], start))
        return self._merge_orphan_sections(sections)

    def _parse_headings_with_paths(self, text: str, code_ranges: list[tuple[int, int]]
                                   ) -> list[tuple[list[str], int]]:
        """解析标题并构造每个标题的层级路径（section_path 栈）。

        逐行扫描文本，跳过 fenced code block 内的行；对每个命中的标题，
        用单调栈维护从根到当前的标题路径 [h1, h2, ...]。

        Args:
            text: 待解析的整段文档文本。
            code_ranges: fenced code block 的全局 [(start, end), ...] 区间列表。

        Returns:
            list[(path, start)]：每个标题的路径栈与全局起始偏移。
        """
        max_level = max((len(prefix) for prefix, _ in self._rules), default=6)
        headings = []
        pos = 0
        for raw_line in text.splitlines(keepends=True):
            line = raw_line.rstrip("\r\n")
            if not inside_code(pos, code_ranges):
                heading_match = _HEADING_REGEX.match(line)
                if heading_match and len(heading_match.group(1)) <= max_level:
                    headings.append((len(heading_match.group(1)), heading_match.group(2).strip(), pos))
            pos += len(raw_line)
        stack: list[tuple[int, str]] = []
        result = []
        for level, heading_text, start in headings:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
            result.append(([heading for _, heading in stack], start))
        return result

    def _merge_orphan_sections(self, sections: list[tuple[list[str], str, int]]
                               ) -> list[tuple[list[str], str, int]]:
        """把"只有标题、没有正文"的孤立 section 并入前一个 section。

        Args:
            sections: 切分后的 (path, content, start) section 列表。

        Returns:
            list[(path, content, start)]：孤立 section（首行是标题且无
            正文）并入前一个 section（换行连接），其余原样保留。
        """
        merged = []
        for path, content, start in sections:
            body = content.split("\n", 1)[1] if "\n" in content else ""
            is_lone_heading = body.strip() == "" and content.strip() != ""
            if is_lone_heading and merged:
                prev_path, prev_content, prev_start = merged[-1]
                merged[-1] = (prev_path, prev_content + "\n" + content, prev_start)
            else:
                merged.append((path, content, start))
        return merged
