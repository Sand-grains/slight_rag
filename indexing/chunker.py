"""滑动窗口文本切分，chunk_id 确定性生成。

核心特性：
    - 固定窗口大小 chunk_size，相邻窗口重叠 chunk_overlap 字符
    - 单文档长度 ≤ chunk_size 时不做切分，chunk_index 固定为 0
    - chunk_id 格式 {doc_id}:{chunk_index}，chunk_index 从 0 递增，切分策略不变时 ID 稳定

用法示例::

    from indexing import chunk
    from indexing import Chunk, DocMetadata
    docs = [Chunk(doc_id="doc/a", content="长文本...")]
    chunks = chunk(docs, chunk_size=500, chunk_overlap=100)

公共接口：
    - chunk: 将 Chunk 列表按滑动窗口切分为更小的 Chunk 列表
"""

from copy import deepcopy
from typing import List
from dataclasses import replace
from .chunk import Chunk
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk(docs: List[Chunk], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """滑动窗口切分：将每篇文档按固定窗口大小切为多个有重叠的 chunk, chunk_id 确定性生成"""
    chunks = []
    for doc in docs:
        text = doc.content
        if len(text) <= chunk_size:
            doc_meta = deepcopy(doc.doc_meta)
            doc_meta.chunk_index = 0
            chunks.append(replace(doc,
                chunk_id=f"{doc.doc_id}:0",
                doc_meta=doc_meta,
            ))
            continue

        step = chunk_size - chunk_overlap
        idx = 0
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            doc_meta = deepcopy(doc.doc_meta)
            doc_meta.chunk_index = idx
            chunks.append(replace(doc,
                content=chunk_text,
                retrieval_text=chunk_text,
                chunk_id=f"{doc.doc_id}:{idx}",
                doc_meta=doc_meta,
            ))
            idx += 1
            start += step

    return chunks
