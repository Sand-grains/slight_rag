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
