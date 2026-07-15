from typing import List
from .loader import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """滑动窗口切分：将每篇文档按固定窗口大小切为多个有重叠的 chunk"""
    chunks = []  # chunks内是一些Document对象, 代表一个个chunk
    for doc in documents:  # 一份份处理单篇文档
        text = doc.content  # 取文档中的content
        if len(text) <= chunk_size:  # 如果text长度小于chunk, 这个分块就不用切
            chunks.append(Document(
                content=text,
                metadata={**doc.metadata, "chunk_index": 0}  # 继承原 metadata 并追加块内序号
            ))
            continue

        # 文档长于chunk_size，需要切分
        step = chunk_size - chunk_overlap  # 窗口每次向前滑动的字符数（500-100=400，保证相邻 chunk 重叠 100 字符）
        idx = 0          # chunk_index，记录当前是这篇文档内的第几块
        start = 0        # 当前窗口的起始位置（字符索引）
        while start < len(text):  # 注意这里是while循环
            end = min(start + chunk_size, len(text))  # 取 start+500，但不超过文本末尾（处理最后一段不足 500 的情况）
            chunk_text = text[start:end]  # 切出当前窗口的字符, 追加到下一个chunk
            chunks.append(Document(
                content=chunk_text,
                metadata={**doc.metadata, "chunk_index": idx}
            ))
            idx += 1
            start += step  # 窗口前移 step 个字符

    return chunks
