"""Splitter 抽象基类：定义 split() 接口契约与 chunk 构建助手。

本模块提供所有切分器的抽象基类 BaseSplitter：声明 split() 契约——输入
整段文本 + 元数据，输出 Chunk 列表，父/子双层切分时输出
(parents, children)；并提供共用的建块工厂 _package_to_chunk，统一
doc_meta 深拷贝、start_char_index 注入与 chunk_id 留空三项约定。
"""
from abc import ABC, abstractmethod
from copy import deepcopy

from indexing.chunk import Chunk, DocMetadata


class BaseSplitter(ABC):
    """split(text, metadata) → list[Chunk] | (parents, children)。

    metadata 约定：{"doc_id": str, "doc_meta": DocMetadata, "chunk_meta": dict}
    chunk_meta 中的键并入 chunk.metadata（如 section_path / parent_id）
    """

    def _package_to_chunk(self, content: str, metadata: dict, start_char_index: int = 0) -> Chunk:
        """把切分产出的一个文本片段包装成 Chunk（所有 splitter 共用的建块工厂）。

        统一处理三件事：
        - doc_meta 深拷贝 → 每块独立的文档级元数据，改 chunk_level 不污染共享引用
        - 强制写入 start_char_index → 块在原文的起始偏移，永不缺失
        - chunk_id 留空 → 编号规则（p{i} / c{j}）由子类后填

        Args:
            content: 切分后的一个文本片段。
            metadata: 文档元数据，须含 doc_id、doc_meta、chunk_meta 键。
            start_char_index: 片段在原始文档中的起始字符偏移。

        Returns:
            Chunk：包装后的块。chunk_id 为空（由子类编号），doc_meta 为
            独立深拷贝，start_char_index 写入 chunk_meta。
        """
        doc_meta = deepcopy(metadata.get("doc_meta")) if metadata.get("doc_meta") else DocMetadata()
        chunk_meta = dict(metadata.get("chunk_meta", {}))
        chunk_meta["start_char_index"] = start_char_index
        return Chunk(
            doc_id=metadata["doc_id"],
            chunk_id="",
            content=content,
            origin_metadata=doc_meta,
            metadata=chunk_meta,
        )

    @abstractmethod
    def split(self, text: str, metadata: dict) -> list[Chunk] | tuple[list[Chunk], list[Chunk]]:
        """将整段文本切分为 Chunk 列表；父/子双层时返回 (parents, children)。

        Args:
            text: 待切分的整段文档文本。
            metadata: 文档元数据，键约定见类 docstring。

        Returns:
            list[Chunk] 或 (parents, children)：单层切分返回 Chunk 列表；
            父/子双层切分返回 (父块, 子块) 两个列表。
        """
        pass
