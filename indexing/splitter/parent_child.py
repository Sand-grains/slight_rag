"""父子映射包装器：把父/子 splitter 的输出重写为父子双层层级。

本模块实现 ParentChildMappingWrapper：先用父 splitter 切出父块
（{doc_id}:p{i}），再对每个父块用子 splitter 切出子块
（{doc_id}:p{i}:c{j}），并在子块 metadata 记录 parent_id。这是 RAG
Parent-Child Chunking 的落地：子块做向量检索、父块做上下文生成。
"""
from indexing.chunk import Chunk
from .base import BaseSplitter


class ParentChildMappingWrapper(BaseSplitter):
    """父块 {doc_id}:p{i}，子块 {doc_id}:p{i}:c{j}；子块 metadata 记录 parent_id。"""

    def __init__(self, parent_splitter: BaseSplitter, child_splitter: BaseSplitter):
        super().__init__()
        self._parent_splitter = parent_splitter
        self._child_splitter = child_splitter

    def split(self, text: str, metadata: dict) -> tuple[list[Chunk], list[Chunk]]:
        """把文档切分为 (父块, 子块) 双层结果。

        Args:
            text: 待切分的整段文档文本。
            metadata: 文档元数据，须含 doc_id。

        Returns:
            (parents, children)：父块与子块两个列表。父块 id 形如
            f"{doc_id}:p{i}"，子块 id 形如 f"{doc_id}:p{i}:c{j}"，子块
            metadata 记录 parent_id 指向所属父块。
        """
        doc_id = metadata["doc_id"]
        parents = self._parent_splitter.split(text, metadata)
        for parent_index, parent in enumerate(parents):
            parent.chunk_id = f"{doc_id}:p{parent_index}"
            parent.origin_metadata.chunk_level = "parent"
        children = []
        for parent_index, parent in enumerate(parents):
            section_path = parent.metadata.get("section_path", [])
            sub = self._child_splitter.split(
                parent.content,
                {**metadata, "chunk_meta": {"section_path": section_path}},
            )
            for child_index, child in enumerate(sub):
                child.chunk_id = f"{doc_id}:p{parent_index}:c{child_index}"
                child.origin_metadata.chunk_level = "child"
                child.metadata["parent_id"] = parent.chunk_id
            children.extend(sub)
        return parents, children
