import uuid
from dataclasses import dataclass, field


@dataclass
class Document:
    """RAG 管线中流转的通用文本容器，承载一段文本及其来源元数据"""
    content: str                                                   # 文本内容
    metadata: dict = field(default_factory=dict)                   # 来源信息
    doc_id: str = ""                                               # 原始文档ID, loader 阶段生成
    chunk_id: str = ""                                             # 切分后的片段ID, chunker 阶段生成
