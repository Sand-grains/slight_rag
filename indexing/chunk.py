"""管线流转的通用文本数据模型。

核心特性：
    - Chunk 承载一段文本及其来源元数据，贯穿索引 → 检索 → 生成全管线
    - DocMetadata 作为共享引用挂在每个 Chunk 上，避免文档级信息冗余存储
    - chunk_id 确定性生成（{doc_id}:{chunk_index}），支持前后扩展定位

用法示例::

    from indexing import Chunk, DocMetadata
    meta = DocMetadata(title="README", doc_type=".md")
    c = Chunk(doc_id="docs/readme", chunk_id="docs/readme:0", content="# Hello", doc_meta=meta)

公共接口：
    - DocMetadata: 文档级元数据（title / author / source / source_url / doc_type / language / chunk_index）
    - Chunk: 通用文本容器（doc_id / chunk_id / content / retrieval_text / doc_meta / created_at）
"""

from dataclasses import dataclass, field


@dataclass
class DocMetadata:
    """文档级元数据（一篇文档一份，chunk 共享引用）"""
    title: str = ""              # 文档标题
    author: str = ""             # 作者（本地文档默认 "sd"）
    source: str = ""             # 来源路径/URL
    source_url: str = ""         # 原始链接（Web 来源溯源）
    doc_type: str = ""           # .txt / .md / .pdf / .docx / .xlsx
    language: str = "zh"         # 语言（预留，当前默认中文）
    chunk_index: int = 0         # 当前块在文档中的序号，支持前后扩展

    # ---- 后续父子检索时新增 ----
    # parent_chunk_id: str | None
    # child_chunk_ids: list[str]
    # page_number: int | None        # PDF 页码 (仅 PDF)
    # page_start: int | None
    # page_end: int | None


@dataclass
class Chunk:
    """管线流转的通用文本容器，承载一段文本及其来源元数据"""
    doc_id: str = ""             # 来源文档 ID (如 Knowledge/MainLine/RAG基础架构)
    chunk_id: str = ""           # 分块 ID, 确定性生成: {doc_id}:{chunk_index}
    content: str = ""            # chunk 原始文本（给 LLM）
    retrieval_text: str = ""     # 检索用文本（可选, 默认同 content, 后期可做预处理）
    doc_meta: DocMetadata = field(default_factory=DocMetadata)  # 文档级元数据（共享引用）
    created_at: str = ""         # 创建时间, 增量索引用

    # ---- Phase B 写入 Milvus 时新增 ----
    # embedding: list[float] | None   # 向量字段（当前由 VectorStore 单独管理）
