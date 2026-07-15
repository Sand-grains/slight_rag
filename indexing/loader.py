from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Document:
    """RAG 管线中流转的通用文本容器，承载一段文本及其来源元数据"""
    content: str                                       # 文本内容
    metadata: dict = field(default_factory=dict)       # 来源信息，如 {"source": "文件路径", "chunk_index": 0}


def load(file_path: str) -> List[Document]:
    """根据文件后缀分发到对应的加载器，返回该文件产出的 Document 列表"""
    path = Path(file_path)             # 字符串路径 → Path 对象，便于取后缀和跨平台处理
    suffix = path.suffix.lower()       # 统一小写，避免 .TXT vs .txt 匹配失败

    if suffix in (".txt", ".md"):
        return _load_text(path)        # 纯文本类文件走同一个加载逻辑
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")


def _load_text(path: Path) -> List[Document]:
    """纯文本加载：一次性读取全文件，内容作为单个 Document 返回"""
    content = path.read_text(encoding="utf-8")         # 以 UTF-8 解码读全文件
    return [Document(content=content, metadata={"source": str(path)})]
