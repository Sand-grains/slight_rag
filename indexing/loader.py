from pathlib import Path
from typing import List
from .document import Document


def load(file_path: str, base_dir: str = "data") -> List[Document]:
    """根据文件后缀分发到对应的加载器，返回该文件产出的 Document 列表"""
    path = Path(file_path)             # 字符串路径 → Path 对象，便于取后缀和跨平台处理
    suffix = path.suffix.lower()       # 统一小写，避免 .TXT vs .txt 匹配失败
    base = Path(base_dir)

    if suffix in (".txt", ".md"):
        return _load_text(path, base)  # 纯文本类文件走同一个加载逻辑
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")


def _load_text(path: Path, base_dir: Path = Path("data")) -> List[Document]:
    """纯文本加载：一次性读取全文件，内容作为单个 Document 返回"""
    content = path.read_text(encoding="utf-8")                     # 以 UTF-8 解码读全文件
    # 用 path 相对于 base_dir 的路径作为 doc_id（如 Knowledge/MainLine/RAG基础架构）
    try:
        relative = path.relative_to(base_dir)
        doc_id = str(relative.with_suffix("")).replace("\\", "/")
    except ValueError:
        doc_id = path.stem
    return [Document(
        content=content,
        metadata={"source": str(path)},
        doc_id=doc_id                                               # 相对路径 → 确定性 ID，人可读
    )]
