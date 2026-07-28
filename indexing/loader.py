"""文档加载器：按文件后缀分发，产出未切分的原始 Chunk 列表。

核心特性：
    - 支持 .txt / .md 纯文本格式，一次性读取全文
    - doc_id 由文件相对于 data/ 的路径推导（去后缀、斜杠归一化）
    - 每个文件产出一个 Chunk，后续由 chunker 做滑动窗口切分

用法示例::

    from indexing import load
    chunks = load("data/Agent/README.md")  # → [Chunk(doc_id="Agent/README", ...)]

公共接口：
    - load: 按后缀分发加载，返回 Chunk 列表
"""

from pathlib import Path
from typing import List
from .chunk import Chunk, DocMetadata


def load(file_path: str, base_dir: str = "data") -> List[Chunk]:
    """根据文件后缀分发到对应的加载器，返回该文件产出的 Chunk 列表"""
    path = Path(file_path)             # 字符串路径 → Path 对象，便于取后缀和跨平台处理
    suffix = path.suffix.lower()       # 统一小写，避免 .TXT vs .txt 匹配失败
    base = Path(base_dir)

    if suffix in (".txt", ".md"):
        return _load_text(path, base)  # 纯文本类文件走同一个加载逻辑
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")


def _load_text(path: Path, base_dir: Path = Path("data")) -> List[Chunk]:
    """纯文本加载：一次性读取全文件，内容作为单个 Chunk 返回（后续由 chunker 切分）"""
    content = path.read_text(encoding="utf-8")
    try:
        relative = path.relative_to(base_dir)
        doc_id = str(relative.with_suffix("")).replace("\\", "/")
    except ValueError:
        doc_id = path.stem
    doc_meta = DocMetadata(
        title=path.stem,
        author="sd",                            # 本地文档默认作者
        source=str(path),
        doc_type=path.suffix.lower(),
    )
    return [Chunk(
        content=content,
        retrieval_text=content,
        doc_id=doc_id,
        doc_meta=doc_meta,
    )]
