"""文档加载器：按文件后缀分发，产出未切分的原始 Chunk 列表。

核心特性：
    - 支持 .txt / .md 纯文本格式，一次性读取全文
    - doc_id 由文件相对于 data/ 的路径推导（去后缀、斜杠归一化）
    - 每个文件产出一个 Chunk，后续由 Router→Splitter 分块

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
    path = Path(file_path)             # 字符串路径 → Path 对象, 便于对其操作(如取后缀)
    suffix = path.suffix.lower()       # 后缀统一小写, 便于识别
    base = Path(base_dir)

    if suffix in (".txt", ".md"):# 仅支持.md/.txt(它们走同一个加载逻辑)
        return _load_text(path, base)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")


def _load_text(path: Path, base_dir: Path = Path("data")) -> List[Chunk]:
    """一次性读取全文件，内容作为单个大 Chunk 对象返回"""
    content = path.read_text(encoding="utf-8")
    try:
        relative = path.relative_to(base_dir)
        doc_id = str(relative.with_suffix("")).replace("\\", "/") # 推导doc_id
    except ValueError:
        doc_id = path.stem
    origin_metadata = DocMetadata(
        title=path.stem,
        author="sd", # 本地文档默认作者
        source=str(path),
        doc_type=path.suffix.lower(),
    )
    return [Chunk(
        content=content,
        doc_id=doc_id,
        origin_metadata=origin_metadata,
    )]
