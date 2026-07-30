"""文档索引管线：加载 → 分块，产出 Chunk 列表供检索管线消费。

公共接口：
    - Chunk: 通用文本容器（doc_id / chunk_id / content / origin_metadata）
    - DocMetadata: 文档级元数据（title / source / doc_type / chunk_index）
    - load: 文档加载（按后缀分发 .txt / .md）
    - chunk: 滑动窗口切分（chunk_size + chunk_overlap）
"""

from .chunk import Chunk, DocMetadata     # 文本容器 + 文档元数据
from .loader import load                    # 文档加载
from .chunker import chunk                  # 文本切分

# 外部导入目录indexing时, 其实就是导入其内部的这些命名空间
