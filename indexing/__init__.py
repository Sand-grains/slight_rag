from .chunk import Chunk, DocMetadata     # 文本容器 + 文档元数据
from .loader import load                    # 文档加载
from .chunker import chunk                  # 文本切分

# 外部导入目录indexing时, 其实就是导入其内部的这些命名空间
