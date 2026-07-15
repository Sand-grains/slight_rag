import sys
from pathlib import Path
from indexing.loader import load
from indexing.chunker import chunk
from retrieval.embedding import embed
from retrieval.store import VectorStore
from retrieval.generator import generate

# ==================== 离线索引管线 ====================

# 1. 加载 data/ 目录下所有文档
data_dir = Path("data")
docs = []
for file_path in data_dir.glob("*"):
    if file_path.is_file() and file_path.suffix in (".txt", ".md"):
        docs.extend(load(str(file_path)))

if not docs:
    print("data/ 目录下没有找到 .txt 或 .md 文档，请放入测试文档后重试")
    exit(1)

print(f"已加载 {len(docs)} 篇文档")

# 2. 文本切分
chunks = chunk(docs)
print(f"切分后共 {len(chunks)} 个 chunk")

# 3. Embedding 向量化
texts = [c.content for c in chunks]
vectors = embed(texts)
print(f"向量化完成，维度: {len(vectors[0])}")

# 4. 向量入库
store = VectorStore()
store.add(chunks, vectors)
print("向量入库完成")

# ==================== 在线查询管线 ====================

# --- 调试用：命令行传参，正式使用时注释掉下面两行 ---
if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])               # 命令行传参：python pipeline.py 你的问题
# --- 调试用结束 ---
else:
    query = input("\n请输入你的问题: ")            # 交互式输入

# 5. Query 向量化
query_vector = embed([query])[0]

# 6. 检索召回
retrieved = store.search(query_vector)

# 7. LLM 生成
answer = generate(query, retrieved)

print("\n" + "=" * 50)
print(answer)
