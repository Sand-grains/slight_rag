# =============================================================================
# 文件名:   pipeline.py
# 创建时间: 2026-07-15
# 所属版本: slight_rag v1.0（骨架版本）
# 功能:     端到端 RAG 入口脚本——加载 data/ 下的文档，走完"文档加载 → 文本切分
#           → Embedding 向量化 → 内存向量入库 → 用户提问 → 检索召回 → Promp  t
#           构造 → LLM 生成→ 打印结果"的全流程。所有环节均手写实现，未依赖
#           LangChain 或任何 RAG 框架。
# 局限性:   1. 硬编码串联流程，无 Agent 编排，无法自主决定是否检索
#           2. 单轮交互，无对话历史，无 Memory
#           3. 向量存储在内存，进程退出即丢失，不可持久化
#           4. 仅支持 .txt/.md，无 PDF 解析
#           5. 滑动窗口切分不感知文档结构（段落、句子边界）
#           6. 无 RRF 融合、Rerank 等精排环节
# 替代版本: v2 中由 agent_pipeline.py 取代，引入 SimpleAgent 做编排
# =============================================================================

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
