"""阈值校准诊断: 输出 benchmark 前 N 条的 GT-chunk 余弦相似度平均分位数"""
import json
import numpy
from pathlib import Path
from indexing.loader import load
from indexing.chunker import chunk
from retrieval.embedding import embed
from retrieval.store import VectorStore
from config import RELEVANCE_THRESHOLD

N = 10

# ---- 加载或构建索引 ----
store = VectorStore.vector_restore()
if store is None:
    store = VectorStore()
    for file_path in Path("data").rglob("*"):
        if file_path.is_file() and file_path.suffix in (".txt", ".md"):
            docs = load(str(file_path))
            chunks = chunk(docs)
            vectors = embed([c.content for c in chunks])
            store.add(chunks, vectors)
    store.vector_persistence()

# ---- 取 benchmark 前 N 条 ----
with open("benchmark.json", "r", encoding="utf-8") as f:
    questions = json.load(f)[:N]

gt_texts = [q["ground_truth"] for q in questions]
gt_vectors = numpy.array(embed(gt_texts))                       # (N, dim)
all_scores = numpy.dot(store._vectors, gt_vectors.T)            # (chunks, N)

# ---- 每道题各分位线 → 跨题求均值 ----
PERCENTILES = [50, 60, 70, 75, 80, 85, 90, 95, 99]

print(f"{'threshold':>12}: {RELEVANCE_THRESHOLD}")
print(f"{'chunks':>12}: {len(store._chunks)}")
print(f"{'questions':>12}: {N}\n")

# 收集每道题各分位线的值
p_values = {p: [] for p in PERCENTILES}
above_counts = []

for i in range(N):
    scores = all_scores[:, i]
    for p in PERCENTILES:
        p_values[p].append(numpy.percentile(scores, p))
    above_counts.append((scores >= RELEVANCE_THRESHOLD).sum())

# 均值
print(f"  {'avg_P':>8}  {'value':>8}")
print(f"  {'-'*8}  {'-'*8}")
for p in PERCENTILES:
    print(f"  {'avg_P'+str(p):>8}: {numpy.mean(p_values[p]):.4f}")

# >= threshold 的 chunk 占比
avg_above = numpy.mean(above_counts)
print(f"\n  >= {RELEVANCE_THRESHOLD}: avg {avg_above:.0f}/{len(store._chunks)} chunks ({avg_above/len(store._chunks)*100:.1f}%)")
