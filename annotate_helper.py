"""Benchmark 标注辅助脚本：逐条逐块标注 expected_chunk_ids + relevance + difficulty。

流程（以 query_id 为单元，从 Q0001 开始）：
  1. 检测该条目是否已有 expected_chunk_ids，若有则询问是否覆盖
  2. 逐块打印 source_doc 的全部 chunk 全文（Enter 下一块，e 跳过）
  3. 打印 query 和 reference_facts
  4. 输入数字索引（自动拼接完整 chunk_id）+ 立即打 relevance（1/2/3）
     u 撤销上一组，d 完成
  5. 询问 difficulty
  6. 保存，进入下一条

用法:  uv run python annotate_helper.py
"""

import json
import os
import sys
from collections import defaultdict

from retrieval.store import IndexStore
from config import VECTOR_CACHE_DIR

EXIT_KEY = "e"

VALID_DIFFICULTIES = ["single_chunk", "multi_chunk"]

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARK_PATH = os.path.join(_PROJECT_DIR, "benchmark_private.json")
_RELIABILITY_PATH = os.path.join(_PROJECT_DIR, "auto_reliability.md")


# ---------------------------------------------------------------------------
# 加载 / 保存
# ---------------------------------------------------------------------------

def load_benchmark(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_benchmark(items: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"  [已保存] {path}\n")


# ---------------------------------------------------------------------------
# chunk 索引
# ---------------------------------------------------------------------------

def build_doc_index(store: IndexStore) -> dict[str, dict[int, object]]:
    """doc_id → {chunk_index: Chunk}"""
    index: dict[str, dict[int, object]] = defaultdict(dict)
    for c in store.chunks:
        index[c.doc_id][c.origin_metadata.chunk_index] = c
    return index


# ---------------------------------------------------------------------------
# 逐块打印全文
# ---------------------------------------------------------------------------

def print_chunks(chunk_by_index: dict[int, object], query: str, reference_facts: str, source_doc: str) -> list[dict]:
    """逐块打印全文，每块顶部显示 query + reference_facts 供对照。

    Enter → 立即标注当前 chunk（输入 relevance 1/2/3），然后下一块
    e → 跳过当前块，下一块
    返回 list[dict]：已标注的 {chunk_id, relevance}。
    """
    sorted_indices = sorted(chunk_by_index.keys())
    annotations: list[dict] = []

    i = 0
    while i < len(sorted_indices):
        ci = sorted_indices[i]
        c = chunk_by_index[ci]

        # 清屏
        os.system("cls" if os.name == "nt" else "clear")

        print(f"Q: {query}")
        if reference_facts:
            print(f"reference_facts: {reference_facts}")
        print()

        print(f"[chunk {ci}/{len(sorted_indices) - 1}]  {c.chunk_id}")
        print("─" * 70)
        print(c.content)
        print("─" * 70)

        if annotations:
            labeled = [f"[{a['chunk_id'].split(':')[-1]}]={a['relevance']}" for a in annotations]
            print(f"  已标注: {', '.join(labeled)}")

        raw = input("  Enter=标注  e=跳过  > ").strip().lower()
        if raw == EXIT_KEY:
            i += 1
            continue

        # Enter / 其他 → 标注
        rel_raw = input(f"  [{ci}] relevance? (1/2/3, 空=跳过)\n  > ").strip()
        if rel_raw:
            try:
                rel = int(rel_raw)
                if rel in (1, 2, 3):
                    full_id = f"{source_doc}:{ci}"
                    annotations.append({"chunk_id": full_id, "relevance": rel})
                    print(f"  ✓ [{ci}] {full_id}  relevance={rel}")
                else:
                    print(f"  ⚠ relevance 应为 1/2/3，本次跳过")
            except ValueError:
                print(f"  ⚠ 无法解析: '{rel_raw}'，本次跳过")
        i += 1

    # 最后清屏
    os.system("cls" if os.name == "nt" else "clear")
    return annotations


# ---------------------------------------------------------------------------
# 单条标注
# ---------------------------------------------------------------------------

def annotate_entry(
    entry: dict,
    chunk_by_index: dict[int, object],
    source_doc: str,
) -> bool:
    """标注单条 entry。返回 False 表示退出程序。"""
    qid = entry.get("query_id", "?")
    existing_ids = entry.get("expected_chunk_ids", [])
    ref = entry.get("reference_facts") or entry.get("ground_truth", "")

    # 捕获 AI 标注：relevance 为空说明 AI 未标注，本条的 expected_chunk_ids 即 AI 预测结果
    is_ai_entry = not entry.get("relevance")
    ai_chunks = list(existing_ids) if is_ai_entry else None

    # ---- 步骤 1: 已有标注时询问是否重新输入 ----
    if existing_ids:
        print(f"\n{'=' * 60}")
        print(f"[{qid}]  source_doc: {source_doc}")
        print(f"已有 expected_chunk_ids: {existing_ids}")
        print(f"已有 relevance: {entry.get('relevance', {})}")
        print(f"{'=' * 60}")
        raw = input("  是否重新输入? (y=覆盖 / n=跳过 / q=退出)\n  > ").strip().lower()
        if raw == "q":
            return False
        if raw == "n":
            return True

    # ---- 步骤 2: 逐块打印 + 边看边标 ----
    print(f"\n{'─' * 70}")
    print(f"[{qid}]  逐块打印 {source_doc} 全文")
    print(f"  Enter=标注当前块   e=跳过")
    input("  按 Enter 开始...")

    annotations = print_chunks(chunk_by_index, entry["query"], ref, source_doc)

    # ---- 步骤 3: 汇总 ----
    print("=" * 60)
    print(f"[{qid}]  source_doc: {source_doc}")
    print(f"Q: {entry['query']}")
    if ref:
        print(f"reference_facts: {ref}")
    print("=" * 60)

    new_relevance: dict[str, int] = {}
    new_chunk_ids: list[str] = []
    for a in annotations:
        new_chunk_ids.append(a["chunk_id"])
        new_relevance[a["chunk_id"]] = a["relevance"]

    if new_chunk_ids:
        print(f"\n  本次标注 ({len(new_chunk_ids)} 个):")
        for cid in new_chunk_ids:
            idx_part = cid.split(":")[-1] if ":" in cid else cid
            print(f"    [{idx_part}] {cid}  relevance={new_relevance[cid]}")
    else:
        print(f"\n  本次标注: 0 个 chunk")

    # ---- 步骤 4: 询问 difficulty ----
    current_diff = entry.get("difficulty", "single_chunk")
    if current_diff not in VALID_DIFFICULTIES:
        current_diff = "single_chunk"
    raw = input(f"\n  Difficulty? ({'/'.join(VALID_DIFFICULTIES)}, 空=保留 '{current_diff}')\n  > ").strip()
    if raw and raw in VALID_DIFFICULTIES:
        current_diff = raw
    elif raw and raw not in VALID_DIFFICULTIES:
        print(f"  ⚠ 无效值: '{raw}'，保留 '{current_diff}'")

    # ---- 写入 ----
    entry["expected_chunk_ids"] = new_chunk_ids
    entry["relevance"] = new_relevance
    entry["difficulty"] = current_diff
    if "query_id" not in entry:
        entry["query_id"] = "Q0000"

    # 计算 AI 标注可信度
    if ai_chunks is not None:
        jac = compute_jaccard(ai_chunks, new_chunk_ids)
        save_reliability(qid, jac)
        print(f"  AI 可信度 (Jaccard): {jac:.3f}")

    print(f"\n  ✓ [{qid}] 标注完成: {len(new_chunk_ids)} chunks, difficulty={current_diff}\n")
    return True


# ---------------------------------------------------------------------------
# AI 标注可信度
# ---------------------------------------------------------------------------

def compute_jaccard(ai_chunks: list[str], human_chunks: list[str]) -> float:
    """两个 chunk_id 集合的 Jaccard 相似度。两者均为空 → 1.0；一者为空 → 0.0。"""
    ai_set = set(ai_chunks)
    human_set = set(human_chunks)
    union = ai_set | human_set
    if not union:
        return 1.0
    return len(ai_set & human_set) / len(union)


def load_reliability() -> list[dict]:
    """从 auto_reliability.md 读取已有记录。"""
    records = []
    if os.path.exists(_RELIABILITY_PATH):
        with open(_RELIABILITY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("---"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        records.append({
                            "query_id": parts[0],
                            "jaccard": float(parts[1]),
                            "time": " ".join(parts[2:]),
                        })
                    except ValueError:
                        pass
    return records


def save_reliability(query_id: str, jaccard: float):
    """追加/更新单条可信度记录，重写 auto_reliability.md。"""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    records = load_reliability()
    updated = False
    for r in records:
        if r["query_id"] == query_id:
            r["jaccard"] = jaccard
            r["time"] = now
            updated = True
            break
    if not updated:
        records.append({"query_id": query_id, "jaccard": jaccard, "time": now})

    with open(_RELIABILITY_PATH, "w", encoding="utf-8") as f:
        f.write("# AI Auto-Annotation Reliability\n")
        f.write("# Format: query_id jaccard time\n\n")
        for r in records:
            f.write(f"{r['query_id']} {r['jaccard']:.3f} {r['time']}\n")
        if records:
            mean_jac = sum(r["jaccard"] for r in records) / len(records)
            f.write(f"\n---\n{mean_jac:.3f}\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(_BENCHMARK_PATH):
        print(f"错误: 找不到 {_BENCHMARK_PATH}")
        sys.exit(1)

    print("加载索引...")
    store = IndexStore.vector_restore(VECTOR_CACHE_DIR)
    if store is None:
        print("错误: 索引缓存不存在，请先运行 agent_pipeline.py 构建索引")
        sys.exit(1)

    doc_index = build_doc_index(store)
    items = load_benchmark(_BENCHMARK_PATH)

    # 统计
    total = len(items)
    unannotated = sum(1 for e in items if not e.get("relevance"))
    annotated = total - unannotated
    print(f"总条目: {total}  已标注: {annotated}  未标注: {unannotated}")

    # 逐条处理（按 query_id 顺序）
    for i, entry in enumerate(items):
        qid = entry.get("query_id", f"Q{i+1:04d}")
        source_doc = entry.get("source_doc", "")

        # 查找 chunk 索引
        chunk_by_index = doc_index.get(source_doc)
        if chunk_by_index is None:
            # 模糊匹配
            for doc_id, ci_map in doc_index.items():
                if doc_id in source_doc or source_doc in doc_id:
                    chunk_by_index = ci_map
                    print(f"  匹配: source_doc='{source_doc}' → doc_id='{doc_id}'")
                    break

        if chunk_by_index is None:
            print(f"\n  ⚠ [{qid}] 未找到 source_doc='{source_doc}' 的 chunk，跳过")
            continue

        print(f"\n{'#' * 60}")
        print(f"# [{i + 1}/{total}]  {qid}  source_doc: {source_doc}")
        print(f"# chunks: {len(chunk_by_index)} ({min(chunk_by_index)} ~ {max(chunk_by_index)})")
        print(f"{'#' * 60}")

        should_continue = annotate_entry(entry, chunk_by_index, source_doc)
        save_benchmark(items, _BENCHMARK_PATH)

        if not should_continue:
            print("退出标注。")
            return


if __name__ == "__main__":
    main()
