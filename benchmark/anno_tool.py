"""Benchmark 标注工具：逐块标注 expected_chunk_ids + relevance + difficulty。

特性：
  - 启动时可指定从第 N 条开始（断点续标）
  - 逐块展示 chunk 全文，Enter 标注 / e 跳过 / m 结束本条

用法:  uv run python benchmark/anno_tool.py
"""

import json
import os
import sys
from collections import defaultdict

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmark/ → 项目根
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from retrieval.store import IndexStore
from config import VECTOR_CACHE_DIR

EXIT_KEY = "e"
DONE_KEY = "m"

VALID_DIFFICULTIES = ["single_chunk", "multi_chunk"]

_BENCHMARK_PATH = os.path.join(_PROJECT_DIR, "benchmark", "private.json")


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

    Enter → 标注当前 chunk（输入 relevance 1/2/3），然后下一块
    e     → 跳过当前块，下一块
    m     → 结束本条标注，返回已标注的 chunk
    返回 list[dict]：已标注的 {chunk_id, relevance}。
    """
    sorted_indices = sorted(chunk_by_index.keys())
    annotations: list[dict] = []

    i = 0
    while i < len(sorted_indices):
        ci = sorted_indices[i]
        c = chunk_by_index[ci]

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

        raw = input("  Enter=标注  e=跳过  m=结束本条  > ").strip().lower()
        if raw == EXIT_KEY:
            i += 1
            continue
        if raw == DONE_KEY:
            break

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
    print(f"  Enter=标注  e=跳过  m=结束本条")
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

    print(f"\n  ✓ [{qid}] 标注完成: {len(new_chunk_ids)} chunks, difficulty={current_diff}\n")
    return True


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

    total = len(items)
    unannotated = sum(1 for e in items if not e.get("relevance"))
    annotated = total - unannotated
    print(f"总条目: {total}  已标注: {annotated}  未标注: {unannotated}")

    # 询问起始位置
    raw = input("\n是否从头开始标注? (y=从头开始 / n=指定起始位置)\n  > ").strip().lower()
    start_idx = 0
    if raw == "n":
        n_raw = input(f"  从第几条开始? (1~{total})\n  > ").strip()
        try:
            n = int(n_raw)
            if 1 <= n <= total:
                start_idx = n - 1
            else:
                print(f"  超出范围 (1~{total})，从头开始")
        except ValueError:
            print(f"  无效输入，从头开始")

    for i in range(start_idx, len(items)):
        entry = items[i]
        qid = entry.get("query_id", f"Q{i+1:04d}")
        source_doc = entry.get("source_doc", "")

        chunk_by_index = doc_index.get(source_doc)
        if chunk_by_index is None:
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
