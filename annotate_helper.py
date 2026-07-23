"""标注辅助脚本：按 source_doc 分组，逐条标注 relevance / category / difficulty。

- 按 chunk_index（而非顺序序号）定位 chunk，输入 "20" 即指 chunk_id 为 "...:20" 的那块
- 每条 prompt 输入 e 可跳过当前条，已填的字段会保留
- 同一文档的 chunk 列表只打印一次，多个 question 共享
- 每条标注完毕立即写回 benchmark_private.json
"""

import json
import os
import sys
from collections import defaultdict

from retrieval.store import VectorStore


EXIT_KEY = "e"


def load_benchmark_raw(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_benchmark_raw(items: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"  [已保存] {path}\n")


def build_doc_index(store: VectorStore) -> dict[str, dict]:
    """构建 doc_id → {chunk_index: Chunk} 的映射。"""
    index: dict[str, dict[int, object]] = defaultdict(dict)
    for c in store.chunks:
        index[c.doc_id][c.doc_meta.chunk_index] = c
    return index


def is_annotated(item: dict) -> bool:
    """判断一条 benchmark 条目是否已完成标注。"""
    return bool(
        item.get("query_id")
        and item.get("category") and item.get("category") != "unknown"
        and item.get("difficulty") and item.get("difficulty") != "unknown"
        and item.get("relevance")
    )


VALID_CATEGORIES = ["Database", "Java", "Python", "Agent", "Middleware", "Distributed", "OS", "Programming", "Web"]
VALID_DIFFICULTIES = ["single_chunk", "multi_chunk", "cross_section"]


def prompt_field(prompt: str, current: str, valid_values: list[str] | None = None) -> str | None:
    """带当前值提示的单行输入。空输入 = 保留当前值。输入 e = 退出返回 None。"""
    hint = f"  (当前: {current})" if current else ""
    if valid_values:
        hint += f"  有效值: {valid_values}"
    raw = input(f"  {prompt}{hint}\n  > ").strip()
    if raw.lower() == EXIT_KEY:
        return None
    return raw if raw else current


def annotate_item(idx: int, item: dict, chunk_by_index: dict[int, object], source_doc: str):
    """标注单条 benchmark 条目。任意 prompt 输入 e 跳过当前条。"""
    print(f"\n  --- [{idx}] {item.get('query_id', '?')} ---")
    print(f"  Q: {item['question']}")

    ref = item.get("reference_facts") or item.get("ground_truth", "")
    if ref:
        print(f"  reference_facts: {ref}")

    # category
    val = prompt_field("Category?", item.get("category", ""), VALID_CATEGORIES)
    if val is None:
        print("  [跳过本条]\n")
        return
    item["category"] = val

    # difficulty
    val = prompt_field("Difficulty?", item.get("difficulty", ""), VALID_DIFFICULTIES)
    if val is None:
        print("  [跳过本条]\n")
        return
    item["difficulty"] = val

    # relevance（逐块标注：选 chunk → 看原文 → 打分 → 继续？y/n）
    new_relevance: dict[str, int] = dict(item.get("relevance", {}))
    while True:
        raw = input(
            "  选 chunk_index? (空=保留旧值并结束, e=跳过本条)\n"
            "  > "
        ).strip()

        if raw.lower() == EXIT_KEY:
            print("  [跳过本条]\n")
            return

        if not raw:
            break

        try:
            ci = int(raw)
        except ValueError:
            print(f"  ⚠ 无法解析: '{raw}'，请重试。\n")
            continue

        if ci not in chunk_by_index:
            print(f"  ⚠ chunk_index {ci} 不存在，请重试。\n")
            continue

        # 打印该 chunk 原文
        c = chunk_by_index[ci]
        print(f"\n  {'─'*60}")
        print(f"  [{ci}] {c.chunk_id}")
        print(f"  {'─'*60}")
        print(c.content)
        print(f"  {'─'*60}\n")

        # 打分
        rel_raw = input(
            f"  给 [{ci}] 打 relevance? (1/2/3, e=跳过本条)\n"
            "  > "
        ).strip()

        if rel_raw.lower() == EXIT_KEY:
            print("  [跳过本条]\n")
            return

        try:
            rel = int(rel_raw)
            if rel not in (1, 2, 3):
                print(f"  ⚠ relevance 应为 1/2/3，收到 {rel}，本次不计入。\n")
            else:
                new_relevance[chunk_by_index[ci].chunk_id] = rel
                print(f"  ✓ [{ci}] → relevance={rel}\n")
        except ValueError:
            print(f"  ⚠ 无法解析: '{rel_raw}'，本次不计入。\n")

        # 继续？
        cont = input("  继续选下一个 chunk? (y=继续 / n=确认结束, 写入)\n  > ").strip().lower()
        if cont == "n":
            break

    # 将本轮标注写入 item
    item["relevance"] = new_relevance
    item["expected_chunk_ids"] = sorted(new_relevance.keys())
    if "reference_facts" not in item and "ground_truth" in item:
        item["reference_facts"] = item.pop("ground_truth")

    if "query_id" not in item:
        item["query_id"] = f"Q{idx + 1:04d}"


_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_VECTOR_CACHE = os.path.join(_PROJECT_DIR, ".vector_cache")
_BENCHMARK_PATH = os.path.join(_PROJECT_DIR, "benchmark_private.json")


def main():
    if not os.path.exists(_BENCHMARK_PATH):
        print(f"错误: 找不到 {_BENCHMARK_PATH}")
        sys.exit(1)

    print("加载索引...")
    store = VectorStore.vector_restore(_VECTOR_CACHE)
    if store is None:
        print("错误: 索引缓存不存在，请先运行 agent_pipeline.py 构建索引")
        sys.exit(1)

    doc_index = build_doc_index(store)
    items = load_benchmark_raw(_BENCHMARK_PATH)

    # 按 source_doc 分组（保留原始顺序）
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, item in enumerate(items):
        source_doc = item.get("source_doc", "")
        groups[source_doc].append((idx, item))

    for source_doc, entries in groups.items():
        chunk_by_index = doc_index.get(source_doc)
        if chunk_by_index is None:
            for doc_id, ci_map in doc_index.items():
                if doc_id in source_doc or source_doc in doc_id:
                    chunk_by_index = ci_map
                    print(f"  匹配: source_doc='{source_doc}' → doc_id='{doc_id}'")
                    break
        if chunk_by_index is None:
            print(f"\n  ⚠ 未找到文档 '{source_doc}' 的 chunk，跳过 {len(entries)} 条")
            continue

        print(f"\n  文档: {source_doc}  chunk_index 范围: {min(chunk_by_index)} ~ {max(chunk_by_index)}  共 {len(chunk_by_index)} 个")
        print(f"  {len(entries)} 条待标注\n")

        for idx, item in entries:
            if is_annotated(item):
                continue
            annotate_item(idx, item, chunk_by_index, source_doc)
            save_benchmark_raw(items, _BENCHMARK_PATH)

        raw = input(f"  文档 '{source_doc}' 标注完毕，Enter 继续 / e 退出程序\n  > ").strip().lower()
        if raw == EXIT_KEY:
            print("退出。")
            return


if __name__ == "__main__":
    main()
