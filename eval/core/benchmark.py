"""Benchmark 加载、校验、默认值填充。

核心特性：
    - 从 JSON 文件加载 benchmark，支持 query_id / query / category / difficulty / relevance / reference_facts / expected_chunk_ids 字段
    - 校验 expected_chunk_ids 与 IndexStore 当前 chunk_id 集合的一致性（chunk 切分变更 → 校验失败，强制重新标注）
    - 缺失字段自动填充默认值，兼容旧版 benchmark 格式
    - ground_truth 字段自动转换为 reference_facts（向前兼容）

用法示例::

    from eval.core.benchmark import load_benchmark, BenchmarkItem, BenchmarkLoadResult
    result = load_benchmark("benchmark/private.json", valid_chunk_ids=store.chunk_ids)
    for item in result.valid_items:
        print(item.query_id, item.query, item.expected_chunk_ids)

公共接口：
    - BenchmarkItem: 单条 benchmark 条目（所有字段已填充默认值）
    - BenchmarkLoadResult: 加载结果（items + warnings + errors）
    - load_benchmark: 加载并校验 benchmark JSON 文件
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # eval/core/ → eval/ → 项目根


@dataclass
class BenchmarkItem:
    """单条 benchmark 条目, 所有字段已填充默认值。"""
    query_id: str
    query: str
    reference_facts: str
    source_doc: str
    category: str
    difficulty: str
    expected_chunk_ids: list[str]
    relevance: dict[str, int]  # chunk_id → 3/2/1
    expected_files: list[str]
    expected_pages: list[str]


@dataclass
class BenchmarkLoadResult:
    """加载结果：有效条目 + 校验信息。"""
    valid_items: list[BenchmarkItem] = field(default_factory=list)
    missing_query_id: list[int] = field(default_factory=list)       # 自动生成 query_id 的条目索引
    missing_relevance: list[int] = field(default_factory=list)      # 使用默认 relevance=3 的条目索引
    missing_category: list[int] = field(default_factory=list)       # 填充 "unknown" 的条目索引
    missing_difficulty: list[int] = field(default_factory=list)     # 填充 "unknown" 的条目索引
    invalid_chunk_ids: dict[int, list[str]] = field(default_factory=dict)  # 条目索引 → 无效 chunk_id 列表


def load_benchmark(path: str, valid_chunk_ids: set[str] | None = None) -> BenchmarkLoadResult:
    """加载 benchmark JSON 文件，校验并填充默认值。

    Args:
        path: benchmark JSON 文件路径（数组格式）
        valid_chunk_ids: 当前索引中实际存在的 chunk_id 集合，传入则校验 expected_chunk_ids 有效性
    """
    raw = _load_json(path)
    result = BenchmarkLoadResult()

    for idx, item in enumerate(raw):
        has_query_id = "query_id" in item
        has_relevance = "relevance" in item
        has_category = "category" in item
        has_difficulty = "difficulty" in item

        if not has_query_id:
            result.missing_query_id.append(idx)
        if not has_relevance:
            result.missing_relevance.append(idx)
        if not has_category:
            result.missing_category.append(idx)
        if not has_difficulty:
            result.missing_difficulty.append(idx)

        query_id = item.get("query_id") or _auto_query_id(idx)
        expected_chunk_ids = item.get("expected_chunk_ids", [])
        relevance = item.get("relevance") or {cid: 3 for cid in expected_chunk_ids}
        reference_facts = item.get("reference_facts") or item.get("ground_truth", "")

        # 校验 chunk_id 存在性
        if valid_chunk_ids is not None:
            missing = [cid for cid in expected_chunk_ids if cid not in valid_chunk_ids]
            if missing:
                result.invalid_chunk_ids[idx] = missing

        result.valid_items.append(BenchmarkItem(
            query_id=query_id,
            query=item.get("query", ""),
            reference_facts=reference_facts,
            source_doc=item.get("source_doc", ""),
            category=item.get("category") or "unknown",
            difficulty=item.get("difficulty") or "unknown",
            expected_chunk_ids=expected_chunk_ids,
            relevance=relevance,
            expected_files=item.get("expected_files", []),
            expected_pages=item.get("expected_pages", []),
        ))

    return result


def _auto_query_id(index: int) -> str:
    """从数组索引自动生成 query_id，如索引 0 → Q0001。"""
    return f"Q{index + 1:04d}"


def _load_json(path: str):
    """加载 JSON 文件，相对路径基于项目根目录解析。"""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Benchmark 文件不存在: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
