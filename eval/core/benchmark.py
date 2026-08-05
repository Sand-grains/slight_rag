"""Benchmark 加载、校验、默认值填充。

核心特性：
    - 从 JSON 文件加载 benchmark，支持 query_id / query / category / difficulty / relevance / reference_facts / expected_parent_ids / expected_child_ids 字段
    - 校验 expected_parent_ids 与 IndexStore 当前父块 chunk_id 集合的一致性（chunk 切分变更 → 校验失败，强制重新标注）
    - expected_chunk_ids 旧 key 自动兼容（读取时回退），expected_child_ids 为证据子块（可选，缺省空）
    - 缺失字段自动填充默认值，兼容旧版 benchmark 格式
    - ground_truth 字段自动转换为 reference_facts（向前兼容）

用法示例::

    from eval.core.benchmark import load_benchmark, BenchmarkItem, BenchmarkLoadResult
    result = load_benchmark("benchmark/private_v5.json", valid_chunk_ids=index_store.chunk_ids)
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
    expected_parent_ids: list[str]
    relevance: dict[str, int]  # parent chunk_id → 3/2/1
    expected_files: list[str]
    expected_pages: list[str]
    expected_child_ids: list[str] = field(default_factory=list)  # 证据子块（可选，缺省空）


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
        path: benchmark JSON 文件路径（数组格式）。
        valid_chunk_ids: 当前索引中实际存在的 chunk_id 集合，传入则校验 expected_chunk_ids 有效性。

    Returns:
        BenchmarkLoadResult：有效条目 + 缺失字段索引 + 无效 chunk_id 索引。
    """
    raw_items = _load_json(path)
    result = BenchmarkLoadResult()

    for index, item in enumerate(raw_items):
        has_query_id = "query_id" in item
        has_relevance = "relevance" in item
        has_category = "category" in item
        has_difficulty = "difficulty" in item

        if not has_query_id:
            result.missing_query_id.append(index)
        if not has_relevance:
            result.missing_relevance.append(index)
        if not has_category:
            result.missing_category.append(index)
        if not has_difficulty:
            result.missing_difficulty.append(index)

        query_id = item.get("query_id") or _auto_query_id(index)
        expected_parent_ids = item.get("expected_parent_ids") or item.get("expected_chunk_ids", [])
        relevance = item.get("relevance") or {chunk_id: 3 for chunk_id in expected_parent_ids}
        reference_facts = item.get("reference_facts") or item.get("ground_truth", "")

        # 校验 parent chunk_id 存在性
        if valid_chunk_ids is not None:
            missing = [chunk_id for chunk_id in expected_parent_ids if chunk_id not in valid_chunk_ids]
            if missing:
                result.invalid_chunk_ids[index] = missing

        result.valid_items.append(BenchmarkItem(
            query_id=query_id,
            query=item.get("query", ""),
            reference_facts=reference_facts,
            source_doc=item.get("source_doc", ""),
            category=item.get("category") or "unknown",
            difficulty=item.get("difficulty") or "unknown",
            expected_parent_ids=expected_parent_ids,
            relevance=relevance,
            expected_files=item.get("expected_files", []),
            expected_pages=item.get("expected_pages", []),
            expected_child_ids=item.get("expected_child_ids", []),
        ))

    return result


def _auto_query_id(index: int) -> str:
    """从数组索引自动生成 query_id，如索引 0 → Q0001。"""
    return f"Q{index + 1:04d}"


def _load_json(path: str) -> list:
    """加载 JSON 文件，相对路径基于项目根目录解析。

    Args:
        path: benchmark 文件路径（相对路径基于项目根目录）。

    Returns:
        list：JSON 数组内容。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = _PROJECT_ROOT / path_obj
    if not path_obj.exists():
        raise FileNotFoundError(f"Benchmark 文件不存在: {path}")
    with open(path_obj, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)
