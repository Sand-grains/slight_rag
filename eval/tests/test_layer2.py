"""Layer 2 Judge 单条试跑测试脚本。

手动验证 Generator 生成 + Judge 双调用 + 缓存查/写的端到端正确性。
"""
from pathlib import Path

from config import TOP_K, LLM_MODEL_ID
from eval.core.benchmark import load_benchmark
from eval.core.llm_as_judge.judge import run_judge
from indexing.index_store import IndexStore
from retrieval.retriever import Retriever
from retrieval.generator import Generator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # eval/tests/ → 项目根
_BENCHMARK_PATH = str(_PROJECT_ROOT / "benchmark" / "private_v5.json")

store = IndexStore.vector_restore(str(_PROJECT_ROOT / ".vector_cache"))
retriever = Retriever(store)
result = load_benchmark(_BENCHMARK_PATH, valid_chunk_ids=store.chunk_ids)

item = result.valid_items[0]
print(f"Q: {item.query}")
print(f"reference_facts: {item.reference_facts[:120]}...")
print(f"expected_parents: {item.expected_parent_ids}")
print(f"relevance: {item.relevance}")
print(f"category: {item.category}, difficulty: {item.difficulty}")

chunks = retriever.retrieve(item.query, top_k=TOP_K)
print(f"\n检索到 {len(chunks)} 个 chunk:")
for chunk in chunks:
    print(f"  [{chunk.chunk_id}]")

generator = Generator(model=LLM_MODEL_ID)
answer = generator.generate(item.query, chunks)
print(f"\n回答:\n{answer}")

print("\n执行 Judge...")
judge_result = run_judge(
    query_id=item.query_id,
    query=item.query,
    chunks=chunks,
    answer=answer,
    reference_facts=item.reference_facts,
)
print(f"\nfaithfulness: {judge_result.faithfulness}")
print(f"answer_relevancy: {judge_result.answer_relevancy}")
print(f"context_precision: {judge_result.context_precision}")
print(f"context_recall: {judge_result.context_recall}")
print(f"answer_correctness: {judge_result.answer_correctness}")
print(f"verdict: {judge_result.verdict}")
if judge_result.parse_error:
    print(f"parse_error: {judge_result.parse_error}")
if judge_result.grounded_claims:
    print(f"\ngrounded_claims ({len(judge_result.grounded_claims)}):")
    for claim in judge_result.grounded_claims[:5]:
        print(f"  [{claim.get('grounded')}] {claim.get('claim', '')[:80]}...")
