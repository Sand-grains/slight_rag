"""Layer 2 Judge 单条试跑测试。"""
from eval.core.benchmark import load_benchmark
from eval.core.llm_as_judge import run_judge
from retrieval.store import VectorStore
from retrieval.retriever import Retriever
from retrieval.generator import Generator
from config import TOP_K, LLM_MODEL_ID

store = VectorStore.vector_restore("D:/Pycharm/slight_rag/.vector_cache/")
retriever = Retriever(store)
result = load_benchmark("D:/Pycharm/slight_rag/benchmark_private.json", valid_chunk_ids=store.chunk_ids)

item = result.items[0]
print(f"Q: {item.query}")
print(f"reference_facts: {item.reference_facts[:120]}...")
print(f"expected_chunks: {item.expected_chunk_ids}")
print(f"relevance: {item.relevance}")
print(f"category: {item.category}, difficulty: {item.difficulty}")

chunks = retriever.retrieve(item.query, top_k=TOP_K)
print(f"\n检索到 {len(chunks)} 个 chunk:")
for c in chunks:
    print(f"  [{c.chunk_id}]")

generator = Generator(model=LLM_MODEL_ID)
answer = generator.generate(item.query, chunks)
print(f"\n回答:\n{answer}")

print("\n执行 Judge...")
jr = run_judge(
    query_id=item.query_id,
    query=item.query,
    chunks=chunks,
    answer=answer,
    reference_facts=item.reference_facts,
)
print(f"\nfaithfulness: {jr.faithfulness}")
print(f"answer_relevancy: {jr.answer_relevancy}")
print(f"context_precision: {jr.context_precision}")
print(f"context_recall: {jr.context_recall}")
print(f"answer_correctness: {jr.answer_correctness}")
print(f"verdict: {jr.verdict}")
if jr.parse_error:
    print(f"parse_error: {jr.parse_error}")
if jr.grounded_claims:
    print(f"\ngrounded_claims ({len(jr.grounded_claims)}):")
    for gc in jr.grounded_claims[:5]:
        print(f"  [{gc.get('grounded')}] {gc.get('claim', '')[:80]}...")
