"""Layer 1 检索评估端到端测试。"""
from eval.core.benchmark import load_benchmark
from eval.core.retrieval_layer import run_layer1
from eval.reporter import generate_report, build_run_info
from retrieval.store import VectorStore
from retrieval.retriever import Retriever
import os
from datetime import datetime

store = VectorStore.vector_restore("D:/Pycharm/slight_rag/.vector_cache/")
retriever = Retriever(store)

result = load_benchmark("D:/Pycharm/slight_rag/benchmark_private.json", valid_chunk_ids=store.chunk_ids)
print(f"Items: {len(result.items)}")

output = run_layer1(retriever, result.items)

ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
results_dir = os.path.join("D:/Pycharm/slight_rag/eval/results", ts)
run_info = build_run_info("benchmark_private.json")
generate_report(output, results_dir, run_info)
print(f"Report: {results_dir}")
