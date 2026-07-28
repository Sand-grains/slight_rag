"""Layer 1 检索评估端到端测试脚本。

手动验证检索管线 + IR 指标计算 + 诊断分类的端到端正确性。
"""

from eval.core.benchmark import load_benchmark
from eval.core.retrieval_layer import run_retrieval_eval
from eval.reporter import generate_report, build_run_info
from retrieval.store import VectorStore
from retrieval.retriever import Retriever
import os
from datetime import datetime

store = VectorStore.vector_restore("D:/Pycharm/slight_rag/.vector_cache/")
retriever = Retriever(store)

result = load_benchmark("D:/Pycharm/slight_rag/benchmark_private.json", valid_chunk_ids=store.chunk_ids)
print(f"Items: {len(result.items)}")

output = run_retrieval_eval(retriever, result.items)

ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
results_dir = os.path.join("D:/Pycharm/slight_rag/eval/results", ts)
run_info = build_run_info("benchmark_private.json")
generate_report(output, results_dir, run_info)
print(f"Report: {results_dir}")
