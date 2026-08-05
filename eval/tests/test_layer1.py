"""Layer 1 检索评估端到端测试脚本。

手动验证检索管线 + IR 指标计算 + 诊断分类的端到端正确性。
"""
from datetime import datetime
from pathlib import Path

from eval.core.benchmark import load_benchmark
from eval.core.retrieval.retrieval_layer import run_retrieval_eval
from eval.reporter import generate_report, build_run_info
from indexing.index_store import IndexStore
from retrieval.retriever import Retriever

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # eval/tests/ → 项目根
_BENCHMARK_PATH = str(_PROJECT_ROOT / "benchmark" / "private_v5.json")

store = IndexStore.vector_restore(str(_PROJECT_ROOT / ".vector_cache"))
retriever = Retriever(store)

result = load_benchmark(_BENCHMARK_PATH, valid_chunk_ids=store.chunk_ids)
print(f"Items: {len(result.valid_items)}")

output = run_retrieval_eval(retriever, result.valid_items)

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
results_dir = str(_PROJECT_ROOT / "eval" / "results" / timestamp)
run_info = build_run_info(_BENCHMARK_PATH)
generate_report(output, results_dir, run_info)
print(f"Report: {results_dir}")
