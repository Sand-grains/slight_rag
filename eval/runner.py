"""CLI 主入口：test_mode 分支 + --compare 对比。

用法: cd D:\Pycharm\slight_rag
默认读 benchmark_private.json，可用 --benchmark 指定其他文件。结果写入 eval/results/<timestamp>/
    # 仅 Layer 1（检索指标，不调 LLM，免费）
    uv run python -m eval.runner --mode retrieval

    # 完整评估（Layer 1 + Layer 2，调 Judge LLM，计费）
    uv run python -m eval.runner --mode full

    # 对比两次运行
    uv run python -m eval.runner --compare <run_id_1> <run_id_2>

"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from retrieval.store import VectorStore
from retrieval.retriever import Retriever
from retrieval.generator import Generator
from config import TOP_K, LLM_MODEL_ID, EVAL_MAX_WORKERS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # eval/runner.py → eval/ → 项目根

_OUTER_POOL: ThreadPoolExecutor | None = None


def _get_outer_pool() -> ThreadPoolExecutor:
    """模块级外层线程池单例。限制同时运行的 query 数。"""
    global _OUTER_POOL
    if _OUTER_POOL is None:
        _OUTER_POOL = ThreadPoolExecutor(
            max_workers=EVAL_MAX_WORKERS,
            thread_name_prefix="eval-outer",
        )
    return _OUTER_POOL


def _evaluate_one(item, retriever, generator):
    """单条 query 的完整 eval。try/except 隔离，失败不影响其余 query。"""
    from eval.core.llm_as_judge import run_judge, JudgeResult
    from eval.core.formatter import get_formatter
    try:
        chunks = retriever.retrieve(item.query, top_k=TOP_K)
        answer = generator.generate(item.query, chunks)
        return run_judge(item.query_id, item.query, chunks, answer,
                         item.reference_facts, formatter=get_formatter())
    except Exception as e:
        jr = JudgeResult(query_id=item.query_id)
        jr.judge_error = f"evaluate_one failed: {e}"
        jr.verdict = "error"
        return jr


def run_retrieval_mode(benchmark_path: str):
    """仅对 Layer 1 retrieval 进行评估"""
    from eval.core.benchmark import load_benchmark
    from eval.core.retrieval_layer import run_retrieval_eval
    from eval.reporter import generate_report, build_run_info

    print("加载索引...")
    store = VectorStore.vector_restore()
    if store is None:
        print("错误: 索引缓存不存在")
        sys.exit(1)
    retriever = Retriever(store)

    print(f"加载 benchmark: {benchmark_path}")
    result = load_benchmark(benchmark_path, valid_chunk_ids=store.chunk_ids)
    print(f"  条目: {len(result.items)}")

    print("执行 Layer 1 检索评估...")
    output = run_retrieval_eval(retriever, result.items)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = str(_PROJECT_ROOT / "eval" / "results" / "timeline" / ts)
    run_info = build_run_info(benchmark_path)
    generate_report(output, results_dir, run_info)
    return results_dir


def run_full_mode(benchmark_path: str):
    """Layer 1 + Layer 2 完整评估。"""
    from eval.core.benchmark import load_benchmark
    from eval.core.retrieval_layer import run_retrieval_eval
    from eval.reporter import generate_report, build_run_info
    from eval.core.llm_as_judge import _get_client
    from eval.core.monitor_metrics import get_metrics, reset_metrics

    reset_metrics()

    print("加载索引...")
    store = VectorStore.vector_restore()
    if store is None:
        print("错误: 索引缓存不存在")
        sys.exit(1)
    retriever = Retriever(store)
    generator = Generator(model=LLM_MODEL_ID)

    # 主线程预初始化 OpenAI client，避免并行区域多线程竞态
    _get_client()

    print(f"加载 benchmark: {benchmark_path}")
    result = load_benchmark(benchmark_path, valid_chunk_ids=store.chunk_ids)
    items = result.items
    print(f"  条目: {len(items)}")

    print("执行 Layer 1 检索评估...")
    layer1 = run_retrieval_eval(retriever, items)

    print("执行 Layer 2 Judge...")
    judge_results = []
    pool = _get_outer_pool()
    futures = {}
    for item in items:
        f = pool.submit(_evaluate_one, item, retriever, generator)
        futures[f] = item

    for future in as_completed(futures):
        jr = future.result()
        judge_results.append(jr)
        done = len(judge_results)
        if done % 10 == 0:
            print(f"  {done}/{len(items)}")

    # 写入报告
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = str(_PROJECT_ROOT / "eval" / "results" / "timeline" / ts)
    run_info = build_run_info(benchmark_path, run_mode="full")
    generate_report(layer1, results_dir, run_info,
                    judge_results=judge_results,
                    metrics_summary=get_metrics().summary_dict())

    # 追加 Layer 2 数据到 per_query.json
    per_query_path = os.path.join(results_dir, "per_query.json")
    with open(per_query_path, "r", encoding="utf-8") as f:
        per_query = json.load(f)
    for jr in judge_results:
        if jr.query_id in per_query:
            per_query[jr.query_id].update({
                "faithfulness": jr.faithfulness,
                "answer_relevancy": jr.answer_relevancy,
                "context_precision": jr.context_precision,
                "context_recall": jr.context_recall,
                "answer_correctness": jr.answer_correctness,
                "verdict": jr.verdict,
                "parse_error": jr.parse_error,
                "judge_error": jr.judge_error,
                "generator_error": jr.generator_error,
            })
    with open(per_query_path, "w", encoding="utf-8") as f:
        json.dump(per_query, f, ensure_ascii=False, indent=2)

    # Layer 2 聚合
    valid = [jr for jr in judge_results if jr.faithfulness is not None]
    if valid:
        print(f"\nLayer 2 聚合 ({len(valid)}/{len(judge_results)} 条有效):")
        print(f"  faithfulness 均值:    {sum(j.faithfulness or 0 for j in judge_results) / len(valid):.4f}")
        print(f"  answer_relevancy 均值: {sum(j.answer_relevancy or 0 for j in judge_results) / len(valid):.4f}")
        print(f"  context_precision 均值: {sum(j.context_precision or 0 for j in judge_results) / len(valid):.4f}")
        print(f"  context_recall 均值:   {sum(j.context_recall or 0 for j in judge_results) / len(valid):.4f}")
        print(f"  answer_correctness 均值:{sum(j.answer_correctness or 0 for j in judge_results) / len(valid):.4f}")
        verdicts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
        for jr in judge_results:
            v = jr.verdict if jr.verdict in verdicts else "error"
            verdicts[v] += 1
        print(f"  verdict 分布: {verdicts}")

    get_metrics().print_summary()
    print(f"\n报告: {results_dir}")
    return results_dir


def run_compare(run_a: str, run_b: str):
    """对比两次运行的指标。"""
    base = str(_PROJECT_ROOT / "eval" / "results" / "timeline")
    dir_a = os.path.join(base, run_a)
    dir_b = os.path.join(base, run_b)

    summary_a = _load_summary(dir_a)
    summary_b = _load_summary(dir_b)

    if not summary_a or not summary_b:
        print("错误: 找不到 summary.json")
        sys.exit(1)

    agg_a = summary_a["aggregate"]
    agg_b = summary_b["aggregate"]

    metrics = ["recall_at_k", "precision_at_k", "hit_at_k", "mrr", "map_at_k", "ndcg_at_k"]
    print(f"\n{'Metric':<18} {'Run A':>10} {'Run B':>10} {'Delta':>10}")
    print("-" * 50)
    for m in metrics:
        va = agg_a.get(m, 0)
        vb = agg_b.get(m, 0)
        delta = vb - va
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else " ")
        print(f"{m:<18} {va:>10.4f} {vb:>10.4f} {arrow} {abs(delta):>8.4f}")


def _load_summary(results_dir: str) -> dict | None:
    path = os.path.join(results_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for noisy in ("httpx", "openai", "jieba", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="slight_rag eval runner")
    parser.add_argument("--mode", choices=["retrieval", "full"], help="评估模式")
    parser.add_argument("--benchmark", default="benchmark_private.json", help="benchmark 文件路径")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="对比两次运行")
    args = parser.parse_args()

    if args.compare:
        run_compare(args.compare[0], args.compare[1])
    elif args.mode == "retrieval":
        run_retrieval_mode(args.benchmark)
    elif args.mode == "full":
        run_full_mode(args.benchmark)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
