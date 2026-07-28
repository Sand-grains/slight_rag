"""Eval CLI 主入口：retrieval / full 两种模式 + --compare 对比。

核心特性：
    - --mode retrieval：仅 Layer 1 检索评估（不调 LLM，免费），LivePanel.render_final() 输出终端报告
    - --mode full：Layer 1 + Layer 2 完整评估（调 Judge LLM，计费），LivePanel daemon 实时面板 + 最终报告
    - --compare RUN_A RUN_B：对比两次运行的指标差异
    - 外层 ThreadPoolExecutor（max_workers=5）控 query 级并发，_evaluate_one 做 per-query 隔离
    - _load_previous_run() 加载最近一次 per_query.json 作为 Delta 基线

用法示例::

    uv run python -m eval.runner --mode retrieval
    uv run python -m eval.runner --mode full
    uv run python -m eval.runner --compare 2026-07-26_120000 2026-07-26_150000

公共接口：
    - run_retrieval_mode: Layer 1 检索评估
    - run_full_mode: Layer 1 + Layer 2 完整评估（含 LivePanel）
    - run_compare: 对比两次运行
    - main: argparse CLI 入口

默认读 benchmark_private.json，可用 --benchmark 指定其他文件。结果写入 eval/results/<timestamp>/
    # 仅 Layer 1（检索指标，不调 LLM，免费）
    uv run python -m eval.runner --mode retrieval

    # 完整评估（Layer 1 + Layer 2，调 Judge LLM，计费）
    uv run python -m eval.runner --mode full

    # 对比某两次运行
    uv run python -m eval.runner --compare <run_id_1> <run_id_2>

"""

import argparse
import json
import logging
import os
import sys
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from retrieval.store import VectorStore
from retrieval.retriever import Retriever
from retrieval.generator import Generator
from config import TOP_K, LLM_MODEL_ID, EVAL_LLM_MODEL_ID, EVAL_MAX_WORKERS, GENERATOR_TEMPERATURE

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


def _load_previous_run() -> dict[str, dict] | None:
    """加载最近一次运行的 per_query.json（排除当前这次），按 query_id 索引。"""
    timeline = _PROJECT_ROOT / "eval" / "results" / "timeline"
    if not timeline.exists():
        return None
    runs = sorted(
        [d for d in timeline.iterdir() if d.is_dir()],
        reverse=True,
    )
    for run_dir in runs:
        per_query_path = run_dir / "per_query.json"
        if per_query_path.exists():
            try:
                with open(per_query_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if data and any(
                "faithfulness" in v for v in data.values()
                if isinstance(v, dict)
            ):
                return data
    return None


def _evaluate_one(item, retriever, generator):
    """单条 query 的完整 eval。含 Generator 缓存 + 阶段打点。try/except 隔离。"""
    from eval.core.llm_as_judge import run_judge, JudgeResult
    from eval.core.judge_formatter import get_formatter, build_judge_context
    from eval.core.judge_cache import _generator_cache_key
    from eval.core.monitor_metrics import get_metrics
    from eval.core.live_panel import get_panel
    from infra.cache import get_cache as get_cache_backend
    from infra.config import REDIS_DEFAULT_TTL

    try:
        # Stage: retrieve
        t0 = time_module.time()
        chunks = retriever.retrieve(item.query, top_k=TOP_K)
        retrieve_ms = (time_module.time() - t0) * 1000

        # Generator cache
        context_str = build_judge_context(chunks)
        gen_cache_key = _generator_cache_key(item.query_id, context_str)
        cache_backend = get_cache_backend()
        gen_cached = cache_backend.get(gen_cache_key)

        metrics = get_metrics()

        # Stage: generate
        if gen_cached is not None:
            metrics.record_generator_cache_hit()
            answer = gen_cached
            generate_ms = 0.0
        else:
            metrics.record_generator_cache_miss()
            t1 = time_module.time()
            answer = generator.generate(item.query, chunks, temperature=GENERATOR_TEMPERATURE)
            generate_ms = (time_module.time() - t1) * 1000
            metrics.record_llm_call("generator")
            cache_backend.set(gen_cache_key, answer, ttl_seconds=REDIS_DEFAULT_TTL)

        # Judge（temperature=0 保证确定性）
        result = run_judge(item.query_id, item.query, chunks, answer,
                          item.reference_facts, formatter=get_formatter(),
                          temperature=0.0)
        result.retrieve_ms = retrieve_ms
        result.generate_ms = generate_ms

        return result
    except Exception as e:
        panel = get_panel()
        if panel:
            panel.push_alert(item.query_id, f"Generator {type(e).__name__}: {e}")
        jr = JudgeResult(query_id=item.query_id)
        jr.generator_error = f"evaluate_one failed: {e}"
        jr.verdict = "error"
        return jr


def run_retrieval_mode(benchmark_path: str):
    """仅对 Layer 1 retrieval 进行评估（v5: LivePanel 终端报告）。"""
    from eval.core.benchmark import load_benchmark
    from eval.core.retrieval_layer import run_retrieval_eval
    from eval.reporter import generate_report, build_run_info
    from eval.core.monitor_metrics import get_metrics, reset_metrics
    from eval.core.live_panel import LivePanel

    reset_metrics()
    metrics = get_metrics()

    print("加载索引...")
    store = VectorStore.vector_restore()
    if store is None:
        print("错误: 索引缓存不存在")
        sys.exit(1)
    retriever = Retriever(store)

    print(f"加载 benchmark: {benchmark_path}")
    result = load_benchmark(benchmark_path, valid_chunk_ids=store.chunk_ids)
    items = result.items
    total = len(items)
    print(f"  条目: {total}")

    # LivePanel（仅 render_final，无 daemon 线程——retrieval 无 LLM，跑得太快）
    previous_per_query = _load_previous_run()
    panel = LivePanel(metrics, previous_per_query)
    panel.set_total(total)
    panel.set_meta(benchmark_name=benchmark_path, eval_mode="retrieval")

    print("执行 Layer 1 检索评估...")
    output = run_retrieval_eval(retriever, items)
    metrics.layer1_results = output.results

    panel.query_count = total
    panel.render_final()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = str(_PROJECT_ROOT / "eval" / "results" / "timeline" / ts)
    run_info = build_run_info(benchmark_path)
    generate_report(output, results_dir, run_info)
    return results_dir


def run_full_mode(benchmark_path: str):
    """Layer 1 + Layer 2 完整评估（v5: LivePanel 终端监控）。"""
    from eval.core.benchmark import load_benchmark
    from eval.core.retrieval_layer import run_retrieval_eval
    from eval.reporter import generate_report, build_run_info
    from eval.core.llm_as_judge import _get_client
    from eval.core.monitor_metrics import get_metrics, reset_metrics
    from eval.core.live_panel import LivePanel, set_panel

    reset_metrics()
    metrics = get_metrics()

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
    total = len(items)
    print(f"  条目: {total}")

    # 加载上次运行（Delta 基线）
    previous_per_query = _load_previous_run()

    # 创建 LivePanel
    panel = LivePanel(metrics, previous_per_query)
    set_panel(panel)
    panel.set_total(total)
    panel.set_meta(
        benchmark_name=benchmark_path,
        generator_model=LLM_MODEL_ID or "unknown",
        judge_model=EVAL_LLM_MODEL_ID,
    )
    panel.start()

    print("执行 Layer 1 检索评估...")
    layer1 = run_retrieval_eval(retriever, items)
    metrics.layer1_results = layer1.results

    print("执行 Layer 2 Judge...")
    judge_results = []
    pool = _get_outer_pool()
    futures = {}
    for item in items:
        f = pool.submit(_evaluate_one, item, retriever, generator)
        futures[f] = item

    try:
        for future in as_completed(futures):
            jr = future.result()
            judge_results.append(jr)

            # 推入唯一真源
            metrics.layer2_results.append(jr)
            if jr.retrieve_ms is not None:
                metrics.record_stage("retrieve", jr.retrieve_ms)
            if jr.generate_ms is not None:
                metrics.record_stage("generate", jr.generate_ms)
            if jr.judge_faithfulness_ms is not None:
                metrics.record_stage("judge_faithfulness", jr.judge_faithfulness_ms)
            if jr.judge_quality_ms is not None:
                metrics.record_stage("judge_quality", jr.judge_quality_ms)

            # 端到端延迟 = retrieve + generate + max(faith, qual)，逐 query 计算后取百分位
            if jr.retrieve_ms is not None and jr.generate_ms is not None:
                e2e = jr.retrieve_ms + jr.generate_ms + max(
                    jr.judge_faithfulness_ms or 0,
                    jr.judge_quality_ms or 0,
                )
                metrics.record_stage("end_to_end", e2e)

            panel.query_done()
    finally:
        panel.stop()  # 幂等

    # 最终报告（终端）
    panel.render_final()

    # 写入文件报告
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = str(_PROJECT_ROOT / "eval" / "results" / "timeline" / ts)
    run_info = build_run_info(benchmark_path, run_mode="full")
    generate_report(layer1, results_dir, run_info,
                    judge_results=judge_results,
                    metrics_summary=metrics.summary_dict())

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

    metrics_list = ["recall_at_k", "precision_at_k", "hit_at_k", "mrr", "map_at_k", "ndcg_at_k"]
    print(f"\n{'Metric':<18} {'Run A':>10} {'Run B':>10} {'Delta':>10}")
    print("-" * 50)
    for m in metrics_list:
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
