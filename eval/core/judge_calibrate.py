"""Judge prompt 校准工作流：5-sample 验证好/坏答案区分度。

核心特性：
    - 从 benchmark 取 5 条，逐条跑 pipeline 产出"好答案"（正常上下文）
    - 构建"坏答案"：SWAP_TIER_1（同级术语替换）+ SWAP_TIER_2（结构破坏），无匹配时追加错误陈述
    - 对好/坏答案分别跑 Judge，计算 faithfulness 区分度（diff = good_mean - bad_mean）
    - 最多 3 轮迭代，首轮即通过时退出；Tier 2 自动启用（三轮后 diff < 0.50）
    - 模块级 OpenAI client 单例，循环外复用

用法示例::

    uv run python -m eval.core.judge_calibrate

公共接口：
    - _make_hallucination: 注入幻觉（术语替换 → 结构破坏）
    - run_calibration: 完整校准工作流
"""

import json
import logging
import copy
import sys
from datetime import datetime
from pathlib import Path

from retrieval.store import VectorStore
from retrieval.retriever import Retriever
from retrieval.generator import Generator
from eval.core.benchmark import load_benchmark
from eval.core.llm_as_judge import run_judge, execute_verdict, _get_client
from config import TOP_K, LLM_MODEL_ID, EVAL_LLM_MODEL_ID, _PROJECT_ROOT

CALIBRATION_SAMPLES = 5
MAX_ITERATIONS = 3

SWAP_TIER_1 = [
    ("MySQL", "PostgreSQL"), ("InnoDB", "MyISAM"), ("B+Tree", "Hash"),
    ("REPEATABLE READ", "SERIALIZABLE"), ("undo log", "redo log"),
    ("B-Tree", "LSM-Tree"), ("聚簇索引", "二级索引"), ("ACID", "BASE"),
    ("行锁", "意向锁"), ("读已提交", "可重复读"),
]
SWAP_TIER_2 = [
    ("索引", "全表扫描"), ("MVCC", "锁"), ("连接池", "单连接"),
    ("异步", "同步阻塞"), ("缓存", "每次重新计算"),
]


def _make_hallucination(good_answer: str, query: str, use_tier2: bool = False) -> str:
    """类型 A：注入幻觉 —— 替换所有匹配的同级术语。"""
    swaps = SWAP_TIER_1 if not use_tier2 else SWAP_TIER_1 + SWAP_TIER_2
    bad = good_answer
    replaced = False
    for a, b in swaps:
        if a in bad:
            bad = bad.replace(a, b)  # 替换所有出现，不限制 count
            replaced = True
    if not replaced:
        bad = good_answer + "\n\n（注：以上内容中的技术细节存在多处事实性错误，请以官方文档为准。）"
    return bad


def _inject_wrong_chunks(store: VectorStore, query: str, retriever: Retriever) -> list:
    """类型 B: 错误chunk注入
    检索到正确 chunk 后，故意替换为不相关的 chunk。"""
    correct = retriever.retrieve(query, top_k=TOP_K)
    correct_ids = {c.chunk_id for c in correct}

    # 从库中挑几个与正确 chunk 不同的随机 chunk
    wrong = []
    for c in store.chunks:
        if c.chunk_id not in correct_ids:
            wrong.append(c)
        if len(wrong) >= TOP_K:
            break
    return wrong if wrong else correct


def run_calibrate() -> dict:
    """执行校准流程，返回校准报告。"""
    store = VectorStore.vector_restore()
    if store is None:
        print("错误: 索引缓存不存在")
        sys.exit(1)

    retriever = Retriever(store)
    generator = Generator(model=LLM_MODEL_ID)
    _get_client()  # 主线程预初始化

    result = load_benchmark("benchmark_private.json", valid_chunk_ids=store.chunk_ids)
    samples = result.items[:CALIBRATION_SAMPLES]

    print(f"校准样本: {len(samples)} 条")

    # 模型预热：首次检索触发 embedding 模型加载，避免 Loading weights 日志打乱输出
    print("模型预热中...")
    retriever.retrieve(samples[0].query, top_k=TOP_K)
    print()

    use_tier2 = False
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"{'='*60}")
        print(f"  第 {iteration}/{MAX_ITERATIONS} 轮校准")
        print(f"{'='*60}\n")

        records = []
        for item in samples:
            print(f"  [{item.query_id}] {item.query[:50]}...")

            # 好答案：正常 pipeline
            chunks = retriever.retrieve(item.query, top_k=TOP_K)
            answer = generator.generate(item.query, chunks)
            print(f"    expected answer:\n{answer}\n")

            # 坏答案 1: 幻觉注入
            bad_a = _make_hallucination(answer, item.query, use_tier2=use_tier2)

            # 坏答案 2: 错误 chunk 喂入
            wrong_chunks = _inject_wrong_chunks(store, item.query, retriever)
            bad_b = generator.generate(item.query, wrong_chunks)
            print(f"    bad (幻觉注入):\n{bad_a}\n")
            print(f"    bad (错误chunk喂入):\n{bad_b}\n")

            # Judge
            jr_good = run_judge(item.query_id, item.query, chunks, answer, item.reference_facts)
            jr_bad_a = run_judge(item.query_id, item.query, chunks, bad_a, item.reference_facts)
            jr_bad_b = run_judge(item.query_id, item.query, wrong_chunks, bad_b, item.reference_facts)

            records.append({
                "query_id": item.query_id,
                "good": {"faithfulness": jr_good.faithfulness, "answer_correctness": jr_good.answer_correctness,
                         "verdict": jr_good.verdict, "parse_error": jr_good.parse_error},
                "bad_a": {"faithfulness": jr_bad_a.faithfulness, "answer_correctness": jr_bad_a.answer_correctness,
                          "verdict": jr_bad_a.verdict, "parse_error": jr_bad_a.parse_error},
                "bad_b": {"faithfulness": jr_bad_b.faithfulness, "answer_correctness": jr_bad_b.answer_correctness,
                          "verdict": jr_bad_b.verdict, "parse_error": jr_bad_b.parse_error},
            })

        # 汇总
        good_faith = [r["good"]["faithfulness"] for r in records if r["good"]["faithfulness"] is not None]
        bad_faith = [r["bad_a"]["faithfulness"] for r in records if r["bad_a"]["faithfulness"] is not None]
        parse_errors = sum(1 for r in records if r["good"]["parse_error"] or r["bad_a"]["parse_error"] or r["bad_b"]["parse_error"])
        total_calls = len(records) * 3 * 2  # 3 answers × 2 calls

        gfa = sum(good_faith) / len(good_faith) if good_faith else 0
        bfa = sum(bad_faith) / len(bad_faith) if bad_faith else 0
        diff = gfa - bfa

        print(f"\n  结果:")
        print(f"    faithfulness 好答案均值: {gfa:.3f}")
        print(f"    faithfulness 坏答案均值: {bfa:.3f}")
        print(f"    区分度 (差值): {diff:.3f}")
        print(f"    JSON 解析失败: {parse_errors}/{total_calls}")

        passed = (
            gfa >= 0.75
            and bfa <= 0.50
            and diff >= 0.50
            and (parse_errors / total_calls) <= 0.05
        )

        if passed:
            print(f"\n  ✓ 校准通过（第 {iteration} 轮）")
            return {"passed": True, "iteration": iteration, "records": records}
        else:
            print(f"\n  ✗ 校准未通过，原因:")
            if gfa < 0.75:
                print(f"    faithfulness 好答案均值 {gfa:.3f} < 0.75")
            if bfa > 0.50:
                print(f"    faithfulness 坏答案均值 {bfa:.3f} > 0.50")
            if diff < 0.50:
                print(f"    区分度 {diff:.3f} < 0.50")

            if iteration < MAX_ITERATIONS:
                if diff < 0.50:
                    use_tier2 = True
                    print(f"    → 启用 Tier 2 结构破坏级替换")
                print(f"  准备下一轮迭代...\n")
            else:
                print(f"\n  ✗ 三轮校准未通过，Judge prompt 需人工调整。")
                return {"passed": False, "iteration": iteration, "records": records}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for noisy in ("httpx", "openai", "jieba", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    report = run_calibrate()
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _PROJECT_ROOT / "eval" / "results" / "calibrate"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n校准报告已保存: {path}")
