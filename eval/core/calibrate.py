"""5-sample 校准工作流：验证 Judge prompt 对好/坏答案的区分度。

流程：
1. 从 benchmark 取 5 条，逐条跑 pipeline 产出"好答案"
2. 构建"坏答案"：类型 A（注入幻觉）+ 类型 B（错误 chunk 喂入）
3. 对好/坏答案分别跑 Judge，计算区分度
4. 不通过则迭代（最多 3 轮）
"""

import json
import copy
import sys
from datetime import datetime

from retrieval.store import VectorStore
from retrieval.retriever import Retriever
from retrieval.generator import Generator
from eval.core.benchmark import load_benchmark
from eval.core.llm_as_judge import run_judge, compute_verdict
from config import TOP_K, LLM_MODEL_ID, EVAL_LLM_MODEL_ID

CALIBRATION_SAMPLES = 5
MAX_ITERATIONS = 3


def _make_hallucinated_answer(good_answer: str, query: str) -> str:
    """类型 A：用 LLM 基于好答案生成一个注入了事实错误的版本。"""
    # 简单策略：在好答案中替换关键术语
    swaps = [
        ("MySQL", "PostgreSQL"),
        ("InnoDB", "MyISAM"),
        ("B+Tree", "Hash"),
        ("REPEATABLE READ", "SERIALIZABLE"),
        ("MVCC", "锁"),
        ("索引", "全表扫描"),
    ]
    bad = good_answer
    for a, b in swaps:
        if a in bad:
            bad = bad.replace(a, b, 1)
            break
    if bad == good_answer:
        bad = f"[错误版本] {good_answer}"
    return bad


def _collect_wrong_chunks(store: VectorStore, query: str, retriever: Retriever) -> list:
    """类型 B：检索到正确 chunk 后，故意替换为不相关的 chunk。"""
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


def run_calibration() -> dict:
    """执行校准流程，返回校准报告。"""
    store = VectorStore.vector_restore()
    if store is None:
        print("错误: 索引缓存不存在")
        sys.exit(1)

    retriever = Retriever(store)
    result = load_benchmark("benchmark_private.json", valid_chunk_ids=store.chunk_ids)
    samples = result.items[:CALIBRATION_SAMPLES]

    print(f"校准样本: {len(samples)} 条\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"{'='*60}")
        print(f"  第 {iteration}/{MAX_ITERATIONS} 轮校准")
        print(f"{'='*60}\n")

        records = []
        for item in samples:
            print(f"  [{item.query_id}] {item.query[:50]}...")

            # 好答案：正常 pipeline
            chunks = retriever.retrieve(item.query, top_k=TOP_K)
            answer = Generator(model=LLM_MODEL_ID).generate(item.query, chunks)
            print(f"    good answer: {answer[:80]}...")

            # 坏答案 A：注入幻觉
            bad_a = _make_hallucinated_answer(answer, item.query)

            # 坏答案 B：错误 context
            wrong_chunks = _collect_wrong_chunks(store, item.query, retriever)
            bad_b = Generator(model=LLM_MODEL_ID).generate(item.query, wrong_chunks)
            print(f"    bad (type A): {bad_a[:80]}...")
            print(f"    bad (type B): {bad_b[:80]}...")

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
                print(f"  准备下一轮迭代...\n")
            else:
                print(f"\n  ✗ 三轮校准未通过，Judge prompt 需人工调整。")
                return {"passed": False, "iteration": iteration, "records": records}


if __name__ == "__main__":
    report = run_calibration()
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = f"eval/results/calibrate_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n校准报告已保存: {path}")
