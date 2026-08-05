"""Judge prompt 校准工作流：5-sample 验证好/坏答案区分度。

核心特性：
    - 从 benchmark 取 5 条，逐条跑 pipeline 产出"好答案"（正常上下文）
    - 构建"坏答案"：SWAP_TIER_1（同级术语替换）+ SWAP_TIER_2（结构破坏），词表无匹配时兜底到坏答案 B（错误 chunk 生成的整段幻觉答案）
    - 对好/坏答案分别跑 Judge，计算 faithfulness 区分度（diff = good_avg - bad_avg）
    - 最多 3 轮迭代，首轮即通过时退出；Tier 2 自动启用（三轮后 diff < 0.50）
    - 模块级 OpenAI client 单例，循环外复用

用法示例::

    uv run python -m eval.core.llm_as_judge.judge_calibrate

公共接口：
    - _make_hallucination: 注入幻觉（术语替换 → 结构破坏）
    - run_calibrate: 完整校准工作流
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import TYPE_CHECKING

from config import TOP_K, LLM_MODEL_ID, _PROJECT_ROOT
from eval.core.benchmark import load_benchmark
from eval.core.llm_as_judge.judge import run_judge, _get_client
from indexing.index_store import IndexStore
from retrieval.retriever import Retriever
from retrieval.generator import Generator

if TYPE_CHECKING:
    from indexing.chunk import Chunk

CALIBRATION_SAMPLE_COUNT = 5
MAX_ITERATIONS = 3

SWAP_TIER_1 = [
    ("MySQL", "PostgreSQL"), ("InnoDB", "MyISAM"), ("B+Tree", "Hash"),
    ("REPEATABLE READ", "SERIALIZABLE"), ("undo log", "redo log"),
    ("B-Tree", "LSM-Tree"), ("聚簇索引", "二级索引"), ("ACID", "BASE"),
    ("行锁", "意向锁"), ("读已提交", "可重复读"),
    # Agent/LLM 域（benchmark 前 5 条全为 Agent 查询）
    ("Agentic System", "Workflow"), ("ReAct", "Plan-and-Execute"),
    ("思维链", "直接推理"), ("工具调用", "文本拼接"),
    ("长期记忆", "短期记忆"), ("任务分解", "任务合并"),
]
SWAP_TIER_2 = [
    ("索引", "全表扫描"), ("MVCC", "锁"), ("连接池", "单连接"),
    ("异步", "同步阻塞"), ("缓存", "每次重新计算"),
    ("反思", "复述"), ("规划", "立即执行"),
]


def _make_hallucination(good_answer: str, use_tier2: bool = False) -> tuple[str, bool]:
    """类型 A：注入幻觉 —— 替换所有匹配的同级术语。

    Args:
        good_answer: 正常 pipeline 产出的好答案。
        use_tier2: 是否启用 Tier 2 结构破坏级替换词表。

    Returns:
        tuple[str, bool]：(替换后的答案, 词表是否有命中)。
        无命中时不兜底，由调用方 fallback 到坏答案 B（错误 chunk 生成的整段幻觉答案）——
        原说明兜底是弱负样本（good 的 claims 仍与 context 一致，faithfulness 按 rubric 应给高分）。
    """
    swaps = SWAP_TIER_1 if not use_tier2 else SWAP_TIER_1 + SWAP_TIER_2
    bad_answer = good_answer
    replaced = False
    for original, replacement in swaps:
        if original in bad_answer:
            bad_answer = bad_answer.replace(original, replacement)  # 替换所有出现，不限制 count
            replaced = True
    return bad_answer, replaced


def _inject_wrong_chunks(store: IndexStore, query: str, retriever: Retriever) -> list[Chunk]:
    """类型 B：错误 chunk 注入——检索到正确 chunk 后，故意替换为不相关的 chunk。

    Args:
        store: 索引存储门面，用于枚举库中全部 chunk。
        query: 检索问题。
        retriever: 检索器，先取正确 chunk。

    Returns:
        list[Chunk]：与正确 chunk 不同的随机 chunk（不足 TOP_K 时退回正确列表）。
    """
    correct_chunks = retriever.retrieve(query, top_k=TOP_K)
    correct_ids = {chunk.chunk_id for chunk in correct_chunks}

    # 从库中挑几个与正确 chunk 不同的随机 chunk
    wrong_chunks = []
    for chunk in store.chunks:
        if chunk.chunk_id not in correct_ids:
            wrong_chunks.append(chunk)
        if len(wrong_chunks) >= TOP_K:
            break
    return wrong_chunks if wrong_chunks else correct_chunks


def run_calibrate() -> dict:
    """执行校准流程，返回校准报告。

    Returns:
        dict：含 passed / iteration / records 的校准结果；三轮未通过时 passed=False。
    """
    store = IndexStore.vector_restore()
    if store is None:
        print("错误: 索引缓存不存在")
        sys.exit(1)

    retriever = Retriever(store)
    generator = Generator(model=LLM_MODEL_ID)
    _get_client()  # 主线程预初始化

    result = load_benchmark("benchmark/private_v5.json", valid_chunk_ids=store.chunk_ids)
    samples = result.valid_items[:CALIBRATION_SAMPLE_COUNT]

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

            # 坏答案 2: 错误 chunk 喂入（先生成，供坏答案 A 兜底引用）
            wrong_context = _inject_wrong_chunks(store, item.query, retriever)
            wrong_context_answer = generator.generate(item.query, wrong_context)

            # 坏答案 1: 幻觉注入（词表无命中时兜底到坏答案 B）
            hallucinated_answer, replaced = _make_hallucination(answer, use_tier2=use_tier2)
            if not replaced:
                hallucinated_answer = wrong_context_answer
            print(f"    bad (幻觉注入):\n{hallucinated_answer}\n")
            print(f"    bad (错误chunk喂入):\n{wrong_context_answer}\n")

            # Judge（校准路径 skip_cache=True 强制绕过缓存：同 query+context 评不同答案，缓存键不含 answer 会串台）
            good_result = run_judge(item.query_id, item.query, chunks, answer, item.reference_facts, skip_cache=True)
            hallucinated_result = run_judge(item.query_id, item.query, chunks, hallucinated_answer, item.reference_facts, skip_cache=True)
            # 坏答案 B 由错误 chunk 生成、用正确 chunks 评 → claims 无支撑 → faithfulness 真正降低
            wrong_context_result = run_judge(item.query_id, item.query, chunks, wrong_context_answer, item.reference_facts, skip_cache=True)

            records.append({
                "query_id": item.query_id,
                "good": {"faithfulness": good_result.faithfulness, "answer_correctness": good_result.answer_correctness,
                         "verdict": good_result.verdict, "parse_error": good_result.parse_error,
                         "judge_error": good_result.judge_error},
                "bad_a": {"faithfulness": hallucinated_result.faithfulness, "answer_correctness": hallucinated_result.answer_correctness,
                          "verdict": hallucinated_result.verdict, "parse_error": hallucinated_result.parse_error,
                          "judge_error": hallucinated_result.judge_error},
                "bad_b": {"faithfulness": wrong_context_result.faithfulness, "answer_correctness": wrong_context_result.answer_correctness,
                          "verdict": wrong_context_result.verdict, "parse_error": wrong_context_result.parse_error,
                          "judge_error": wrong_context_result.judge_error},
            })

        # 汇总
        good_faithfulness_scores = [record["good"]["faithfulness"] for record in records if record["good"]["faithfulness"] is not None]
        bad_faithfulness_scores = [record["bad_a"]["faithfulness"] for record in records if record["bad_a"]["faithfulness"] is not None]
        parse_errors = sum(1 for record in records if record["good"]["parse_error"] or record["bad_a"]["parse_error"] or record["bad_b"]["parse_error"])
        total_calls = len(records) * 3 * 2  # 3 answers × 2 calls

        # verdict 判别诊断（不参与通过判定）：区分"分数低"与"调用错误"
        # execute_verdict 在 scores 全为 None 时返回 fail —— judge 超时/报错也会是 fail，须分开
        good_verdict_fail = []   # good 被真评出 fail（分数低）
        good_judge_error = []    # good 调用出错，verdict=fail 不代表分数低
        for record in records:
            good = record["good"]
            if good["verdict"] == "fail":
                if good["judge_error"] or good["parse_error"]:
                    good_judge_error.append(record["query_id"])
                else:
                    good_verdict_fail.append(record["query_id"])

        good_avg = sum(good_faithfulness_scores) / len(good_faithfulness_scores) if good_faithfulness_scores else 0
        bad_avg = sum(bad_faithfulness_scores) / len(bad_faithfulness_scores) if bad_faithfulness_scores else 0
        diff = good_avg - bad_avg

        print(f"\n  结果:")
        print(f"    faithfulness 好答案均值: {good_avg:.3f}")
        print(f"    faithfulness 坏答案均值: {bad_avg:.3f}")
        print(f"    区分度 (差值): {diff:.3f}")
        print(f"    JSON 解析失败: {parse_errors}/{total_calls}")
        print(f"    好答案 verdict=fail（分数低）: {good_verdict_fail if good_verdict_fail else '无'}")
        print(f"    好答案 verdict=fail（调用错误）: {good_judge_error if good_judge_error else '无'}")

        passed = (
            good_avg >= 0.75
            and bad_avg <= 0.50
            and diff >= 0.50
            and (parse_errors / total_calls) <= 0.05
        )

        if passed:
            print(f"\n  ✓ 校准通过（第 {iteration} 轮）")
            return {"passed": True, "iteration": iteration, "records": records}
        else:
            print(f"\n  ✗ 校准未通过，原因:")
            if good_avg < 0.75:
                print(f"    faithfulness 好答案均值 {good_avg:.3f} < 0.75")
            if bad_avg > 0.50:
                print(f"    faithfulness 坏答案均值 {bad_avg:.3f} > 0.50")
            if diff < 0.50:
                print(f"    区分度 {diff:.3f} < 0.50")
            if good_verdict_fail:
                print(f"    提示（不阻断）: 好答案 verdict=fail（分数低）: {good_verdict_fail}"
                      f" —— 可能 pipeline 产出坏答案或 Judge 过严，需人工判断")

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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _PROJECT_ROOT / "eval" / "results" / "calibrate"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{timestamp}.json"
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, ensure_ascii=False, indent=2, default=str)
    print(f"\n校准报告已保存: {path}")
