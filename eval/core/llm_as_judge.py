"""Layer 2: LLM-as-Judge。两次 LLM 调用 + compute_verdict 纯函数。"""

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

import httpx
import openai
from openai import OpenAI

from eval.core.calculator.utils import _extract_json, _clamp_score
from eval.core.formatter import Formatter, build_judge_context, get_formatter
from eval.core.judge_cache import _judge_cache_key, get_cached_result, set_cached_result
from config import (LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL, EVAL_LLM_MODEL_ID,
                    JUDGE_MAX_RETRIES, JUDGE_BASE_DELAY, JUDGE_DEADLINE, EVAL_MAX_WORKERS)

class JudgeFailedError(Exception):
    """Judge LLM 调用重试耗尽后抛出。"""


RETRYABLE_ERRORS = (
    openai.RateLimitError,       # 429
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@dataclass
class JudgeResult:
    """单条 query 的 Layer 2 评估结果。"""
    query_id: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    grounded_claims: list[dict] = field(default_factory=list)
    verdict: str = ""
    parse_error: str | None = None
    judge_error: str | None = None
    generator_error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "JudgeResult":
        return cls(**json.loads(json_str))


def _call_llm(client: OpenAI, model: str, prompt: str) -> dict:
    """调用 LLM 并解析 JSON 输出（经 _extract_json 三层兜底）。

    Returns:
        解析成功的 JSON dict，解析失败则返回 {"error": ..., "raw": ...}
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user", "content": prompt
            }
        ],
    )
    try:
        from eval.core.monitor_metrics import get_metrics
        get_metrics().record_llm_call()
        if response.usage:
            get_metrics().record_tokens(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
    except ImportError:
        pass
    text = response.choices[0].message.content
    if text is None:
        return {"error": "LLM returned empty content", "raw": ""}
    try:
        return _extract_json(text)
    except (json.JSONDecodeError, ValueError) as e:
        return {"error": str(e), "raw": text[:500]}


def _judge_with_retry(
    judge_fn,
    judge_args: tuple,
    max_retries: int = 3,
    base_delay: float = 1.0,
    deadline: float = 120.0,
):
    """对 judge_fn 做指数退避重试 + deadline 超时。

    pool 在循环外创建，finally shutdown(wait=False) 避免阻塞等孤儿线程。
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge-retry")
    try:
        last_error = None
        for attempt in range(max_retries):
            future = pool.submit(judge_fn, *judge_args)
            try:
                return future.result(timeout=deadline)
            except TimeoutError:
                future.cancel()
                last_error = TimeoutError(f"Judge call timed out after {deadline}s")
            except RETRYABLE_ERRORS as e:
                if isinstance(e, openai.RateLimitError):
                    logging.warning(
                        "DeepSeek 限流 (429)，第 %d/%d 次尝试",
                        attempt + 1, max_retries,
                    )
                last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) * random.uniform(0.5, 1.5)
                time.sleep(delay)
        raise JudgeFailedError(
            f"Judge call failed after {max_retries} retries: {last_error}"
        )
    finally:
        pool.shutdown(wait=False)


def judge_faithfulness(
    client: OpenAI, model: str, formatter: Formatter,
    query: str, context_str: str, answer: str,
) -> tuple[float | None, list[dict], str | None]:
    """调用 1：评估 faithfulness，返回 (分数, grounded_claims, 解析错误描述)。"""
    prompt = formatter.build_faithfulness_prompt(query, context_str, answer)
    result = _call_llm(client, model, prompt)

    if "error" in result:
        return None, [], result.get("error")

    score = result.get("faithfulness")
    if score is not None:
        try:
            score = _clamp_score(float(score))
        except (ValueError, TypeError):
            score = None
    return score, result.get("grounded_claims", []), None


def judge_quality(
    client: OpenAI, model: str, formatter: Formatter,
    query: str, context_str: str, answer: str, reference_facts: str,
) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    """调用 2：评估四个质量维度。

    Returns:
        (answer_relevancy, context_precision, context_recall, answer_correctness, 解析错误描述)
    """
    prompt = formatter.build_quality_prompt(query, context_str, answer, reference_facts)
    result = _call_llm(client, model, prompt)

    if "error" in result:
        return None, None, None, None, result.get("error")

    def _get(key: str) -> float | None:
        v = result.get(key)
        if v is None:
            return None
        try:
            return _clamp_score(float(v))
        except (ValueError, TypeError):
            return None

    return (
        _get("answer_relevancy"),
        _get("context_precision"),
        _get("context_recall"),
        _get("answer_correctness"),
        None,
    )


def execute_verdict(scores: dict[str, float | None]) -> str:
    """纯函数：有值维度中 >= 0.75 的比例 → pass/partial/fail。
    >>> execute_verdict({"faithfulness": 0.75, "answer_relevancy": 0.75, "context_precision": 0.50, "context_recall": 0.75, "answer_correctness": 0.75})
    'pass'
    >>> execute_verdict({"faithfulness": 0.75, "answer_relevancy": 0.50, "context_precision": 0.75, "context_recall": 0.50, "answer_correctness": None})
    'partial'
    >>> execute_verdict({"faithfulness": 0.25, "answer_relevancy": 0.25, "context_precision": 0.25, "context_recall": None, "answer_correctness": 0.25})
    'fail'
    """
    valid = [v for v in scores.values() if v is not None]
    if not valid:
        return "fail"
    ratio = sum(1 for v in valid if v >= 0.75) / len(valid)
    if ratio >= 0.75:
        return "pass"
    elif ratio >= 0.50:
        return "partial"
    else:
        return "fail"


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """模块级 OpenAI client 单例，避免每次 run_judge 重复创建连接池。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_keepalive_connections=max(20, EVAL_MAX_WORKERS * 4),
                    max_connections=max(24, EVAL_MAX_WORKERS * 5),
                ),
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=JUDGE_DEADLINE + 10,
                    write=30.0,
                    pool=5.0,
                ),
            ),
        )
    return _client


def run_judge(
    query_id: str, query: str, chunks, answer: str, reference_facts: str,
    client: OpenAI | None = None,
    model: str | None = None,
    formatter: Formatter | None = None,
) -> JudgeResult:
    """执行完整的双调用 Judge 流程，返回 JudgeResult。"""
    if client is None:
        client = _get_client()
    if model is None:
        model = EVAL_LLM_MODEL_ID
    if formatter is None:
        formatter = get_formatter()

    context_str = build_judge_context(chunks)
    answer_str = answer or "（模型未生成回答）"

    # 查缓存
    prompt_version = formatter.prompt_version
    cache_key = _judge_cache_key(query_id, context_str, answer_str,
                                 prompt_version, model)
    cached = get_cached_result(cache_key)
    if cached is not None:
        return cached

    result = JudgeResult(query_id=query_id)

    # 内层 2-worker 池：faithfulness + quality 并行
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="judge") as pool:
        faith_future = pool.submit(
            _judge_with_retry, judge_faithfulness,
            (client, model, formatter, query, context_str, answer_str),
            JUDGE_MAX_RETRIES, JUDGE_BASE_DELAY, JUDGE_DEADLINE,
        )
        quality_future = pool.submit(
            _judge_with_retry, judge_quality,
            (client, model, formatter, query, context_str, answer_str, reference_facts),
            JUDGE_MAX_RETRIES, JUDGE_BASE_DELAY, JUDGE_DEADLINE,
        )

        try:
            faith, claims, err1 = faith_future.result()
        except JudgeFailedError as e:
            faith, claims, err1 = None, [], str(e)

        try:
            ar, cp, cr, ac, err2 = quality_future.result()
        except JudgeFailedError as e:
            ar, cp, cr, ac, err2 = None, None, None, None, str(e)

    result.faithfulness = faith
    result.grounded_claims = claims
    result.answer_relevancy = ar
    result.context_precision = cp
    result.context_recall = cr
    result.answer_correctness = ac

    # 区分 parse_error 和 judge_error：JudgeFailedError → judge_error；其余 → parse_error
    judge_errors = []
    for label, err_val in [("faithfulness", err1), ("quality", err2)]:
        if not err_val:
            continue
        if "Judge call failed" in err_val:
            judge_errors.append(f"{label}: {err_val}")
        else:
            if result.parse_error:
                result.parse_error += f"; {label}: {err_val}"
            else:
                result.parse_error = f"{label}: {err_val}"
    if judge_errors:
        result.judge_error = "; ".join(judge_errors)

    # verdict
    scores = {
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
        "context_precision": result.context_precision,
        "context_recall": result.context_recall,
        "answer_correctness": result.answer_correctness,
    }
    result.verdict = execute_verdict(scores)

    # 写缓存
    set_cached_result(cache_key, result)

    return result
