"""Layer 2: LLM-as-Judge。两次 LLM 调用 + compute_verdict 纯函数。"""

import json
from dataclasses import dataclass, field
from openai import OpenAI

from eval.core.calculator.utils import _extract_json, _clamp_score
from eval.core.formatter import Formatter, build_judge_context
from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL, EVAL_LLM_MODEL_ID


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


def _call_llm(client: OpenAI, model: str, prompt: str) -> dict:
    """调用 LLM 并解析 JSON 输出（经 _extract_json 三层兜底）。

    Returns:
        解析成功的 JSON dict，解析失败则返回 {"error": ..., "raw": ...}
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    if text is None:
        return {"error": "LLM returned empty content", "raw": ""}
    try:
        return _extract_json(text)
    except (json.JSONDecodeError, ValueError) as e:
        return {"error": str(e), "raw": text[:500]}


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


def compute_verdict(scores: dict[str, float | None]) -> str:
    """纯函数：有值维度中 >= 0.75 的比例 → pass/partial/fail。

    >>> compute_verdict({"faithfulness": 0.75, "answer_relevancy": 0.75, "context_precision": 0.50, "context_recall": 0.75, "answer_correctness": 0.75})
    'pass'
    >>> compute_verdict({"faithfulness": 0.75, "answer_relevancy": 0.50, "context_precision": 0.75, "context_recall": 0.50, "answer_correctness": None})
    'partial'
    >>> compute_verdict({"faithfulness": 0.25, "answer_relevancy": 0.25, "context_precision": 0.25, "context_recall": None, "answer_correctness": 0.25})
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
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
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
        formatter = Formatter()

    context_str = build_judge_context(chunks)
    answer_str = answer or "（模型未生成回答）"

    result = JudgeResult(query_id=query_id)

    # 调用 1：faithfulness
    faith, claims, err1 = judge_faithfulness(
        client, model, formatter, query, context_str, answer_str
    )
    result.faithfulness = faith
    result.grounded_claims = claims
    if err1:
        result.parse_error = f"faithfulness: {err1}"

    # 调用 2：质量四维度
    ar, cp, cr, ac, err2 = judge_quality(
        client, model, formatter, query, context_str, answer_str, reference_facts
    )
    result.answer_relevancy = ar
    result.context_precision = cp
    result.context_recall = cr
    result.answer_correctness = ac
    if err2:
        msg = f"quality: {err2}"
        result.parse_error = f"{result.parse_error}; {msg}" if result.parse_error else msg

    # verdict
    scores = {
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
        "context_precision": result.context_precision,
        "context_recall": result.context_recall,
        "answer_correctness": result.answer_correctness,
    }
    result.verdict = compute_verdict(scores)

    return result
