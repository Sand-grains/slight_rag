"""
三层评估脚本
  Layer1 — 单独测 Retriever: Precision@K, Recall@K, MRR, NDCG
  Layer2 — 单独测 Generator: 喂标准答案作上下文, 测 Faithfulness + Answer Relevancy
  Layer3 — 端到端: Retriever + Generator 真实走一遍, 全指标 + 诊断矩阵
"""
import json
import numpy
import time
from pathlib import Path
from typing import List

# ---- 项目内部模块 ----
from indexing.loader import load          # 文件 → Chunk 列表
from indexing.chunker import chunk        # Chunk 列表 → Chunk 列表 (切分后)
from retrieval.embedding import embed     # 文本 → 向量 (batch)
from retrieval.store import VectorStore   # 向量存储 (内存, numpy 矩阵)
from retrieval.retriever import Retriever # 检索器 (余弦相似度 + Top-K)
from retrieval.generator import Generator # 生成器 (Prompt 模板 + LLM 调用)
from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL, EVAL_LLM_MODEL_ID, TOP_K, RELEVANCE_THRESHOLD

# ---- 评估框架 ----
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper           # Ragas 的 LLM 封装 (已弃用但兼容当前版本)
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI              # 用 OpenAI 兼容协议访问 DeepSeek
from langchain_community.embeddings import HuggingFaceEmbeddings  # 本地 Embedding 模型

# ==================== 基础设施 ====================

def build_index(data_dir: str = "data") -> tuple[VectorStore, Retriever]:
    """离线索引管线: 优先从缓存恢复, 缓存失效时重新编码并持久化"""
    store = VectorStore.vector_restore()
    if store is not None:
        return store, Retriever(store)
    store = VectorStore()

    for file_path in Path(data_dir).rglob("*"):
        if file_path.is_file() and file_path.suffix in (".txt", ".md"):
            docs = load(str(file_path))
            chunks = chunk(docs)
            vectors = embed([c.content for c in chunks])
            store.add(chunks, vectors)
    store.vector_persistence()
    return store, Retriever(store)


def _safe_score(value):
    """从 Ragas 评估结果中安全提取标量分数

    Ragas 部分 job 失败时某些指标会返回 per-sample list 而非聚合标量,
    取非 None 值的均值兜底. 全部为 None 或空列表时返回 0.0.
    """
    if isinstance(value, list):
        return float(numpy.mean([v for v in value if v is not None])) if value else 0.0
    return float(value)


def _judge_llm():
    """裁判 LLM — 走低成本模型做评估"""
    # bypass_n=True: DeepSeek API 不支持 n>1 (单次请求返回多个候选).
    # Ragas 默认对 ChatOpenAI 类型的 LLM 传 n>1, 会导致 BadRequestError.
    # 开启后 Ragas 改为发多个独立请求 (每个隐式 n=1), 结果等价.
    return LangchainLLMWrapper(ChatOpenAI(
        model=EVAL_LLM_MODEL_ID, api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
    ), bypass_n=True)


def _judge_embeddings():
    """裁判 Embeddings — 本地轻量模型, 避免调 API 计费"""
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="D:/Model"
    ))


# ==================== Layer1: Retriever ====================

def eval_retriever(retriever: Retriever, questions: list[dict], top_k: int = TOP_K,
                   threshold: float = RELEVANCE_THRESHOLD):
    """Layer1: 单独测 Retriever
    返回指标为Precision@K, Recall@K, MRR, NDCG

    threshold: 余弦相似度阈值, >= threshold 则认为该 chunk 与 ground_truth 相关.
    走矩阵乘一次算出所有 (chunk, question) 对的相似度, 避免逐条调用 embed.
    """
    store = retriever.store
    all_vectors = store._vectors                                  # (N_chunks, 384), 预存的 chunk 向量
    all_chunks = store._chunks
    num_questions = len(questions)

    # ---- 预计算: 一次性 batch + 矩阵乘, O(N * Q * D) 但常数极小 ----
    gt_texts = [q["ground_truth"] for q in questions]
    gt_vectors = numpy.array(embed(gt_texts))                     # (Q, 384), ground_truth 的向量
    all_scores = numpy.dot(all_vectors, gt_vectors.T)             # (N, Q), 每个 chunk 与每道题的相关度
    all_relevant = (all_scores >= threshold).astype(int)           # (N, Q), 0/1 二值化

    # chunk_id → 矩阵行号映射, 后续 O(1) 定位检索结果的向量行
    id_to_index = {c.chunk_id: j for j, c in enumerate(all_chunks)}

    precisions, recalls, mrrs, ndcgs = [], [], [], []
    total_rel_counts = all_relevant.sum(axis=0)                   # (Q,), 每题全局有多少相关 chunk

    for i, q in enumerate(questions):
        retrieved = retriever.retrieve(q["question"], top_k=top_k)
        k = len(retrieved)
        retrieved_indices = [id_to_index[c.chunk_id] for c in retrieved]

        # 检索结果中相关的 chunk: 从预计算矩阵取对应行, 向量化判断
        rel_retrieved = all_relevant[retrieved_indices, i]        # (k,)
        rel_count = int(rel_retrieved.sum())
        total_rel = max(int(total_rel_counts[i]), 1)              # 避免除以 0

        # Precision@K = 检索结果中相关的 / K
        precisions.append(rel_count / k)
        # Recall@K = 检索结果中相关的 / 全库所有相关的
        recalls.append(rel_count / total_rel)

        # MRR: 第一个相关结果排名的倒数, 排名越靠前越接近 1
        mrr = 0.0
        for rank, r in enumerate(rel_retrieved, start=1):
            if r == 1:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

        # NDCG: 位置加权折损, 靠前的结果贡献更大
        ideal_n = min(total_rel, k)
        dcg = sum((2 ** int(r) - 1) / numpy.log2(j + 2)
                  for j, r in enumerate(rel_retrieved))
        idcg = sum(1.0 / numpy.log2(j + 2) for j in range(ideal_n))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    print(f"\n  ---- 汇总 ({num_questions} 条) ----")
    print(f"  {'Precision@' + str(top_k):>16}: {numpy.mean(precisions):.4f}")
    print(f"  {'Recall@' + str(top_k):>16}: {numpy.mean(recalls):.4f}")
    print(f"  {'MRR':>16}: {numpy.mean(mrrs):.4f}")
    print(f"  {'NDCG':>16}: {numpy.mean(ndcgs):.4f}")
    return {"precision": float(numpy.mean(precisions)),
            "recall": float(numpy.mean(recalls)),
            "mrr": float(numpy.mean(mrrs)),
            "ndcg": float(numpy.mean(ndcgs))}


# ==================== Layer2: Generator ====================

def eval_generator(questions: list[dict]):
    """Layer2: 单独测 Generator — 直接用 ground_truth 作上下文

    这样做是为了排除检索噪声: 如果模型拿着标准答案都答不好,
    那问题在 Generator (Prompt 模板 / LLM 选型), 不在 Retriever.
    """
    generator = Generator(model=EVAL_LLM_MODEL_ID)
    judge_llm = _judge_llm()
    judge_emb = _judge_embeddings()

    samples = []
    for i, q in enumerate(questions):
        # 把 ground_truth 包装成 Chunk, 假装是检索到的"完美 chunk"
        from indexing.chunk import Chunk, DocMetadata
        perfect_chunk = Chunk(content=q["ground_truth"], doc_meta=DocMetadata(title="ground_truth"))
        answer = generator.generate(q["question"], [perfect_chunk])
        samples.append({
            "user_input": q["question"],
            "response": answer,                              # 模型实际输出
            "retrieved_contexts": [q["ground_truth"]],       # 喂入的上下文 (标准答案)
            "reference": q["ground_truth"]                   # 参考答案 (与上下文相同)
        })
        print(f"  生成进度: [{i + 1}/{len(questions)}]", end="\r")

    print()  # 换行, 结束进度条
    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy],
                      llm=judge_llm, embeddings=judge_emb)
    # faithfulness:      回答中的每个 claim 能否从给定上下文推断 (防幻觉)
    # answer_relevancy:  回答与问题的语义相关程度

    print(f"\n  ---- 汇总 ({len(questions)} 条) ----")
    scores = {
        "faithfulness": _safe_score(result["faithfulness"]),
        "answer_relevancy": _safe_score(result["answer_relevancy"])
    }
    for name, score in scores.items():
        print(f"  {name:>20}: {score:.4f}")
    return scores


# ==================== Layer3: 端到端 ====================

def eval_end2end(retriever: Retriever, questions: list[dict], top_k: int = TOP_K):
    """Layer3: 端到端 — Retriever + Generator 真实串联, 全指标 + 诊断矩阵

    真正的"检索→生成"链路, 同时评估检索质量和生成质量,
    最后通过诊断矩阵判断优化方向.
    """
    generator = Generator(model=EVAL_LLM_MODEL_ID)
    judge_llm = _judge_llm()
    judge_emb = _judge_embeddings()

    # ---- 真实链路: 检索 + 生成 ----
    samples = []
    for i, q in enumerate(questions):
        chunks = retriever.retrieve(q["question"], top_k=top_k)  # 真实检索结果
        answer = generator.generate(q["question"], chunks)        # 基于检索结果生成
        samples.append({
            "user_input": q["question"],
            "response": answer,
            "retrieved_contexts": [c.content for c in chunks],    # 检索到的 chunk 内容
            "reference": q["ground_truth"]
        })
        print(f"  生成进度: [{i + 1}/{len(questions)}]", end="\r")

    print()  # 换行

    # ---- RAGAS 评估: 四项指标 ----
    dataset = EvaluationDataset.from_list(samples)
    gen_result = evaluate(dataset,
                          metrics=[faithfulness, answer_relevancy,
                                   context_precision, context_recall],
                          llm=judge_llm, embeddings=judge_emb)
    # context_precision: 检索到的上下文中, 有多少与问题真正相关 (信噪比)
    # context_recall:    参考答案中需要的信息, 检索到的上下文覆盖了多少 (召回)

    # ---- Retriever 独立评估: 检索质量 ----
    ret_result = eval_retriever(retriever, questions, top_k)

    # ---- 合并所有指标 ----
    all_metrics = {}
    all_metrics.update({f"ret_{k}": v for k, v in ret_result.items()})
    all_metrics["faithfulness"] = _safe_score(gen_result["faithfulness"])
    all_metrics["answer_relevancy"] = _safe_score(gen_result["answer_relevancy"])
    all_metrics["context_precision"] = _safe_score(gen_result["context_precision"])
    all_metrics["context_recall"] = _safe_score(gen_result["context_recall"])

    print(f"\n  ---- 全指标 ({len(questions)} 条) ----")
    for k, v in all_metrics.items():
        print(f"  {k:>25}: {v:.4f}")

    # ---- 诊断矩阵: 二分判断定位瓶颈 ----
    print(f"\n  ---- 诊断矩阵 ----")
    ret_ok = ret_result["recall"] >= 0.6                         # 检索的召回率阈值(>=0.6才认为相关)
    gen_ok = all_metrics.get("faithfulness", 0) >= 0.6           # 生成的忠实度阈值(>=0.6才认为相关)

    if ret_ok and gen_ok:
        print("  Retriever ✓ | Generator ✓  → 系统健康")
    elif ret_ok and not gen_ok:
        print("  Retriever ✓ | Generator ✗  → Generator有问题")
    elif not ret_ok and gen_ok:
        print("  Retriever ✗ | Generator ✓  → Retriever有问题")
    else:
        print("  Retriever ✗ | Generator ✗  → 双方都有问题")

    return all_metrics


# ==================== 主入口 ====================

def main():
    print("=" * 50)
    print("选择评估层级:")
    print("  1 — Layer1: 只测 Retriever (检索质量)")
    print("  2 — Layer2: 只测 Generator (生成质量, 跳过检索)")
    print("  3 — Layer3: 端到端 (全链路 + 诊断矩阵)")
    print("=" * 50)
    which_layer = input("请输入 1/2/3: ").strip()
    total_start = time.time()                                     # 全流程计时起点

    with open("benchmark.json", "r", encoding="utf-8") as f:
        questions = json.load(f)[:20]                             # 暂时取前 20 条, 控制评估成本

    print(f"\n{'=' * 50}")
    layer_names = {"1": "Layer1: Retriever 评估", "2": "Layer2: Generator 评估", "3": "Layer3: 端到端评估"}
    print(layer_names.get(which_layer, "Layer3: 端到端评估"))
    print(f"测试数据: benchmark.json ({len(questions)} 条)")
    print("=" * 50)

    store, retriever = build_index()
    print(f"索引构建完成, 共 {len(store._chunks)} 个 chunk\n")

    if which_layer == "1":
        eval_retriever(retriever, questions)
    elif which_layer == "2":
        eval_generator(questions)
    else:
        eval_end2end(retriever, questions)

    print(f"\n全流程耗时: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
