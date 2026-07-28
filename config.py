"""项目根配置：LLM API、Embedding 模型、chunk 参数、eval 并发与重试、Generator 配置指纹。

核心特性：
    - 所有路径基于 __file__ 推导 _PROJECT_ROOT，不依赖 CWD
    - .env 通过 load_dotenv 加载，常量通过 os.getenv 读取并带默认值
    - PROMPT_TEMPLATE 从 retrieval/generator.py 迁入（避免循环导入且语义为配置常量）
    - GENERATOR_CONFIG_HASH 模块级一次计算，sha256(model + temperature + max_tokens + top_p + prompt_template) 捕获 Generator 全部配置变更

用法示例::

    from config import LLM_MODEL_ID, GENERATOR_CONFIG_HASH, TOP_K, _PROJECT_ROOT
    data_path = _PROJECT_ROOT / "data"

公共接口（常量）：
    - LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL: DeepSeek API 配置
    - EVAL_LLM_MODEL_ID: 评估专用低成本模型
    - PROMPT_TEMPLATE: Generator 中文 RAG prompt 模板
    - GENERATOR_TEMPERATURE / GENERATOR_MAX_TOKENS / GENERATOR_TOP_P: Generator 配置（eval 场景默认 temperature=0）
    - GENERATOR_CONFIG_HASH: Generator 配置指纹（12 位 hex）
    - LIVE_PANEL_MODE: 终端面板模式（"ansi" | "plain"）
    - EMBEDDING_MODEL / CHUNK_SIZE / CHUNK_OVERLAP / TOP_K / RELEVANCE_THRESHOLD: 检索管线参数
    - _PROJECT_ROOT: 项目根目录 Path（CWD 无关）
    - EVAL_MAX_WORKERS / JUDGE_MAX_RETRIES / JUDGE_BASE_DELAY / JUDGE_DEADLINE: Eval 并发与重试参数
    - COST_INPUT_PRICE_PER_1K / COST_OUTPUT_PRICE_PER_1K: 成本估算单价
"""

import hashlib
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # 基于 config.py 自身位置定位 .env, 不依赖 CWD

# DeepSeek API 配置，从 .env 读取
LLM_API_KEY = os.getenv("LLM_API_KEY")        # API 密钥
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")      # 模型 ID，如 deepseek-v4-pro
LLM_BASE_URL = os.getenv("LLM_BASE_URL")      # API 地址，如 https://api.deepseek.com

EVAL_LLM_MODEL_ID = os.getenv("EVAL_LLM_MODEL_ID", "deepseek-chat")  # 评估专用低成本模型

# === Generator prompt template（从 retrieval/generator.py 迁入, 本质是配置常量）===
PROMPT_TEMPLATE = """
## Role
你是一个专业的知识库问答助手, 你的任务是严格根据提供的【参考文档】回答用户的问题

## Rules(关键)
1.必须**仅依赖**下方的【参考文档】进行回答, 不要使用你内部的训练知识
2.如果【参考文档】中没有包含回答问题所需的信息, 请直接回答: "知识库中未找到相关信息". **严禁编造**
3.回答需要简洁, 逻辑清晰, 准确, 有条理, 分点描述
4.引用来源时标注[来源X], 在回答的末尾注明引用的文档名称

## Context(检索到的片段)
以下是参考文档片段:
<context>
{context_str}
</context>

## User Question
用户问题是:
{query_str}

## 回答
请开始回答:"""

# === Generator 配置常量（eval 场景专用）===
GENERATOR_TEMPERATURE = float(os.getenv("GENERATOR_TEMPERATURE", "0"))
GENERATOR_MAX_TOKENS = int(os.getenv("GENERATOR_MAX_TOKENS", "0")) or None  # 0 表示不截断
GENERATOR_TOP_P = float(os.getenv("GENERATOR_TOP_P", "1.0"))

# === 监控模式 ===
LIVE_PANEL_MODE = os.getenv("LIVE_PANEL_MODE", "ansi")  # "ansi" 或 "plain"

# === Generator 配置指纹（模块级常量, 一次计算, 整个 eval run 不变）===
_generator_fingerprint = "|".join([
    LLM_MODEL_ID,
    str(GENERATOR_TEMPERATURE),
    str(GENERATOR_MAX_TOKENS),
    str(GENERATOR_TOP_P),
    hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:16],
])
GENERATOR_CONFIG_HASH = hashlib.sha256(_generator_fingerprint.encode()).hexdigest()[:12]


# Embedding 模型名称，首次运行自动从 HuggingFace 下载到本地缓存
EMBEDDING_MODEL = "D:/Model"  # BGE-M3, 1024 维, 本地路径

# chunk配置层
CHUNK_SIZE = 500      # 文本切分的窗口大小（字符数）
CHUNK_OVERLAP = 100   # 相邻 chunk 之间的重叠量（字符数）
TOP_K = 5             # 检索时返回相似度最高的 Top-K 个 chunk
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))  # 余弦相似度 >= 此值认为 chunk 与 GT 相关

# ---- 项目根（CWD 无关） ----
_PROJECT_ROOT = Path(__file__).resolve().parent

# ---- Eval 并发 ----
EVAL_MAX_WORKERS = int(os.getenv("EVAL_MAX_WORKERS", "5"))  # 外层线程池 worker 数（query 级并发）

# ---- Judge 重试 ----
JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "3"))        # 最大重试次数
JUDGE_BASE_DELAY = float(os.getenv("JUDGE_BASE_DELAY", "1.0"))      # 指数退避初始延迟（秒）
JUDGE_DEADLINE = float(os.getenv("JUDGE_DEADLINE", "120.0"))        # 单次 LLM 调用超时（秒）

# ---- 成本监控 ----
COST_INPUT_PRICE_PER_1K = float(os.getenv("COST_INPUT_PRICE_PER_1K", "0.0003"))
COST_OUTPUT_PRICE_PER_1K = float(os.getenv("COST_OUTPUT_PRICE_PER_1K", "0.0012"))
