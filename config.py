import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # 基于 config.py 自身位置定位 .env, 不依赖 CWD

# DeepSeek API 配置，从 .env 读取
LLM_API_KEY = os.getenv("LLM_API_KEY")        # API 密钥
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")      # 模型 ID，如 deepseek-v4-pro
LLM_BASE_URL = os.getenv("LLM_BASE_URL")      # API 地址，如 https://api.deepseek.com

EVAL_LLM_MODEL_ID = os.getenv("EVAL_LLM_MODEL_ID", "deepseek-chat")  # 评估专用低成本模型

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
