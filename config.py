import os
from dotenv import load_dotenv

load_dotenv()  # 从项目根目录的 .env 文件中加载环境变量到 os.environ

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
