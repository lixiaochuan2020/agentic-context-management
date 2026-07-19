"""
Pilot Study 配置文件
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载 .env
load_dotenv(PROJECT_ROOT / ".env")

# ── 模型配置 ──────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3-4B-Instruct")  # 本地路径或 HF model ID

# ── 数据配置 ──────────────────────────────────────────────
DATA_PATH = str(PROJECT_ROOT / "data" / "browsecomp_subset_50.json")

# ── 实验参数 ──────────────────────────────────────────────
MAX_TURNS = 999_999             # 无限制，由 context window 或 final answer 自然终止

# NB: max_new_tokens moved to AgentConfig.rollout.max_new_tokens (see src/configs/agent.py)

# ── 结果目录 ──────────────────────────────────────────────
RESULTS_DIR = str(PROJECT_ROOT / "results")

# ── Serper 搜索配置 ──────────────────────────────────────
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_NUM_RESULTS = 5              # 每次搜索返回结果数（旧默认值，保持兼容）

# search: returns URL list + model calls open(url)
SEARCH_SEP_NUM_RESULTS = 50

# 网页正文截断字符数
MAX_PAGE_CHARS_SEP = 5000       # open 时每页截断

# ── Grader 配置（官方 simple-evals LLM-as-judge）─────────
GRADER_API_KEY = os.getenv("OPENAI_API_KEY", "")
GRADER_MODEL = "gpt-5"
