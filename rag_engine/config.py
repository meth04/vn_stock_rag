import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/financial_reports"
GOLDEN_DATASET_PATH = PROJECT_ROOT / "data" / "golden_dataset.json"
WORK_DIR = PROJECT_ROOT / "lightrag_index"

# --- PROXY CONFIG ---
OPENAI_BASE_URL = os.getenv("PROXY_BASE_URL", "http://localhost:8317/v1")
OPENAI_API_KEY = os.getenv("PROXY_API_KEY", "sk-fake-key")

# --- MODEL SELECTION ---
INDEXING_MODEL = "deepseek-v3"
QUERY_MODEL = "deepseek-v3"
JUDGE_MODEL = "qwen3-max"

# --- EMBEDDING LOCAL ---
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
DEVICE = "cuda"
MAX_TOKEN_SIZE = 8192

GENERATION_PARAMS = {
    "temperature": 0.0,
    "max_tokens": 1500,
    "top_p": 0.95,
}

# --- EVALUATION ---
DEFAULT_EVAL_SAMPLE_SIZE = int(os.getenv("RAG_EVAL_SAMPLE_SIZE", "50"))
DEFAULT_EVAL_OUTPUT_DIR = PROJECT_ROOT / "rag_engine" / "artifacts"
