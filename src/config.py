"""Single source of truth for project-wide constants."""
from pathlib import Path

SEED = 42

CATEGORIES = ["cs.CL", "cs.LG", "cs.CV", "cs.IR", "cs.AI", "stat.ML"]
LABEL_NAMES = {
    "cs.CL":   "cl_nlp",
    "cs.LG":   "lg_ml",
    "cs.CV":   "cv_vision",
    "cs.IR":   "ir_retrieval",
    "cs.AI":   "ai_general",
    "stat.ML": "stat_ml",
}

# Sampling
PER_CATEGORY_TARGET = 800
MIN_ABSTRACT_WORDS = 50
TIME_RANGE = ("2023-01-01", "2025-12-31")

# Embedders (slug -> HF model id). Slug is what the CLI accepts.
EMBEDDERS = {
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5":     "intfloat/e5-base-v2",
}
PRIMARY_EMBEDDER_SLUG = "mpnet"

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw_arxiv.jsonl"
PROCESSED_DIR = DATA_DIR / "processed"
EMB_DIR = DATA_DIR / "embeddings"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
RESULTS_DIR = ROOT / "results"

# HF Hub
HF_USERNAME = "Tiansve"
HF_DATASET_REPO = f"{HF_USERNAME}/arxiv-cs-subfield"
HF_MODEL_REPO = f"{HF_USERNAME}/arxiv-subfield-linear-probe"
HF_SPACE_REPO = f"{HF_USERNAME}/arxiv-subfield-demo"
