"""Shared config for the GeoRAG pipeline."""
from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "features"
GEORAG_DIR = PROJECT_ROOT / "georag"
OUTPUT_DIR = GEORAG_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILES = [
    DATA_DIR / "moon_features.json",
    DATA_DIR / "mars_features.json",
    DATA_DIR / "mercury_features.json",
]
ALL_FEATURES_FILE = DATA_DIR / "all_features.json"

# qa generation
QA_TOTAL_TARGET = 1500
QA_TRAIN_RATIO = 0.85          # 85 % train, 15 % test
QA_OUTPUT_TRAIN = OUTPUT_DIR / "qa_train.jsonl"
QA_OUTPUT_TEST = OUTPUT_DIR / "qa_test.jsonl"
QA_OUTPUT_FULL = OUTPUT_DIR / "qa_full.json"
QA_SEED = 42

# fine-tuning
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
TRAIN_GRAD_ACCUM_STEPS = 8        # effective batch = 32
TRAIN_LR = 2e-4
TRAIN_WARMUP_RATIO = 0.06
TRAIN_MAX_SEQ_LEN = 1024
TRAIN_EARLY_STOPPING_PATIENCE = 3

FINETUNED_MODEL_DIR = OUTPUT_DIR / "georag-mistral-lora"

# rag / embeddings
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
PG_CONN_STRING = "postgresql+psycopg://georag:georag@localhost:5432/georag"
PG_TABLE_NAME = "feature_embeddings"
RAG_TOP_K = 5

# evaluation
EVAL_OUTPUT = OUTPUT_DIR / "eval_results.json"

# wandb
WANDB_PROJECT = "georag"
WANDB_RUN_NAME = "mistral-7b-lora-r16"
