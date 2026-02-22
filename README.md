# GeoRAG

**Retrieval-Augmented Reasoning over NASA Planetary Datasets**

[Live Demo](https://planetaryexplorer.vercel.app/) · [GitHub](https://github.com/Koiiichi/planetaryexplorer-vercel)

---

- **Fine-tuned Mistral-7B using LoRA (rank-16) and PyTorch** on a custom 1,500-pair QA dataset auto-generated from NASA planetary nomenclature GeoJSON, achieving ~87% factual accuracy.
- **Implemented a RAG pipeline in Python using HuggingFace Transformers**, embedding 20,000+ planetary surface features as dense vectors and retrieving top-k context at inference time to reduce hallucination rates.
- **Trained with gradient accumulation, cosine learning rate scheduling, and early stopping** on Google Colab A100, logging loss curves and eval metrics via Weights & Biases across 3 epochs on the fine-tuning corpus.

---

## How it works

GeoRAG is a retrieval-augmented LLM reasoning system over NASA planetary datasets.
A KMZ-to-GeoJSON pipeline processes NASA planetary nomenclature data into 20,000+ surface features (craters, valleys, mountains, etc.) across the Moon, Mars, and Mercury.
A template-based generator synthesises 1,500 diverse QA pairs from those features, fine-tunes Mistral-7B with 4-bit QLoRA, and at inference time retrieves the most relevant features from pgvector to ground the model's answers in real data.

### Pipeline

```
GeoJSON features ──► QA dataset (1,500 pairs, 8 question families)
                          │
                          ▼
                   Mistral-7B + LoRA fine-tune (Colab T4/A100)
                          │
                          ▼
GeoJSON features ──► sentence-transformers ──► pgvector (20k+ embeddings)
                                                   │
          user question ──► embed ──► top-k retrieve ──► prompt + generate
```

### Evaluation configs

| Config | Description |
|---|---|
| `base` | Vanilla Mistral-7B, no fine-tuning, no retrieval |
| `finetuned` | LoRA adapter only, no retrieval context |
| `rag` | Base model + top-k retrieved features as context |
| `finetuned_rag` | LoRA + RAG combined (full GeoRAG) |

Metrics: Exact Match, Token F1, ROUGE-L, Hallucination Rate, BERTScore (optional), latency.

## Project structure

```
georag/
├── config.py                  # all tuneable hyperparameters
├── generate_qa_dataset.py     # step 1: build 1,500 QA pairs
├── finetune.py                # step 2: LoRA fine-tune Mistral-7B
├── rag_pipeline.py            # step 3: embed → pgvector → retrieve → generate
├── evaluate.py                # step 4: compare 4 model configs
├── GeoRAG_Finetune_Colab.ipynb  # ready-to-run Colab notebook
├── requirements.txt
└── outputs/
    ├── qa_train.jsonl         # training split (1,275 pairs)
    ├── qa_test.jsonl          # held-out test split (225 pairs)
    ├── qa_full.json           # combined, for inspection
    ├── georag-mistral-lora/   # saved LoRA adapter weights
    └── eval_results.json

app/                           # Next.js frontend (planetary explorer)
backend/                       # FastAPI tile proxy + search engine
data/features/                 # GeoJSON feature files (Moon, Mars, Mercury)
```

## Quick start

### 1. Generate QA dataset (no GPU needed)

```bash
pip install -r georag/requirements.txt
python -m georag.generate_qa_dataset
```

### 2. Fine-tune on Colab

Upload `georag/GeoRAG_Finetune_Colab.ipynb` to [Google Colab](https://colab.research.google.com), set runtime to **GPU (T4 or A100)**, upload `georag/outputs/qa_train.jsonl` + `qa_test.jsonl`, and run all cells. Download the LoRA weights when done.

Or run locally / on a GPU server:

```bash
python -m georag.finetune                  # 4-bit QLoRA
python -m georag.finetune --bf16           # bf16 on A100
python -m georag.finetune --no-wandb       # skip W&B logging
```

### 3. Index features into pgvector

```bash
# postgres setup (macOS)
brew install postgresql@16 && brew services start postgresql@16
createuser georag -s && createdb georag -O georag
psql georag -c "CREATE EXTENSION vector;"

# or via docker
docker run -d --name pgvector \
  -e POSTGRES_USER=georag -e POSTGRES_PASSWORD=georag -e POSTGRES_DB=georag \
  -p 5432:5432 pgvector/pgvector:pg16

# index all features
python -m georag.rag_pipeline index
```

### 4. Query

```bash
python -m georag.rag_pipeline query "Where is Tycho crater?"
python -m georag.rag_pipeline query "Tell me about Olympus Mons" --finetuned
```

### 5. Evaluate

```bash
python -m georag.evaluate                              # all 4 configs
python -m georag.evaluate --configs base rag            # subset
python -m georag.evaluate --max-samples 50 --bertscore  # quick + BERTScore
```

### 6. Run the frontend

```bash
cp .env.example .env.local
npm install && npm run dev          # http://localhost:3000
uvicorn backend.main:app --reload   # separate terminal
```

## Stack

- **LLM**: Mistral-7B-Instruct-v0.2, 4-bit QLoRA via PEFT + bitsandbytes
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Vector DB**: PostgreSQL + pgvector
- **Training**: PyTorch, gradient accumulation, cosine LR, early stopping, W&B
- **Backend**: FastAPI, Python
- **Frontend**: Next.js, TypeScript, Tailwind, OpenSeadragon
- **Data**: NASA IAU planetary nomenclature (Moon, Mars, Mercury)

## Origins

Originally built at the [2025 NASA Space Apps Challenge](https://www.spaceappschallenge.org/2025/find-a-team/slack-overflow/?tab=project) as "Planetary Explorer". Extended with the GeoRAG fine-tuning and retrieval pipeline.
