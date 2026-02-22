#!/usr/bin/env python3
"""
Dense-vector RAG over planetary features.
Embeds features into pgvector, retrieves top-k at query time,
injects as context before generating with Mistral.

  python -m georag.rag_pipeline index
  python -m georag.rag_pipeline query "Where is Tycho crater?"
  python -m georag.rag_pipeline query "Where is Tycho?" --finetuned
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    create_engine,
    text as sa_text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

from georag.config import (
    EMBED_DIM,
    EMBED_MODEL_NAME,
    FINETUNED_MODEL_DIR,
    MODEL_NAME,
    PG_CONN_STRING,
    PG_TABLE_NAME,
    RAG_TOP_K,
    TRAIN_MAX_SEQ_LEN,
)

# --- db schema ---

Base = declarative_base()


class FeatureEmbedding(Base):
    __tablename__ = PG_TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_id = Column(String(256), unique=True, nullable=False, index=True)
    name = Column(String(512), nullable=False)
    body = Column(String(64), nullable=False)
    category = Column(String(128), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    diameter_km = Column(Float, nullable=True)
    origin = Column(Text, nullable=True)
    text_blob = Column(Text, nullable=False)
    embedding = Column(Vector(EMBED_DIM), nullable=False)


# --- embedding ---

_embed_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def feature_to_text(feat: dict) -> str:
    """Turn a feature dict into a sentence for embedding."""
    parts = [
        f"{feat['name']} is a {feat['category'].lower()} on {feat['body'].title()}.",
        f"Coordinates: {abs(feat['lat']):.2f}°{'N' if feat['lat'] >= 0 else 'S'}, "
        f"{abs(feat['lon']):.2f}°{'E' if feat['lon'] >= 0 else 'W'}.",
    ]
    if feat.get("diameter_km"):
        parts.append(f"Diameter: {feat['diameter_km']} km.")
    if feat.get("origin"):
        parts.append(f"Named after: {feat['origin']}.")
    if feat.get("keywords"):
        parts.append(f"Keywords: {', '.join(feat['keywords'])}.")
    return " ".join(parts)


def embed_texts(texts: list[str], batch_size: int = 256) -> np.ndarray:
    model = get_embed_model()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True)


# --- indexing ---

def get_engine():
    return create_engine(PG_CONN_STRING, echo=False)


def _ensure_pgvector_extension(engine):
    with engine.connect() as conn:
        conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def index_features(feature_files: list[Path] | None = None):
    from georag.generate_qa_dataset import load_all_features

    features = load_all_features()

    # Build text blobs
    print("Embedding features …")
    text_blobs = [feature_to_text(f) for f in features]
    vectors = embed_texts(text_blobs)

    # Database
    engine = get_engine()
    _ensure_pgvector_extension(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # insert in batches
    BATCH = 500
    for i in range(0, len(features), BATCH):
        batch_feats = features[i : i + BATCH]
        batch_vecs = vectors[i : i + BATCH]
        batch_texts = text_blobs[i : i + BATCH]

        for feat, vec, blob in zip(batch_feats, batch_vecs, batch_texts):
            existing = session.query(FeatureEmbedding).filter_by(
                feature_id=feat["id"]
            ).first()
            if existing:
                existing.embedding = vec.tolist()
                existing.text_blob = blob
            else:
                session.add(FeatureEmbedding(
                    feature_id=feat["id"],
                    name=feat["name"],
                    body=feat["body"],
                    category=feat["category"],
                    lat=feat["lat"],
                    lon=feat["lon"],
                    diameter_km=feat.get("diameter_km"),
                    origin=feat.get("origin"),
                    text_blob=blob,
                    embedding=vec.tolist(),
                ))
        session.commit()
        print(f"  indexed {min(i + BATCH, len(features))}/{len(features)}")

    session.close()
    print(f"✓ Indexed {len(features)} features into pgvector ({PG_TABLE_NAME})")


# --- retrieval ---

def retrieve(query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    model = get_embed_model()
    q_vec = model.encode([query], normalize_embeddings=True)[0].tolist()

    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    results = (
        session.query(FeatureEmbedding)
        .order_by(FeatureEmbedding.embedding.cosine_distance(q_vec))
        .limit(top_k)
        .all()
    )

    retrieved = []
    for row in results:
        retrieved.append({
            "feature_id": row.feature_id,
            "name": row.name,
            "body": row.body,
            "category": row.category,
            "lat": row.lat,
            "lon": row.lon,
            "diameter_km": row.diameter_km,
            "text_blob": row.text_blob,
        })
    session.close()
    return retrieved


def format_context(retrieved: list[dict]) -> str:
    lines = ["Retrieved planetary feature data:"]
    for i, r in enumerate(retrieved, 1):
        lines.append(f"  [{i}] {r['text_blob']}")
    return "\n".join(lines)


# --- generation ---

def load_generation_model(use_finetuned: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    model_path = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_finetuned:
        adapter_path = str(FINETUNED_MODEL_DIR)
        print(f"Loading LoRA adapter from {adapter_path} …")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


SYSTEM_PROMPT = (
    "You are GeoRAG, an expert assistant for NASA planetary science. "
    "Answer questions about surface features on the Moon, Mars, Mercury, "
    "and other celestial bodies using precise nomenclature data. "
    "Use the retrieved context to ground your answers in facts."
)


def generate_answer(
    question: str,
    model,
    tokenizer,
    context: str = "",
    max_new_tokens: int = 256,
) -> str:
    parts = [f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"]
    if context:
        parts.append(f"{context}\n\n")
    parts.append(f"{question} [/INST]")
    prompt = "".join(parts)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=TRAIN_MAX_SEQ_LEN).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
        )

    # decode only new tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# --- end-to-end query ---

def rag_query(
    question: str,
    use_finetuned: bool = False,
    top_k: int = RAG_TOP_K,
) -> dict:
    retrieved = retrieve(question, top_k=top_k)
    context = format_context(retrieved)
    model, tokenizer = load_generation_model(use_finetuned=use_finetuned)
    answer = generate_answer(question, model, tokenizer, context=context)
    return {
        "question": question,
        "answer": answer,
        "retrieved": retrieved,
        "context": context,
    }



def main():
    parser = argparse.ArgumentParser(description="GeoRAG pipeline")
    sub = parser.add_subparsers(dest="command")

    # index sub-command
    sub.add_parser("index", help="Embed and index all features into pgvector")

    # query sub-command
    q_parser = sub.add_parser("query", help="Ask a question with RAG")
    q_parser.add_argument("question", type=str)
    q_parser.add_argument("--finetuned", action="store_true",
                          help="Use fine-tuned LoRA model")
    q_parser.add_argument("--top-k", type=int, default=RAG_TOP_K)

    args = parser.parse_args()

    if args.command == "index":
        index_features()
    elif args.command == "query":
        result = rag_query(args.question, use_finetuned=args.finetuned,
                           top_k=args.top_k)
        print(f"\n{'='*60}")
        print(f"Question: {result['question']}")
        print(f"{'='*60}")
        print(f"Answer:   {result['answer']}")
        print(f"{'='*60}")
        print(f"Retrieved {len(result['retrieved'])} features:")
        for r in result["retrieved"]:
            print(f"  • {r['name']} ({r['category']}, {r['body']})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
