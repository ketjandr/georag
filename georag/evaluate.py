#!/usr/bin/env python3
"""
Eval harness — compare base / finetuned / RAG / finetuned+RAG
on held-out test set.  Measures EM, F1, ROUGE-L, hallucination rate.

  python -m georag.evaluate
  python -m georag.evaluate --configs base rag --max-samples 50
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from georag.config import (
    EVAL_OUTPUT,
    FINETUNED_MODEL_DIR,
    MODEL_NAME,
    QA_OUTPUT_TEST,
    RAG_TOP_K,
)

# --- text normalisation ---

def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return _normalise(text).split()


# --- metrics ---

def exact_match(pred: str, gold: str) -> float:
    return float(_normalise(pred) == _normalise(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    precision = len(common) / len(pred_toks)
    recall = len(common) / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    p = _tokens(pred)
    g = _tokens(gold)
    if not p or not g:
        return float(p == g)
    m, n = len(p), len(g)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i - 1] == g[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def hallucination_rate(pred: str, gold: str, context: str = "") -> float:
    # grab capitalised spans as rough NER
    pred_entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", pred))
    if not pred_entities:
        return 0.0
    reference_text = (gold + " " + context).lower()
    hallucinated = sum(1 for ent in pred_entities if ent.lower() not in reference_text)
    return hallucinated / len(pred_entities)


def compute_bertscore(preds: list[str], golds: list[str]) -> list[float]:
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(preds, golds, lang="en", verbose=False,
                                   rescale_with_baseline=True)
        return F1.tolist()
    except ImportError:
        print("⚠  bert_score not installed — skipping BERTScore.")
        return [0.0] * len(preds)


# --- per-config inference ---

CONFIG_NAMES = ["base", "finetuned", "rag", "finetuned_rag"]


def _load_test_set(max_samples: int | None = None) -> list[dict]:
    with open(QA_OUTPUT_TEST, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if max_samples:
        items = items[:max_samples]
    return items


def _run_config(
    config_name: str,
    test_items: list[dict],
) -> list[dict]:
    from georag.rag_pipeline import (
        format_context,
        generate_answer,
        load_generation_model,
        retrieve,
    )

    use_finetuned = config_name in ("finetuned", "finetuned_rag")
    use_rag = config_name in ("rag", "finetuned_rag")

    print(f"\n{'─'*60}")
    print(f"Config: {config_name}  (finetuned={use_finetuned}, rag={use_rag})")
    print(f"{'─'*60}")

    model, tokenizer = load_generation_model(use_finetuned=use_finetuned)

    results = []
    for item in tqdm(test_items, desc=config_name):
        context = ""
        retrieved = []
        if use_rag:
            retrieved = retrieve(item["question"], top_k=RAG_TOP_K)
            context = format_context(retrieved)

        t0 = time.time()
        pred = generate_answer(item["question"], model, tokenizer, context=context)
        elapsed = time.time() - t0

        gold = item["answer"]
        results.append({
            "id": item.get("id"),
            "question": item["question"],
            "gold": gold,
            "pred": pred,
            "context": context,
            "exact_match": exact_match(pred, gold),
            "token_f1": token_f1(pred, gold),
            "rouge_l": rouge_l(pred, gold),
            "hallucination_rate": hallucination_rate(pred, gold, context),
            "latency_s": elapsed,
        })

    # free gpu mem
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# --- aggregation & reporting ---

def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    metrics = {}
    for key in ("exact_match", "token_f1", "rouge_l", "hallucination_rate", "latency_s"):
        vals = [r[key] for r in results]
        metrics[key] = {
            "mean": sum(vals) / n,
            "min": min(vals),
            "max": max(vals),
        }
    return metrics


def print_report(all_results: dict[str, dict]):
    header = f"{'Config':<18} {'EM':>6} {'F1':>6} {'ROUGE-L':>8} {'Halluc%':>8} {'BERTSc':>7} {'Latency':>8}"
    print(f"\n{'═'*70}")
    print(header)
    print(f"{'═'*70}")
    for config, agg in all_results.items():
        em = agg.get("exact_match", {}).get("mean", 0)
        f1 = agg.get("token_f1", {}).get("mean", 0)
        rl = agg.get("rouge_l", {}).get("mean", 0)
        hall = agg.get("hallucination_rate", {}).get("mean", 0)
        bs = agg.get("bertscore_f1", {}).get("mean", 0)
        lat = agg.get("latency_s", {}).get("mean", 0)
        print(f"{config:<18} {em:>6.3f} {f1:>6.3f} {rl:>8.3f} {hall:>7.1%} {bs:>7.3f} {lat:>7.2f}s")
    print(f"{'═'*70}")



def main():
    parser = argparse.ArgumentParser(description="GeoRAG evaluation harness")
    parser.add_argument("--configs", nargs="+", default=CONFIG_NAMES,
                        choices=CONFIG_NAMES,
                        help="Which configurations to evaluate")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of test samples (for quick runs)")
    parser.add_argument("--bertscore", action="store_true",
                        help="Compute BERTScore (slower)")
    args = parser.parse_args()

    test_items = _load_test_set(args.max_samples)
    print(f"✓ Loaded {len(test_items)} test samples")

    all_agg: dict[str, dict] = {}
    all_raw: dict[str, list[dict]] = {}

    for config in args.configs:
        raw = _run_config(config, test_items)

        # Optional BERTScore
        if args.bertscore:
            preds = [r["pred"] for r in raw]
            golds = [r["gold"] for r in raw]
            bs_scores = compute_bertscore(preds, golds)
            for r, bs in zip(raw, bs_scores):
                r["bertscore_f1"] = bs

        agg = aggregate(raw)

        if args.bertscore:
            bs_vals = [r.get("bertscore_f1", 0) for r in raw]
            agg["bertscore_f1"] = {
                "mean": sum(bs_vals) / len(bs_vals),
                "min": min(bs_vals),
                "max": max(bs_vals),
            }

        all_agg[config] = agg
        all_raw[config] = raw

    # report
    print_report(all_agg)

    # save
    output = {
        "n_samples": len(test_items),
        "configs": {c: {"aggregated": all_agg[c]} for c in args.configs},
    }
    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {EVAL_OUTPUT}")


if __name__ == "__main__":
    main()
