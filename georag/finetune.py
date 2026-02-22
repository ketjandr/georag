#!/usr/bin/env python3
"""
LoRA fine-tune Mistral-7B on GeoRAG QA data.
4-bit QLoRA on T4, bf16 on A100.  Logs to W&B.

  python -m georag.finetune
  python -m georag.finetune --bf16 --no-wandb
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from georag.config import (
    FINETUNED_MODEL_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_NAME,
    QA_OUTPUT_TEST,
    QA_OUTPUT_TRAIN,
    TRAIN_BATCH_SIZE,
    TRAIN_EARLY_STOPPING_PATIENCE,
    TRAIN_EPOCHS,
    TRAIN_GRAD_ACCUM_STEPS,
    TRAIN_LR,
    TRAIN_MAX_SEQ_LEN,
    TRAIN_WARMUP_RATIO,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
)

# --- dataset / tokenisation ---

SYSTEM_PROMPT = (
    "You are GeoRAG, an expert assistant for NASA planetary science. "
    "Answer questions about surface features on the Moon, Mars, Mercury, "
    "and other celestial bodies using precise nomenclature data."
)


def _format_prompt(question: str, context: str = "") -> str:
    parts = [f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"]
    if context:
        parts.append(f"Context:\n{context}\n\n")
    parts.append(f"{question} [/INST]")
    return "".join(parts)


class QADataset(Dataset):

    def __init__(self, path: Path, tokenizer, max_len: int = TRAIN_MAX_SEQ_LEN):
        self.samples: list[dict] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        prompt = _format_prompt(item["question"])
        full_text = f"{prompt} {item['answer']}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # only compute loss on the answer portion
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# --- model loading ---

def load_model_and_tokenizer(use_4bit: bool = True, bf16: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quant_config = None
    torch_dtype = torch.float16
    if bf16:
        torch_dtype = torch.bfloat16
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# --- training loop ---

def evaluate(model, dataloader, device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n += 1
    return total_loss / max(n, 1)


def train(
    model,
    tokenizer,
    train_dataset: QADataset,
    val_dataset: QADataset,
    *,
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    grad_accum: int = TRAIN_GRAD_ACCUM_STEPS,
    lr: float = TRAIN_LR,
    warmup_ratio: float = TRAIN_WARMUP_RATIO,
    patience: int = TRAIN_EARLY_STOPPING_PATIENCE,
    use_wandb: bool = True,
):

    if use_wandb:
        import wandb
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
            "model": MODEL_NAME, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "epochs": epochs, "batch_size": batch_size,
            "effective_batch": batch_size * grad_accum,
            "lr": lr, "warmup_ratio": warmup_ratio,
        })

    device = next(model.parameters()).device

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=2)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=0.01,
    )

    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": outputs.loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                    })

                if global_step % 50 == 0:
                    print(f"  step {global_step:>5d}  loss={outputs.loss.item():.4f}  "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")

        avg_train = epoch_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs}  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")

        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "train/epoch_loss": avg_train,
                        "val/loss": val_loss})

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = str(FINETUNED_MODEL_DIR)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  ✓ Saved best model  →  {save_path}")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("  ✗ Early stopping triggered.")
                break

    if use_wandb:
        import wandb
        wandb.finish()
    print(f"✓ Training complete.  Best val loss = {best_val_loss:.4f}")



def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B with LoRA for GeoRAG")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 (A100)")
    parser.add_argument("--no-4bit", action="store_true", help="Skip 4-bit quantisation")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    args = parser.parse_args()

    use_4bit = not args.no_4bit
    model, tokenizer = load_model_and_tokenizer(use_4bit=use_4bit, bf16=args.bf16)

    train_ds = QADataset(QA_OUTPUT_TRAIN, tokenizer)
    val_ds = QADataset(QA_OUTPUT_TEST, tokenizer)
    print(f"✓ Train: {len(train_ds)}  Val: {len(val_ds)}")

    train(
        model, tokenizer, train_ds, val_ds,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
