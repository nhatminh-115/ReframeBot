import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


LABEL_MAP = {0: "TASK_1", 1: "TASK_2", 2: "TASK_3"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = norm_text(obj.get("text", ""))
            label = obj.get("label", None)
            if text == "" or label is None:
                continue
            rows.append({"text": text, "label": int(label)})
    return rows


def stratified_split(items, label_key="label", val_ratio=0.1, seed=42):
    buckets = {}
    for item in items:
        buckets.setdefault(item[label_key], []).append(item)

    rng = random.Random(seed)
    train, val = [], []
    for lbl, bucket in buckets.items():
        rng.shuffle(bucket)
        n_val = max(1, int(len(bucket) * val_ratio))
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train/retrain guardrail DistilBERT classifier")
    parser.add_argument("--data", default="data/guardrail_dataset_clean.jsonl")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--out", default="guardrail_model_retrained")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.2)  # 0.2 matches the published run (335 val samples)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. "
            "If you intended to use the cleaned dataset, run: "
            "python scripts\\prepare_guardrail_data.py --out data\\guardrail_dataset_clean.jsonl"
        )

    rows = load_jsonl(data_path)
    if len(rows) < 50:
        raise RuntimeError(f"Too few rows in {data_path}: {len(rows)}")

    train_rows, val_rows = stratified_split(rows, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Loaded {len(rows)} rows. Train={len(train_rows)} Val={len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label={i: LABEL_MAP[i] for i in LABEL_MAP},
        label2id={LABEL_MAP[i]: i for i in LABEL_MAP},
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=["text"])
    val_ds = Dataset.from_list(val_rows).map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    preds = trainer.predict(val_ds)
    y_true = np.array(val_ds["label"], dtype=int)
    y_pred = np.argmax(preds.predictions, axis=-1)

    print("\n=== Validation report ===")
    print(classification_report(y_true, y_pred, target_names=[LABEL_MAP[i] for i in range(3)]))

    # Save final
    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))

    print(f"\nSaved best model to: {out_dir / 'best'}")
    print("To use it in app.py:")
    print(f"  set GUARDRAIL_PATH={out_dir / 'best'}")


if __name__ == "__main__":
    main()
