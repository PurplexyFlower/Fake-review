"""C3/C4/C5: same-split transformer baselines (full fine-tuning).

Generic over any HF sequence-classification model id, so it covers:
  C3 roberta-base, C4 microsoft/deberta-v3-base,
  C5 answerdotai/ModernBERT-large, google/gemma-3-270m
Same paper split (seed 1998), same text column, identical eval protocol as the
Rs-QLoRA runs. Reports test acc/F1/ROC/PR (full + leakage-cleaned), appends to
results/baselines.csv (model = "<short>@lr<lr>"), saves test_probs.

Usage:
  python runs/baseline_transformer.py --model roberta-base --seed 1998 --lr 2e-5
"""
import pyarrow  # noqa: F401
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "fake reviews dataset.csv"
RESULTS = ROOT / "results"
LEDGER = RESULTS / "baselines_transformer.csv"
SPLIT_SEED = 1998
LEDGER_COLS = ["model", "lr", "seed", "val_acc", "test_acc", "test_f1_OR",
               "test_f1_CG", "test_roc_auc", "test_pr_auc", "test_acc_cleaned"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=1998)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-seq", type=int, default=512)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--tag", default="", help="suffix on ledger model name")
    return p.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback,
                              Trainer, TrainingArguments, set_seed)
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, roc_auc_score)

    set_seed(args.seed)
    short = args.model.split("/")[-1] + args.tag
    tag = f"{short}@lr{args.lr:g}"
    data_path = Path(args.data).resolve()
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    config = vars(args) | {
        "data": str(data_path),
        "data_sha256": data_sha256,
        "split_seed": SPLIT_SEED,
        "protocol": "transformer-baseline-v1",
    }
    run_dir = RESULTS / (f"baseline_{short}_lr{args.lr:g}_s{args.seed}_"
                         f"{data_sha256[:12]}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"

    # Per-model resumability: run_all invokes this script once for each model.
    # Re-running the pipeline must not skip missing models merely because another
    # model already wrote the shared ledger.
    if (LEDGER.exists() and (run_dir / "test_probs.csv").exists()
            and config_path.exists()):
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        with LEDGER.open(newline="", encoding="utf-8") as f:
            done = any(
                row["model"] == short
                and row["lr"] == f"{args.lr:g}"
                and int(row["seed"]) == args.seed
                for row in csv.DictReader(f)
            )
        if done and saved_config == config:
            print(f"[{tag} s{args.seed}] already complete; skipping")
            return
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    full = load_dataset("csv", data_files=args.data)["train"]
    full = full.add_column("orig_idx", list(range(len(full))))
    sp = full.train_test_split(test_size=0.2, seed=SPLIT_SEED)
    sp2 = sp["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)
    train_ds, val_ds, test_ds = sp["train"], sp2["train"], sp2["test"]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2,
        id2label={0: "OR", 1: "CG"}, label2id={"OR": 0, "CG": 1})
    if tok.pad_token is None:  # decoder models (e.g. gemma) need a pad id
        tok.pad_token = tok.eos_token
        model.config.pad_token_id = tok.pad_token_id

    def prep(ds):
        ds = ds.map(lambda b: tok(b["text_"], truncation=True, max_length=args.max_seq),
                    batched=True)
        ds = ds.map(lambda b: {"labels": [1 if x == "CG" else 0 for x in b["label"]]},
                    batched=True)
        keep = {"input_ids", "attention_mask", "labels"}
        return ds.remove_columns([c for c in ds.column_names if c not in keep])

    train_tok, val_tok, test_tok = prep(train_ds), prep(val_ds), prep(test_ds)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        p = torch.softmax(torch.from_numpy(logits).float(), -1)[:, 1].numpy()
        return {"accuracy": accuracy_score(labels, (p >= 0.5).astype(int))}

    # transformers 4.56 no longer exports is_torch_bf16_gpu_available from its
    # top-level package. PyTorch provides the hardware check directly.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    trainer = Trainer(
        model=model, processing_class=tok,
        train_dataset=train_tok, eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        args=TrainingArguments(
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=64,
            gradient_accumulation_steps=max(1, 32 // args.batch),
            warmup_ratio=0.06, num_train_epochs=args.epochs, learning_rate=args.lr,
            bf16=bf16, fp16=not bf16, weight_decay=0.01, logging_steps=50,
            eval_strategy="steps", eval_steps=200, save_strategy="steps",
            save_steps=200, load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy", greater_is_better=True,
            save_total_limit=1, seed=args.seed, report_to="none",
            output_dir=str(run_dir / "ckpt")),
    )
    trainer.train()
    val_acc = trainer.evaluate().get("eval_accuracy", float("nan"))

    out = trainer.predict(test_tok)
    p_cg = torch.softmax(torch.from_numpy(out.predictions).float(), -1)[:, 1].numpy()
    y = np.asarray(out.label_ids)
    pred = (p_cg >= 0.5).astype(int)
    end_data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if end_data_sha256 != data_sha256:
        raise RuntimeError(
            f"dataset changed during training: {data_path} "
            f"({data_sha256} -> {end_data_sha256})")

    # Leakage exclusions were audited for the original Kaggle row indices only.
    # Applying those integers to a rebuilt modern dataset would remove unrelated
    # rows and produce a meaningless "cleaned" score.
    excl_path = RESULTS / "test_exclusions.csv"
    if Path(args.data).resolve() == DATA.resolve() and excl_path.exists():
        with excl_path.open(encoding="utf-8") as f:
            excl = {int(r["orig_idx"]) for r in csv.DictReader(f)}
        keep = np.array([oi not in excl for oi in test_ds["orig_idx"]])
    else:
        keep = np.ones(len(test_ds), dtype=bool)

    with open(run_dir / "test_probs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row", "true", "p_CG", "category", "rating"])
        for i in range(len(y)):
            w.writerow([i, "CG" if y[i] else "OR", f"{p_cg[i]:.6f}",
                        test_ds["category"][i], test_ds["rating"][i]])

    row = {
        "model": short, "lr": f"{args.lr:g}", "seed": args.seed,
        "val_acc": f"{val_acc:.6f}",
        "test_acc": f"{accuracy_score(y, pred):.6f}",
        "test_f1_OR": f"{f1_score(y, pred, pos_label=0):.6f}",
        "test_f1_CG": f"{f1_score(y, pred, pos_label=1):.6f}",
        "test_roc_auc": f"{roc_auc_score(y, p_cg):.6f}",
        "test_pr_auc": f"{average_precision_score(y, p_cg):.6f}",
        "test_acc_cleaned": f"{accuracy_score(y[keep], pred[keep]):.6f}",
    }
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    import shutil
    shutil.rmtree(run_dir / "ckpt", ignore_errors=True)
    print(f"[{tag} s{args.seed}] test_acc={row['test_acc']} roc={row['test_roc_auc']}")


if __name__ == "__main__":
    main()
