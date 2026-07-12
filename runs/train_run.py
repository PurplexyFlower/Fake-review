"""One parametrized training run under the frozen Rs-QLoRA revision protocol.

Writes results/<run_id>/{config.json,metrics.json,val_probs.csv,test_probs.csv,adapter/}
and appends one row to results/runs.csv (the ledger).

Examples:
  python runs/train_run.py --tag a1 --scaling rs --rank 64 --seed 1998
  python runs/train_run.py --tag b1 --scaling rs --rank 64 --seed 7 --scheduler cosine
  python runs/train_run.py --tag d1 --scaling rs --rank 64 --seed 1998 --holdout-category Home_and_Kitchen_5
"""
import pyarrow  # noqa: F401  must precede unsloth/torch on Windows (DLL init crash)

import argparse
import csv
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "fake reviews dataset.csv"
RESULTS = ROOT / "results"
LEDGER = RESULTS / "runs.csv"
SPLIT_SEED = 1998  # NEVER change: reproduces the paper's exact split

LEDGER_COLS = [
    "run_id", "timestamp", "tag", "scaling", "rank", "alpha", "seed",
    "scheduler", "label_smoothing", "holdout_category", "best_step",
    "best_val_acc", "test_acc", "test_f1_OR", "test_f1_CG", "test_roc_auc",
    "test_pr_auc", "train_runtime_s", "steps_per_s", "peak_alloc_gb",
    "peak_reserved_gb", "notes",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="phase tag, e.g. a1, b2, d1")
    p.add_argument("--scaling", choices=["standard", "rs"], required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--alpha", type=int, default=None, help="default 2*rank")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--scheduler", default="cosine_with_restarts",
                   choices=["cosine_with_restarts", "cosine", "linear"])
    p.add_argument("--label-smoothing", type=float, default=0.01)
    p.add_argument("--holdout-category", default=None,
                   help="LOCO mode: hold this category out as the test set")
    p.add_argument("--data", default=str(DATA),
                   help="dataset CSV (text_/label/category/rating); default = paper set")
    p.add_argument("--notes", default="")
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    alpha = args.alpha if args.alpha is not None else 2 * args.rank
    run_id = (f"{args.tag}_{args.scaling}_r{args.rank}_s{args.seed}"
              + (f"_loco-{args.holdout_category}" if args.holdout_category else "")
              + datetime.now().strftime("_%m%d%H%M"))
    run_dir = RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from datasets import load_dataset
    from unsloth import FastModel, is_bfloat16_supported
    from transformers import (AutoModelForSequenceClassification,
                              EarlyStoppingCallback, Trainer,
                              TrainingArguments, set_seed)
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, roc_auc_score)

    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    id2label = {0: "OR", 1: "CG"}
    label2id = {"OR": 0, "CG": 1}

    # ---- Data: exact paper split (SPLIT_SEED fixed, independent of --seed) ----
    full = load_dataset("csv", data_files=args.data)["train"]
    if args.holdout_category:
        held = full.filter(lambda x: x["category"] == args.holdout_category)
        rest = full.filter(lambda x: x["category"] != args.holdout_category)
        sp = rest.train_test_split(test_size=0.1, seed=SPLIT_SEED)
        train_ds, val_ds, test_ds = sp["train"], sp["test"], held
    else:
        sp = full.train_test_split(test_size=0.2, seed=SPLIT_SEED)
        sp2 = sp["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)
        train_ds, val_ds, test_ds = sp["train"], sp2["train"], sp2["test"]

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit",
        auto_model=AutoModelForSequenceClassification,
        max_seq_length=512, dtype=None, num_labels=2,
        full_finetuning=False, id2label=id2label, label2id=label2id,
        load_in_4bit=True,
    )
    model = FastModel.get_peft_model(
        model, r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=alpha, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=args.seed,
        use_rslora=(args.scaling == "rs"),  # rs: alpha/sqrt(r); standard: alpha/r
        loftq_config=None, task_type="SEQ_CLS",
    )

    def prep(ds):
        ds = ds.map(lambda b: tokenizer(b["text_"], truncation=True, max_length=512),
                    batched=True)
        # new "labels" column -> dtype inferred as int64 (datasets 4.x keeps the
        # original string dtype if you modify "label" in place — known footgun)
        ds = ds.map(lambda b: {"labels": [label2id[x] for x in b["label"]]},
                    batched=True)
        # Trainer's column pruning PROTECTS a column literally named "label", so
        # the original string column would reach the collator and crash. Keep
        # exactly what the model consumes.
        keep = {"input_ids", "attention_mask", "labels"}
        return ds.remove_columns([c for c in ds.column_names if c not in keep])

    train_tok, val_tok, test_tok = prep(train_ds), prep(val_ds), prep(test_ds)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.from_numpy(logits).float()
        p_cg = torch.softmax(logits, dim=-1)[:, 1].numpy()
        preds = (p_cg >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_OR": f1_score(labels, preds, pos_label=0),
            "f1_CG": f1_score(labels, preds, pos_label=1),
            "roc_auc": roc_auc_score(labels, p_cg),
            "pr_auc": average_precision_score(labels, p_cg),
        }

    sched_kwargs = {"num_cycles": 2} if args.scheduler == "cosine_with_restarts" else {}
    ckpt_dir = run_dir / "ckpt"
    trainer = Trainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_tok, eval_dataset=val_tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
        args=TrainingArguments(
            per_device_train_batch_size=32, gradient_accumulation_steps=2,
            warmup_ratio=0.06, num_train_epochs=6, learning_rate=5e-6,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=25, optim="adamw_8bit", weight_decay=0.03,
            eval_strategy="steps", eval_steps=100, max_grad_norm=1.0,
            label_smoothing_factor=args.label_smoothing,
            lr_scheduler_type=args.scheduler, lr_scheduler_kwargs=sched_kwargs,
            seed=args.seed, output_dir=str(ckpt_dir), report_to="none",
            save_strategy="steps", save_steps=100, load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy", greater_is_better=True,
            save_total_limit=2,
        ),
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_out = trainer.train()
    runtime = time.time() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_resv = torch.cuda.max_memory_reserved() / 1024**3
    best_step = trainer.state.global_step
    if trainer.state.best_model_checkpoint:
        best_step = int(trainer.state.best_model_checkpoint.rstrip("\\/").split("-")[-1])

    def dump_probs(tok_ds, raw_ds, name):
        out = trainer.predict(tok_ds)
        logits = torch.from_numpy(out.predictions).float()
        p_cg = torch.softmax(logits, dim=-1)[:, 1].numpy()
        labels = np.asarray(out.label_ids)
        with open(run_dir / f"{name}_probs.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row", "true", "p_CG", "category", "rating", "text"])
            for i in range(len(labels)):
                w.writerow([i, id2label[int(labels[i])], f"{p_cg[i]:.6f}",
                            raw_ds["category"][i], raw_ds["rating"][i],
                            raw_ds["text_"][i]])
        preds = (p_cg >= 0.5).astype(int)
        return {
            "acc": accuracy_score(labels, preds),
            "f1_OR": f1_score(labels, preds, pos_label=0),
            "f1_CG": f1_score(labels, preds, pos_label=1),
            "roc_auc": roc_auc_score(labels, p_cg),
            "pr_auc": average_precision_score(labels, p_cg),
        }

    val_m = dump_probs(val_tok, val_ds, "val")
    test_m = dump_probs(test_tok, test_ds, "test")

    end_data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if end_data_sha256 != data_sha256:
        raise RuntimeError(
            f"dataset changed during training: {data_path} "
            f"({data_sha256} -> {end_data_sha256})")

    model.save_pretrained(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))
    state_src = Path(trainer.state.best_model_checkpoint or ckpt_dir)
    if (state_src / "trainer_state.json").exists():
        shutil.copy(state_src / "trainer_state.json", run_dir / "trainer_state.json")
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # keep adapter, drop checkpoints

    config = vars(args) | {"alpha": alpha, "run_id": run_id,
                           "split_seed": SPLIT_SEED, "protocol": "frozen-v1",
                           "bf16": is_bfloat16_supported(),
                           "data_sha256": data_sha256}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    metrics = {"val": val_m, "test": test_m, "best_step": best_step,
               "train_runtime_s": runtime,
               "steps_per_s": train_out.metrics.get("train_steps_per_second"),
               "peak_alloc_gb": peak_alloc, "peak_reserved_gb": peak_resv}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Full eval/train loss trajectory THROUGH early-stop (not truncated at best
    # step like the checkpoint's trainer_state) — needed to see overfitting:
    # train loss -> 0 while eval_loss turns back up after the peak.
    (run_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2))

    row = {
        "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag, "scaling": args.scaling, "rank": args.rank,
        "alpha": alpha, "seed": args.seed, "scheduler": args.scheduler,
        "label_smoothing": args.label_smoothing,
        "holdout_category": args.holdout_category or "",
        "best_step": best_step, "best_val_acc": f"{val_m['acc']:.6f}",
        "test_acc": f"{test_m['acc']:.6f}",
        "test_f1_OR": f"{test_m['f1_OR']:.6f}",
        "test_f1_CG": f"{test_m['f1_CG']:.6f}",
        "test_roc_auc": f"{test_m['roc_auc']:.6f}",
        "test_pr_auc": f"{test_m['pr_auc']:.6f}",
        "train_runtime_s": f"{runtime:.1f}",
        "steps_per_s": f"{train_out.metrics.get('train_steps_per_second', 0):.4f}",
        "peak_alloc_gb": f"{peak_alloc:.2f}", "peak_reserved_gb": f"{peak_resv:.2f}",
        "notes": args.notes,
    }
    RESULTS.mkdir(exist_ok=True)
    new_file = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        if new_file:
            w.writeheader()
        w.writerow(row)

    print(f"\n[{run_id}] DONE  test_acc={test_m['acc']:.4f} "
          f"roc_auc={test_m['roc_auc']:.4f} best_step={best_step} "
          f"runtime={runtime/60:.0f}min")


if __name__ == "__main__":
    main()
