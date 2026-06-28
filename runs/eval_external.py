"""Zero-shot evaluation of the headline Rs-QLoRA r64 detector on any review CSV.

Used for:
  D3  modern-LLM reviews   (all machine -> report DETECTION RATE / recall)
  D2  Ott deceptive corpus (deceptive vs truthful -> transfer metrics)
  D2  genuine Amazon sample (all human -> report FALSE-POSITIVE RATE)

  python runs/eval_external.py --csv dataset/llm_generated_reviews.csv \
         --name d3_llm --text-col text_ --label-col label --pos-value CG
  python runs/eval_external.py --csv dataset/external/amazon_genuine.csv \
         --name d2_amazon_fpr --text-col text_           # no labels -> FPR mode
"""
import pyarrow  # noqa: F401
import argparse
import csv
import json
from pathlib import Path

from _paths import headline_run_dir

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "external_eval"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--text-col", default="text_")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--pos-value", default="CG",
                    help="label value that maps to CG(1); all else -> OR(0)")
    ap.add_argument("--adapter", default=None,
                    help="adapter dir (default: latest headline a1 rs/r64 run)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    return ap.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from unsloth import FastModel
    from transformers import AutoModelForSequenceClassification

    adapter = Path(args.adapter) if args.adapter else headline_run_dir() / "adapter"
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.max_rows:
        rows = rows[:args.max_rows]
    texts = [r[args.text_col] for r in rows]
    has_label = args.label_col is not None
    y = (np.array([1 if r[args.label_col] == args.pos_value else 0 for r in rows])
         if has_label else None)

    model, tokenizer = FastModel.from_pretrained(
        model_name=str(adapter), auto_model=AutoModelForSequenceClassification,
        max_seq_length=512, dtype=None, num_labels=2, load_in_4bit=True,
        full_finetuning=False)
    FastModel.for_inference(model); model.eval()
    dev = next(model.parameters()).device

    p_cg = np.zeros(len(texts), dtype=float)
    for i in range(0, len(texts), args.batch):
        enc = tokenizer(texts[i:i + args.batch], truncation=True, max_length=512,
                        padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**enc).logits.float()
        p_cg[i:i + args.batch] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    pred = (p_cg >= 0.5).astype(int)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / f"{args.name}_probs.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row", "p_CG", "pred", "text"])
        for i in range(len(texts)):
            w.writerow([i, f"{p_cg[i]:.6f}", "CG" if pred[i] else "OR",
                        texts[i][:300]])

    res = {"name": args.name, "csv": str(args.csv), "n": len(texts),
           "mean_p_CG": float(p_cg.mean()),
           "flag_rate_CG": float(pred.mean())}
    if has_label:
        from sklearn.metrics import (accuracy_score, average_precision_score,
                                     f1_score, roc_auc_score)
        res["n_pos"] = int(y.sum()); res["n_neg"] = int((y == 0).sum())
        res["accuracy"] = float(accuracy_score(y, pred))
        res["f1_CG"] = float(f1_score(y, pred, pos_label=1, zero_division=0))
        res["recall_CG"] = float(((pred == 1) & (y == 1)).sum() / max(y.sum(), 1))
        res["fpr"] = float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
        if 0 < y.sum() < len(y):  # AUC needs both classes
            res["roc_auc"] = float(roc_auc_score(y, p_cg))
            res["pr_auc"] = float(average_precision_score(y, p_cg))
    else:
        # unlabeled -> assume genuine (FPR probe): how many real reviews flagged CG
        res["interpretation"] = "all-genuine assumption: flag_rate_CG == false-positive rate"

    (OUTDIR / f"{args.name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
