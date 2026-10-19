"""Zero-shot evaluation of the source-grouped Rs-QLoRA adapters on independently
collected review corpora. Nothing is retrained.

Adapters: matched sgx{l,d,glm} and pooled Leave-One-Generator-Out
sgpwo{lfm,deepseek,glm}, rank 64, alpha 128, seeds 1998/7/42, each loaded on the
frozen unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit checkpoint exactly as in
runs/evaluate_source_grouped.py (max length 512, softmax P(CG), threshold 0.5).

Inputs come from runs/prepare_external_benchmark.py. Before scoring external
text, each adapter re-scores 512 rows of its own stored source-grouped test
probabilities so that any loading difference is visible in the summary.

Outputs (results/external_benchmark/):
  probs/<run_id>.csv           per-run cache: dataset,row_id,p_CG
  <dataset>_probs.csv          per-example probabilities, one column per run
  by_run.csv                   metrics per run x dataset x slice
  summary.json                 mean +- SD over seeds per detector, plus provenance

  python runs/evaluate_external_benchmark.py --root .
  python runs/evaluate_external_benchmark.py --root ... --summarize-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

DATASETS = ("maideup_en", "detectairev", "mage_yelp")
DETECTORS = {
    # tag: (protocol, training generator(s), in-distribution target)
    "sgxl": ("matched", "LFM2.5", "lfm"),
    "sgxd": ("matched", "DeepSeek V4", "deepseek"),
    "sgxglm": ("matched", "GLM-5.2", "glm"),
    "sgpwolfm": ("pooled", "DeepSeek V4 + GLM-5.2", "lfm"),
    "sgpwodeepseek": ("pooled", "LFM2.5 + GLM-5.2", "deepseek"),
    "sgpwoglm": ("pooled", "LFM2.5 + DeepSeek V4", "glm"),
}
SEEDS = (1998, 7, 42)
BASE = "unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit"
METRIC_KEYS = ("recall_CG", "fpr_OR", "roc_auc", "ap", "recall_CG_at_fpr_1pct", "accuracy",
               "precision_CG", "f1_CG")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_runs(results: Path) -> list[dict]:
    runs = []
    for tag in DETECTORS:
        for seed in SEEDS:
            cands = [d for d in results.glob(f"{tag}_rs_r64_s{seed}_*") if (d / "adapter").is_dir()]
            if len(cands) != 1:
                raise RuntimeError(f"expected one completed run for {tag} s{seed}, found {cands}")
            run = cands[0]
            cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
            acfg = json.loads((run / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
            checks = {
                "rank": (cfg["rank"], 64), "alpha": (cfg["alpha"], 128),
                "scaling": (cfg["scaling"], "rs"), "seed": (cfg["seed"], seed),
                "protocol": (cfg.get("protocol"), "source-grouped-contemporary-v1"),
                "base": (acfg["base_model_name_or_path"], BASE),
                "adapter_r": (acfg["r"], 64), "adapter_alpha": (acfg["lora_alpha"], 128),
            }
            bad = {k: v for k, v in checks.items() if v[0] != v[1]}
            if bad:
                raise RuntimeError(f"{run.name}: unexpected config {bad}")
            runs.append({"tag": tag, "seed": seed, "run_id": run.name, "dir": run,
                         "training_data": cfg["data"]})
    return runs


def metrics(labels: list[int], probs: list[float]) -> dict:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(labels)
    p = np.asarray(probs)
    pred = (p >= 0.5).astype(int)
    pos, neg = y == 1, y == 0
    tp = int((pred[pos] == 1).sum())
    fp = int((pred[neg] == 1).sum())
    out = {"n": int(len(y)), "n_OR": int(neg.sum()), "n_CG": int(pos.sum())}
    out["recall_CG"] = tp / pos.sum() if pos.any() else None
    out["fpr_OR"] = fp / neg.sum() if neg.any() else None
    out["accuracy"] = float((pred == y).mean())
    out["precision_CG"] = tp / (tp + fp) if tp + fp else 0.0
    rec = out["recall_CG"] or 0.0
    out["f1_CG"] = 2 * out["precision_CG"] * rec / (out["precision_CG"] + rec) if out["precision_CG"] + rec else 0.0
    both = pos.any() and neg.any()
    out["roc_auc"] = float(roc_auc_score(y, p)) if both else None
    out["ap"] = float(average_precision_score(y, p)) if both else None
    # Oracle operating point: CG recall at the largest threshold whose OR FPR on
    # this external set is <= 1% (chosen with test labels; an optimistic bound).
    if both:
        thr = np.sort(np.unique(p))[::-1]
        neg_p, pos_p = np.sort(p[neg]), np.sort(p[pos])
        fpr = 1 - np.searchsorted(neg_p, thr, side="left") / len(neg_p)
        tpr = 1 - np.searchsorted(pos_p, thr, side="left") / len(pos_p)
        ok = fpr <= 0.01
        out["recall_CG_at_fpr_1pct"] = float(tpr[ok].max()) if ok.any() else 0.0
    else:
        out["recall_CG_at_fpr_1pct"] = None
    out["mean_p_CG_OR"] = float(p[neg].mean()) if neg.any() else None
    out["mean_p_CG_CG"] = float(p[pos].mean()) if pos.any() else None
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def slices(rows: list[dict]) -> dict[str, list[int]]:
    """Index sets: whole dataset, each domain, and each generator (that
    generator's CG rows plus the OR rows of the domains it appears in)."""
    out = {"all": list(range(len(rows)))}
    domains = sorted({r["domain"] for r in rows})
    if len(domains) > 1:
        for d in domains:
            out[f"domain={d}"] = [i for i, r in enumerate(rows) if r["domain"] == d]
    gens = sorted({r["generator"] for r in rows if r["label"] == "CG"})
    if len(gens) > 1:
        for g in gens:
            gdom = {r["domain"] for r in rows if r["generator"] == g}
            out[f"generator={g}"] = [i for i, r in enumerate(rows)
                                     if r["generator"] == g or (r["label"] == "OR" and r["domain"] in gdom)]
    return out


def score_run(run: dict, data: dict[str, list[dict]], batch_size: int) -> tuple[dict[str, list[float]], dict]:
    import pyarrow  # noqa: F401  must precede unsloth/torch on Windows
    import torch
    from unsloth import FastModel
    from transformers import AutoModelForSequenceClassification

    model, tokenizer = FastModel.from_pretrained(
        model_name=str(run["dir"] / "adapter"),
        auto_model=AutoModelForSequenceClassification,
        max_seq_length=512,
        dtype=None,
        num_labels=2,
        load_in_4bit=True,
        full_finetuning=False,
    )
    FastModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    def predict(texts: list[str]) -> list[float]:
        # Length-sorted batches cut padding; results are returned in input order.
        lengths = [len(ids) for ids in tokenizer(texts, truncation=True, max_length=512)["input_ids"]]
        order = sorted(range(len(texts)), key=lambda i: -lengths[i])
        out = [0.0] * len(texts)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=512,
                            padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().tolist()
            for i, v in zip(idx, p):
                out[i] = float(v)
        return out

    # Reproduction check against this adapter's stored in-distribution probabilities.
    target = DETECTORS[run["tag"]][2]
    stored = read_csv(run["dir"] / f"source_grouped_eval_{target}.csv")
    check_rows = stored[:256] + stored[-256:]
    fresh = predict([r["text"] for r in check_rows])
    diffs = [abs(a - float(r["p_CG"])) for a, r in zip(fresh, check_rows)]
    flips = sum((a >= 0.5) != (float(r["p_CG"]) >= 0.5) for a, r in zip(fresh, check_rows))
    check = {"target": target, "n": len(check_rows), "max_abs_diff": max(diffs),
             "mean_abs_diff": sum(diffs) / len(diffs), "label_flips": flips}

    probs = {name: predict([r["text_"] for r in rows]) for name, rows in data.items()}
    del model
    torch.cuda.empty_cache()
    return probs, check


def in_distribution(run: dict) -> dict:
    ev = json.loads((run["dir"] / "source_grouped_eval.json").read_text(encoding="utf-8"))
    m = ev["targets"][DETECTORS[run["tag"]][2]]["metrics"]
    return {"recall_CG": m["recall_CG"], "fpr_OR": m["fpr_OR"], "roc_auc": m["roc_auc"], "ap": m["pr_auc"]}


def mean_sd(values: list[float | None]) -> dict | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "values": vals}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    ext = root / "dataset" / "external_benchmark"
    out = root / "results" / "external_benchmark"
    (out / "probs").mkdir(parents=True, exist_ok=True)

    data = {name: read_csv(ext / f"{name}.csv") for name in DATASETS}
    data_sha = {name: sha256(ext / f"{name}.csv") for name in DATASETS}
    runs = find_runs(root / "results")
    checks_path = out / "reproduction_checks.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8")) if checks_path.exists() else {}

    probs: dict[str, dict[str, list[float]]] = {}
    for run in runs:
        cache = out / "probs" / f"{run['run_id']}.csv"
        meta = out / "probs" / f"{run['run_id']}.json"
        if cache.exists() and meta.exists() and json.loads(meta.read_text())["data_sha256"] == data_sha:
            loaded: dict[str, dict[int, float]] = {n: {} for n in DATASETS}
            for r in read_csv(cache):
                loaded[r["dataset"]][int(r["row_id"])] = float(r["p_CG"])
            probs[run["run_id"]] = {n: [loaded[n][i] for i in range(len(data[n]))] for n in DATASETS}
            continue
        if args.summarize_only:
            raise RuntimeError(f"no cached probabilities for {run['run_id']}")
        p, check = score_run(run, data, args.batch_size)
        with cache.open("w", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["dataset", "row_id", "p_CG"])
            for name in DATASETS:
                for i, v in enumerate(p[name]):
                    w.writerow([name, data[name][i]["row_id"], f"{v:.9f}"])
        meta.write_text(json.dumps({"data_sha256": data_sha}, indent=2), encoding="utf-8")
        checks[run["run_id"]] = check
        checks_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
        probs[run["run_id"]] = p
        print(f"scored {run['run_id']}: check max|dp|={check['max_abs_diff']:.4f} "
              f"flips={check['label_flips']}", flush=True)

    # Per-example probabilities, one column per run.
    for name in DATASETS:
        with (out / f"{name}_probs.csv").open("w", encoding="utf-8", newline="") as handle:
            cols = [f"p_CG__{r['tag']}_s{r['seed']}" for r in runs]
            w = csv.writer(handle)
            w.writerow(["row_id", "dataset", "domain", "generator", "label", *cols, "text_"])
            for i, row in enumerate(data[name]):
                w.writerow([row["row_id"], name, row["domain"], row["generator"], row["label"],
                            *[f"{probs[r['run_id']][name][i]:.9f}" for r in runs], row["text_"]])

    # Metrics per run and slice.
    by_run = []
    for run in runs:
        for name in DATASETS:
            rows = data[name]
            labels = [1 if r["label"] == "CG" else 0 for r in rows]
            for sl, idx in slices(rows).items():
                m = metrics([labels[i] for i in idx], [probs[run["run_id"]][name][i] for i in idx])
                by_run.append({"run_id": run["run_id"], "tag": run["tag"], "seed": run["seed"],
                               "dataset": name, "slice": sl, **m})
    fields = ["run_id", "tag", "seed", "dataset", "slice", "n", "n_OR", "n_CG", *METRIC_KEYS,
              "mean_p_CG_OR", "mean_p_CG_CG"]
    with (out / "by_run.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(by_run)

    summary: dict = {
        "schema_version": 1,
        "description": "Zero-shot transfer of source-grouped Rs-QLoRA r64 adapters to external "
                       "human-vs-LLM review corpora; threshold 0.5; mean and sample SD over "
                       "training seeds 1998, 7, 42.",
        "base_checkpoint": BASE,
        "threshold": 0.5,
        "seeds": list(SEEDS),
        "datasets": {n: {"file": f"dataset/external_benchmark/{n}.csv", "sha256": data_sha[n],
                         "n_OR": sum(r["label"] == "OR" for r in data[n]),
                         "n_CG": sum(r["label"] == "CG" for r in data[n])} for n in DATASETS},
        "dataset_manifest": json.loads((ext / "manifest.json").read_text(encoding="utf-8")),
        "dataset_sources": json.loads((ext / "sources.json").read_text(encoding="utf-8")),
        "runs": [{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items() if k != "dir"}
                 for r in runs],
        "reproduction_checks": checks,
        "detectors": {},
    }
    for tag, (protocol, trained_on, target) in DETECTORS.items():
        tag_runs = [r for r in runs if r["tag"] == tag]
        entry = {"protocol": protocol, "trained_on": trained_on,
                 "in_distribution_target": target,
                 "in_distribution": {k: mean_sd([in_distribution(r)[k] for r in tag_runs])
                                     for k in ("recall_CG", "fpr_OR", "roc_auc", "ap")},
                 "external": {}}
        for name in DATASETS:
            entry["external"][name] = {}
            for sl in slices(data[name]):
                recs = [b for b in by_run if b["tag"] == tag and b["dataset"] == name and b["slice"] == sl]
                entry["external"][name][sl] = {
                    "n_OR": recs[0]["n_OR"], "n_CG": recs[0]["n_CG"],
                    **{k: mean_sd([b[k] for b in recs]) for k in METRIC_KEYS},
                }
        summary["detectors"][tag] = entry
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def fmt(x):
        return "n/a" if x is None else f"{100 * x['mean']:.2f} ± {100 * x['sd']:.2f}"

    lines = ["# External benchmark: zero-shot transfer of source-grouped Rs-QLoRA r64 adapters", "",
             "Threshold 0.5; values in %, mean ± SD over training seeds 1998, 7, 42. "
             "In-distribution = the adapter's own source-grouped target (matched generator, or "
             "held-out generator for pooled detectors).", "",
             "| Detector | Trained on | Test set | n OR / CG | CG recall | OR FPR | ROC-AUC | AP |",
             "|---|---|---|---|---|---|---|---|"]
    for tag, entry in summary["detectors"].items():
        ind = entry["in_distribution"]
        lines.append(f"| {tag} | {entry['trained_on']} | in-distribution ({entry['in_distribution_target']}) "
                     f"| 2829 / 2829 | {fmt(ind['recall_CG'])} | {fmt(ind['fpr_OR'])} | "
                     f"{fmt(ind['roc_auc'])} | {fmt(ind['ap'])} |")
        for name in DATASETS:
            s = entry["external"][name]["all"]
            lines.append(f"| {tag} | {entry['trained_on']} | {name} | {s['n_OR']} / {s['n_CG']} | "
                         f"{fmt(s['recall_CG'])} | {fmt(s['fpr_OR'])} | {fmt(s['roc_auc'])} | {fmt(s['ap'])} |")
    for name in DATASETS:
        sls = [sl for sl in slices(data[name]) if sl != "all"]
        if not sls:
            continue
        lines += ["", f"## {name}: slices (CG recall / OR FPR / ROC-AUC)", "",
                  "| Slice | n OR / CG | " + " | ".join(DETECTORS) + " |",
                  "|---|---|" + "---|" * len(DETECTORS)]
        for sl in sls:
            first = summary["detectors"]["sgxd"]["external"][name][sl]
            cells = []
            for tag in DETECTORS:
                s = summary["detectors"][tag]["external"][name][sl]
                parts = [s["recall_CG"], s["fpr_OR"], s["roc_auc"]]
                cells.append(" / ".join("n/a" if p is None else f"{100 * p['mean']:.1f}" for p in parts))
            lines.append(f"| {sl} | {first['n_OR']} / {first['n_CG']} | " + " | ".join(cells) + " |")
    (out / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6 + 4 * len(DETECTORS)]))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
