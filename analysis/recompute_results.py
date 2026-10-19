"""Recompute the reported statistics from the per-run outputs in this repository.

All values are derived from saved per-example probability files, run configs,
and log histories. Nothing is copied from
pre-aggregated summary files except the row-level directional/LOGO matrix
(cross_gen_results.json), whose per-example cross-evaluation probabilities were
not retained, and the row-level source-overlap audit (source_overlap.json).

Usage:
    python analysis/recompute_results.py --out analysis/numbers.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import re
import zipfile
from statistics import mean, stdev

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

THRESHOLD = 0.5
SEEDS = (1998, 7, 42)
PREVALENCES = (0.001, 0.005, 0.01, 0.05, 0.1)


# ----------------------------------------------------------------- helpers
def read_probs(fh):
    rows = list(csv.DictReader(fh))
    y = np.array([1 if r["true"] == "CG" else 0 for r in rows])
    p = np.array([float(r["p_CG"]) for r in rows])
    return rows, y, p


def load_probs(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return read_probs(fh)


def binary_metrics(y, p, thr=THRESHOLD):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    rec_cg = tp / (tp + fn) if tp + fn else float("nan")
    rec_or = tn / (tn + fp) if tn + fp else float("nan")
    prec_cg = tp / (tp + fp) if tp + fp else float("nan")
    prec_or = tn / (tn + fn) if tn + fn else float("nan")
    f1_cg = 2 * prec_cg * rec_cg / (prec_cg + rec_cg) if prec_cg + rec_cg else 0.0
    f1_or = 2 * prec_or * rec_or / (prec_or + rec_or) if prec_or + rec_or else 0.0
    out = dict(n=n, TP=tp, TN=tn, FP=fp, FN=fn, accuracy=(tp + tn) / n,
               recall_CG=rec_cg, recall_OR=rec_or, fpr_OR=1 - rec_or,
               precision_CG=prec_cg, f1_CG=f1_cg, f1_OR=f1_or,
               macro_f1=(f1_cg + f1_or) / 2)
    if len(set(y.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    return out


def agg(values):
    # undefined values (e.g. precision with no alerts) are skipped
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return None
    return dict(n=len(vals), mean=mean(vals), sd=stdev(vals) if len(vals) > 1 else 0.0,
                min=min(vals), max=max(vals), values=vals)


def agg_metrics(list_of_dicts, keys):
    return {k: agg([d.get(k) for d in list_of_dicts]) for k in keys}


def seed_of(run_id):
    m = re.search(r"_s(\d+)_", run_id)
    return int(m.group(1)) if m else None


def executed_steps(log_history):
    return max(int(x["step"]) for x in log_history if "step" in x)


def eval_curve(log_history):
    pts = [(int(x["step"]), x["eval_accuracy"], x["eval_loss"]) for x in log_history
           if "eval_accuracy" in x and "eval_loss" in x]
    return sorted(pts)


# --------------------------------------------------------- row-level (R)
def row_level(root):
    res = {}
    rdir = os.path.join(root, "results")
    gens = {"gpt2": "xg", "lfm": "xl", "deepseek": "xd", "glm": "xglm"}
    keys = ["accuracy", "macro_f1", "roc_auc", "ap", "recall_CG", "recall_OR", "fpr_OR"]
    matched = {}
    for g, tag in gens.items():
        runs = sorted(glob.glob(os.path.join(rdir, f"{tag}_rs_r64_s*")))
        per = []
        for r in runs:
            _, y, p = load_probs(os.path.join(r, "test_probs.csv"))
            m = binary_metrics(y, p)
            m["run"] = os.path.basename(r)
            per.append(m)
        matched[g] = dict(per_run=per, summary=agg_metrics(per, keys))
    res["matched"] = matched

    cg = json.load(open(os.path.join(rdir, "cross_gen_results.json")))
    res["directional"] = dict(generators=cg["generators"],
                              rs_recall=cg["rs_qlora"]["recall_matrix"],
                              rs_fpr=cg["rs_qlora"]["fpr"],
                              tfidf_recall=cg["tfidf"]["recall_matrix"],
                              tfidf_fpr=cg["tfidf"]["fpr"])
    rm = np.array(cg["rs_qlora"]["recall_matrix"])
    off = [rm[i, j] for i in range(4) for j in range(4) if i != j]
    mod = [rm[i, j] for i in range(1, 4) for j in range(1, 4) if i != j]
    res["directional"]["mean_diag"] = float(np.mean(np.diag(rm)))
    res["directional"]["mean_offdiag_all"] = float(np.mean(off))
    res["directional"]["mean_offdiag_contemporary"] = float(np.mean(mod))
    res["logo"] = dict(rs=cg["logo_rs_qlora"], tfidf=cg["logo_tfidf"])

    # paired DeepSeek scaling comparison (m1 = Rs, m1s = standard)
    def cohort(prefix, exclude=()):
        per = []
        for r in sorted(glob.glob(os.path.join(rdir, f"{prefix}_r64_s*"))):
            if os.path.basename(r) in exclude:
                continue
            _, y, p = load_probs(os.path.join(r, "test_probs.csv"))
            m = binary_metrics(y, p)
            met = json.load(open(os.path.join(r, "metrics.json")))
            lh = json.load(open(os.path.join(r, "log_history.json")))
            m.update(run=os.path.basename(r), seed=seed_of(os.path.basename(r)),
                     best_step=met.get("best_step"), executed_steps=executed_steps(lh),
                     runtime_min=met["train_runtime_s"] / 60, peak_alloc_gib=met.get("peak_alloc_gb"),
                     curve=eval_curve(lh))
            m["exec_steps_per_s"] = m["executed_steps"] / met["train_runtime_s"]
            per.append(m)
        return per

    # The June m1 s1998 run predates the frozen H100 cohort; the July recovery run is canonical.
    rs = cohort("m1_rs", exclude=("m1_rs_r64_s1998_06280252",))
    st = cohort("m1s_standard")
    k2 = keys + ["best_step", "executed_steps", "runtime_min", "peak_alloc_gib", "exec_steps_per_s"]
    res["paired"] = dict(rs=dict(per_run=[{k: v for k, v in d.items() if k != "curve"} for d in rs],
                                 summary=agg_metrics(rs, k2)),
                         standard=dict(per_run=[{k: v for k, v in d.items() if k != "curve"} for d in st],
                                       summary=agg_metrics(st, k2)),
                         curves_s1998=dict(rs=[d["curve"] for d in rs if d["seed"] == 1998][0],
                                           standard=[d["curve"] for d in st if d["seed"] == 1998][0]))

    # error analysis of the canonical seed-1998 DeepSeek Rs run
    rows, y, p = load_probs(os.path.join(rdir, "m1_rs_r64_s1998_07121622", "test_probs.csv"))
    res["errors_canonical"] = error_analysis(rows, y, p)

    # rank x scaling grid on the original GPT-2-era release
    grid = {}
    for scaling in ("standard", "rs"):
        for rank in (8, 16, 32, 64):
            per = []
            for r in sorted(glob.glob(os.path.join(rdir, f"a1_{scaling}_r{rank}_s*"))):
                _, y, p = load_probs(os.path.join(r, "test_probs.csv"))
                m = binary_metrics(y, p)
                met = json.load(open(os.path.join(r, "metrics.json")))
                m.update(run=os.path.basename(r), peak_alloc_gib=met.get("peak_alloc_gb"))
                per.append(m)
            grid[f"{scaling}_r{rank}"] = agg_metrics(per, ["accuracy", "peak_alloc_gib"])
    res["rank_grid_gpt2"] = grid

    # lexical/transformer baselines on the row-level DeepSeek partition
    base = {}
    with open(os.path.join(rdir, "baselines.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["model"] in ("tfidf_logreg_ds", "tfidf_linsvm_ds") and r["seed"] == "1998":
                base[r["model"]] = {k: float(r[k]) for k in ("test_acc", "test_roc_auc", "test_pr_auc")}
    with open(os.path.join(rdir, "baselines_transformer.csv")) as fh:
        for r in csv.DictReader(fh):
            base[r["model"]] = {k: float(r[k]) for k in ("test_acc", "test_roc_auc", "test_pr_auc")}
    res["baselines"] = base

    ov = json.load(open(os.path.join(rdir, "source_overlap.json")))
    res["overlap"] = dict(pairwise=[(d["train_generator"], d["test_generator"], d["fraction"])
                                    for d in ov["pairwise_actual_train_source_overlap"]],
                          logo=[(d["test_generator"], d["direct_cg_source_overlap"]["fraction"],
                                 d["exact_grounding_or_text_in_train"]["fraction"],
                                 d["support_partition"]["either"]["fraction"])
                                for d in ov["per_target_actual_logo_training_support"]])
    return res


# --------------------------------------------------- source-grouped (S)
def length_bin(words):
    for lo, hi, name in ((0, 7, "<8"), (8, 14, "8-14"), (15, 24, "15-24"), (25, 49, "25-49"),
                         (50, 99, "50-99")):
        if lo <= words <= hi:
            return name
    return "100+"


def error_analysis(rows, y, p):
    pred = (p >= THRESHOLD).astype(int)
    conf = np.maximum(p, 1 - p)
    err = pred != y
    bins = {}
    cats = {}
    for i, r in enumerate(rows):
        w = len(r["text"].split())
        b = length_bin(w)
        bins.setdefault(b, [0, 0])
        bins[b][0] += 1
        bins[b][1] += int(err[i])
        c = r["category"]
        cats.setdefault(c, [0, 0])
        cats[c][0] += 1
        cats[c][1] += int(err[i])
    return dict(n=len(y), errors=int(err.sum()), FP=int((err & (y == 0)).sum()),
                FN=int((err & (y == 1)).sum()),
                accuracy=float(1 - err.mean()),
                mean_conf_correct=float(conf[~err].mean()), mean_conf_error=float(conf[err].mean()),
                errors_conf_ge_090=int((conf[err] >= 0.90).sum()),
                length_bins={k: dict(n=v[0], errors=v[1], rate=v[1] / v[0] if v[0] else None)
                             for k, v in bins.items()},
                categories={k: dict(n=v[0], errors=v[1], rate=v[1] / v[0]) for k, v in cats.items()})


def prevalence_from_rates(tpr, fpr, pi):
    denom = pi * tpr + (1 - pi) * fpr
    return pi * tpr / denom if denom > 0 else float("nan")


def source_grouped(root):
    rdir = os.path.join(root, "results")
    gens = ("lfm", "deepseek", "glm")
    tags = {"lfm": "sgxl", "deepseek": "sgxd", "glm": "sgxglm"}
    keys = ["accuracy", "macro_f1", "recall_CG", "recall_OR", "fpr_OR", "precision_CG", "roc_auc", "ap"]
    res = {}

    directional = {}
    raw = {}
    for tg in gens:
        runs = sorted(glob.glob(os.path.join(rdir, f"{tags[tg]}_rs_r64_s*")))
        assert len(runs) == 3, runs
        for ev in gens:
            per = []
            for r in runs:
                rows, y, p = load_probs(os.path.join(r, f"source_grouped_eval_{ev}.csv"))
                m = binary_metrics(y, p)
                m.update(run=os.path.basename(r), seed=seed_of(os.path.basename(r)))
                per.append(m)
                raw[(tg, ev, m["seed"])] = (y, p)
            directional[f"{tg}->{ev}"] = dict(per_run=per, summary=agg_metrics(per, keys))
    res["directional"] = directional
    off = [directional[f"{a}->{b}"]["summary"]["recall_CG"]["mean"] for a in gens for b in gens if a != b]
    res["directional_mean_offdiag_recall"] = mean(off)
    res["directional_mean_diag_recall"] = mean(directional[f"{g}->{g}"]["summary"]["recall_CG"]["mean"]
                                               for g in gens)
    # per-seed off-diagonal mean
    res["directional_offdiag_by_seed"] = {
        s: mean(next(d["recall_CG"] for d in directional[f"{a}->{b}"]["per_run"] if d["seed"] == s)
                for a in gens for b in gens if a != b) for s in SEEDS}

    # matched source-grouped training metadata (resources)
    meta = {}
    for tg in gens:
        per = []
        for r in sorted(glob.glob(os.path.join(rdir, f"{tags[tg]}_rs_r64_s*"))):
            met = json.load(open(os.path.join(r, "metrics.json")))
            lh = json.load(open(os.path.join(r, "log_history.json")))
            per.append(dict(best_step=met["best_step"], executed_steps=executed_steps(lh),
                            runtime_min=met["train_runtime_s"] / 60, peak_alloc_gib=met["peak_alloc_gb"]))
        meta[tg] = agg_metrics(per, ["best_step", "executed_steps", "runtime_min", "peak_alloc_gib"])
    res["matched_resources"] = meta

    logo = {}
    logo_raw = {}
    logo_val = {}
    for tg in gens:
        runs = sorted(glob.glob(os.path.join(rdir, f"sgpwo{tg}_rs_r64_s*")))
        assert len(runs) == 3, runs
        per = []
        for r in runs:
            rows, y, p = load_probs(os.path.join(r, f"source_grouped_eval_{tg}.csv"))
            m = binary_metrics(y, p)
            m.update(run=os.path.basename(r), seed=seed_of(os.path.basename(r)))
            met = json.load(open(os.path.join(r, "metrics.json")))
            lh = json.load(open(os.path.join(r, "log_history.json")))
            m.update(best_step=met["best_step"], executed_steps=executed_steps(lh),
                     runtime_min=met["train_runtime_s"] / 60, peak_alloc_gib=met["peak_alloc_gb"])
            per.append(m)
            logo_raw[(tg, m["seed"])] = (y, p)
            _, yv, pv = load_probs(os.path.join(r, "val_probs.csv"))
            logo_val[(tg, m["seed"])] = (yv, pv)
        logo[tg] = dict(per_run=per, summary=agg_metrics(per, keys + ["best_step", "executed_steps",
                                                                      "runtime_min", "peak_alloc_gib"]))
        # best single-generator transfer into this target
        singles = {a: directional[f"{a}->{tg}"]["summary"]["recall_CG"]["mean"] for a in gens if a != tg}
        logo[tg]["best_single_source"] = max(singles.items(), key=lambda kv: kv[1])
        logo[tg]["single_sources"] = singles
    res["logo"] = logo

    # fixed-scale rank series on source-grouped DeepSeek
    fr = {}
    for rank, alpha in ((16, 64), (36, 96), (64, 128)):
        per = []
        for r in sorted(glob.glob(os.path.join(rdir, f"sgfr_rs_r{rank}_s*"))):
            cfg = json.load(open(os.path.join(r, "config.json")))
            assert cfg["alpha"] == alpha and cfg["rank"] == rank
            rows, y, p = load_probs(os.path.join(r, "source_grouped_eval_deepseek.csv"))
            m = binary_metrics(y, p)
            met = json.load(open(os.path.join(r, "metrics.json")))
            lh = json.load(open(os.path.join(r, "log_history.json")))
            m.update(run=os.path.basename(r), seed=seed_of(os.path.basename(r)),
                     best_step=met["best_step"], executed_steps=executed_steps(lh),
                     runtime_min=met["train_runtime_s"] / 60, peak_alloc_gib=met["peak_alloc_gb"])
            per.append(m)
        fr[f"r{rank}_a{alpha}"] = dict(rank=rank, alpha=alpha, scale_rs=alpha / rank ** 0.5,
                                       scale_std=alpha / rank,
                                       trainable=rank * 549_888 + 1_792,
                                       per_run=per,
                                       summary=agg_metrics(per, keys + ["best_step", "executed_steps",
                                                                        "runtime_min", "peak_alloc_gib"]))
    res["fixed_scale_rank"] = fr

    # prevalence-adjusted precision (matched and held-out generator detectors)
    prev = {}
    for kind, src in (("logo", logo_raw), ("matched", {(g, s): raw[(g, g, s)] for g in gens for s in SEEDS})):
        for tg in gens:
            for pi in PREVALENCES:
                vals, alerts, tpr_l, fpr_l = [], [], [], []
                for s in SEEDS:
                    y, p = src[(tg, s)]
                    m = binary_metrics(y, p)
                    vals.append(prevalence_from_rates(m["recall_CG"], m["fpr_OR"], pi))
                    alerts.append(10000 * (pi * m["recall_CG"] + (1 - pi) * m["fpr_OR"]))
                    tpr_l.append(m["recall_CG"])
                    fpr_l.append(m["fpr_OR"])
                prev[f"{kind}|{tg}|{pi}"] = dict(precision=agg(vals), alerts_per_10k=agg(alerts),
                                                 false_alerts_per_10k=agg([10000 * (1 - pi) * f for f in fpr_l]))
    res["prevalence_t05"] = prev

    # operating points along the prevalence-adjusted PR curve (held-out detectors)
    ops = {}
    curves = {}
    for tg in gens:
        for s in SEEDS:
            y, p = logo_raw[(tg, s)]
            fpr, tpr, thr = roc_curve(y, p)
            curves[(tg, s)] = (fpr, tpr, thr)
        for pi in (0.01, 0.05):
            for target in (0.8, 0.9):
                rec = []
                for s in SEEDS:
                    fpr, tpr, thr = curves[(tg, s)]
                    prec = np.array([prevalence_from_rates(t, f, pi) for t, f in zip(tpr, fpr)])
                    ok = np.where(prec >= target)[0]
                    rec.append(float(tpr[ok].max()) if len(ok) else 0.0)
                ops[f"{tg}|{pi}|{target}"] = agg(rec)
    res["prevalence_operating_points_test_oracle"] = ops

    # validation-calibrated operating point: threshold chosen on the in-distribution (pooled) validation split
    calib = {}
    for tg in gens:
        for pi, target in ((0.01, 0.9), (0.05, 0.9)):
            out = []
            for s in SEEDS:
                yv, pv = logo_val[(tg, s)]
                fv, tv, thv = roc_curve(yv, pv)
                precv = np.array([prevalence_from_rates(t, f, pi) for t, f in zip(tv, fv)])
                ok = np.where(precv >= target)[0]
                if not len(ok):
                    out.append(None)
                    continue
                # most permissive threshold meeting the validation precision target
                t_star = float(thv[ok[np.argmax(tv[ok])]])
                y, p = logo_raw[(tg, s)]
                m = binary_metrics(y, p, thr=t_star)
                out.append(dict(seed=s, threshold=t_star, recall=m["recall_CG"], fpr=m["fpr_OR"],
                                precision=prevalence_from_rates(m["recall_CG"], m["fpr_OR"], pi)))
            calib[f"{tg}|{pi}|{target}"] = out
    res["prevalence_val_calibrated"] = calib
    res["_curves"] = curves
    res["_logo_raw"] = logo_raw
    res["_directional_raw"] = raw
    return res


# --------------------------------------------- follow-up DeepSeek rebuild
def followup(root):
    by_seed = os.path.join(root, "results", "source_grouped_followup_summary", "by_seed.csv")
    with open(by_seed, newline="") as fh:
        index = list(csv.DictReader(fh))
    jobs = {}
    for row in index:
        jid = row["job_id"]
        job_dir = os.path.join(root, "results", "source_grouped_followup", jid)
        if os.path.isdir(job_dir):
            # per-job outputs extracted from the job archives (repository layout)
            rows, y, p = load_probs(os.path.join(job_dir, "test_probs.csv"))
            met = json.load(open(os.path.join(job_dir, "metrics.json")))
            lh_path = os.path.join(job_dir, "log_history.json")
            lh = json.load(open(lh_path)) if os.path.exists(lh_path) else None
            env_path = os.path.join(job_dir, "runtime_environment.json")
            env = json.load(open(env_path)) if os.path.exists(env_path) else {}
        else:
            with zipfile.ZipFile(row["archive"]) as z:
                names = z.namelist()
                base = [n for n in names if n.endswith(f"source_grouped_rebuild/{jid}/test_probs.csv")]
                assert base, (jid, names[:10])
                d = base[0].rsplit("/", 1)[0]
                with z.open(f"{d}/test_probs.csv") as fh:
                    rows, y, p = read_probs(io.TextIOWrapper(fh, encoding="utf-8"))
                met = json.loads(z.read(f"{d}/metrics.json"))
                lh = json.loads(z.read(f"{d}/log_history.json")) if f"{d}/log_history.json" in names else None
                env = (json.loads(z.read(f"{d}/runtime_environment.json"))
                       if f"{d}/runtime_environment.json" in names else {})
        m = binary_metrics(y, p)
        m.update(job=jid, seed=int(row["seed"]), kind=row["kind"], condition=row["condition"],
                 evaluation=row["evaluation"], target=row["target"], gpu=env.get("gpu", ""),
                 training_instance=row["training_instance_id"], ledger_accuracy=float(row["accuracy"]),
                 ledger_runtime_min=float(row["runtime_minutes"]) if row["runtime_minutes"] else None,
                 ledger_executed_steps=float(row["executed_steps"]) if row["executed_steps"] else None,
                 ledger_best_step=float(row["best_step"]) if row["best_step"] else None,
                 ledger_peak_alloc=float(row["peak_allocated_gib"]) if row["peak_allocated_gib"] else None,
                 trainable=float(row["trainable_parameters"]) if row["trainable_parameters"] else None)
        if lh:
            m["executed_steps"] = executed_steps(lh)
            m["curve"] = eval_curve(lh)
        res_ = met.get("resources", {})
        if "train_runtime_s" in res_:
            m["runtime_min"] = res_["train_runtime_s"] / 60
        m["best_step"] = res_.get("best_step")
        m["peak_alloc_gib"] = res_.get("peak_alloc_gb", res_.get("peak_allocated_gib"))
        if m.get("executed_steps") and m.get("runtime_min"):
            m["exec_steps_per_s"] = m["executed_steps"] / (m["runtime_min"] * 60)
        m["_rows"], m["_y"], m["_p"] = rows, y, p
        jobs[jid] = m
    groups = {}
    for jid, m in jobs.items():
        g = re.sub(r"_s\d+$", "", jid)
        groups.setdefault(g, []).append(m)
    keys = ["accuracy", "macro_f1", "recall_CG", "recall_OR", "fpr_OR", "roc_auc", "ap",
            "best_step", "executed_steps", "runtime_min", "peak_alloc_gib", "exec_steps_per_s", "trainable"]
    summ = {g: agg_metrics(v, keys) for g, v in groups.items()}
    # paired Rs vs standard per seed (same host per seed)
    pairs = []
    for s in SEEDS:
        a, b = jobs[f"pair_rs_s{s}"], jobs[f"pair_standard_s{s}"]
        pairs.append(dict(seed=s, host_rs=a["training_instance"], host_std=b["training_instance"],
                          acc_diff_pp=100 * (a["accuracy"] - b["accuracy"]),
                          runtime_rs=a["runtime_min"], runtime_std=b["runtime_min"],
                          runtime_reduction_pct=100 * (1 - a["runtime_min"] / b["runtime_min"]),
                          steps_rs=a["executed_steps"], steps_std=b["executed_steps"],
                          step_reduction_pct=100 * (1 - a["executed_steps"] / b["executed_steps"]),
                          best_rs=a["best_step"], best_std=b["best_step"],
                          sps_rs=a["exec_steps_per_s"], sps_std=b["exec_steps_per_s"]))
    # pooled error analysis over the three Rs paired runs
    errs = {}
    for s in SEEDS:
        m = jobs[f"pair_rs_s{s}"]
        errs[s] = error_analysis(m["_rows"], m["_y"], m["_p"])
    pooled_rows = sum((jobs[f"pair_rs_s{s}"]["_rows"] for s in SEEDS), [])
    pooled_y = np.concatenate([jobs[f"pair_rs_s{s}"]["_y"] for s in SEEDS])
    pooled_p = np.concatenate([jobs[f"pair_rs_s{s}"]["_p"] for s in SEEDS])
    errs["pooled"] = error_analysis(pooled_rows, pooled_y, pooled_p)
    # errors shared across seeds
    err_sets = []
    for s in SEEDS:
        m = jobs[f"pair_rs_s{s}"]
        pred = (m["_p"] >= THRESHOLD).astype(int)
        err_sets.append({(r["source_id"], r["true"]) for r, e in zip(m["_rows"], pred != m["_y"]) if e})
    errs["shared_all_three"] = len(err_sets[0] & err_sets[1] & err_sets[2])
    errs["union"] = len(err_sets[0] | err_sets[1] | err_sets[2])
    return jobs, summ, pairs, errs


# ------------------------------------------- external corpora (zero-shot)
EXTERNAL = (("maideup_en", "MAiDE-up"), ("detectairev", "DetectAIRev"), ("mage_yelp", "MAGE-Yelp"))
DETECTORS = (("sgxl", "matched", "lfm"), ("sgxd", "matched", "deepseek"), ("sgxglm", "matched", "glm"),
             ("sgpwolfm", "pooled", "lfm"), ("sgpwodeepseek", "pooled", "deepseek"),
             ("sgpwoglm", "pooled", "glm"))


def recall_at_fpr(y, p, max_fpr=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def external(root):
    edir = os.path.join(root, "results", "external_benchmark")
    out = {}
    for ds, _ in EXTERNAL:
        with open(os.path.join(edir, f"{ds}_probs.csv"), newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        y = np.array([1 if r["label"] == "CG" else 0 for r in rows])
        dom = np.array([r.get("domain", "") for r in rows])
        gen = np.array([r.get("generator", "") for r in rows])
        for tag, kind, target in DETECTORS:
            per = []
            for s in SEEDS:
                p = np.array([float(r[f"p_CG__{tag}_s{s}"]) for r in rows])
                m = binary_metrics(y, p)
                m["seed"] = s
                m["recall_at_1pct_fpr"] = recall_at_fpr(y, p)
                m["ppv_1pct"] = prevalence_from_rates(m["recall_CG"], m["fpr_OR"], 0.01)
                # per-domain FPR and per-generator recall / AUC
                m["fpr_by_domain"] = {d: float((p[(dom == d) & (y == 0)] >= THRESHOLD).mean())
                                      for d in sorted(set(dom[y == 0]))}
                m["recall_by_generator"] = {g: float((p[gen == g] >= THRESHOLD).mean())
                                            for g in sorted(set(gen[y == 1]))}
                m["auc_by_generator"] = {g: float(roc_auc_score(np.r_[np.zeros((y == 0).sum()), np.ones((gen == g).sum())],
                                                                np.r_[p[y == 0], p[gen == g]]))
                                         for g in sorted(set(gen[y == 1]))}
                per.append(m)
            summ = agg_metrics(per, ["recall_CG", "fpr_OR", "roc_auc", "ap", "recall_at_1pct_fpr", "ppv_1pct",
                                     "accuracy"])
            summ["fpr_by_domain"] = {d: mean(m["fpr_by_domain"][d] for m in per) for d in per[0]["fpr_by_domain"]}
            summ["recall_by_generator"] = {g: mean(m["recall_by_generator"][g] for m in per)
                                           for g in per[0]["recall_by_generator"]}
            summ["auc_by_generator"] = {g: mean(m["auc_by_generator"][g] for m in per)
                                        for g in per[0]["auc_by_generator"]}
            out[f"{tag}|{ds}"] = dict(kind=kind, target=target, n_OR=int((y == 0).sum()), n_CG=int(y.sum()),
                                      summary=summ)
    return out


def strip(obj):
    if isinstance(obj, dict):
        return {str(k): strip(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [strip(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers.json"))
    args = ap.parse_args()
    R = row_level(args.root)
    S = source_grouped(args.root)
    jobs, summ, pairs, errs = followup(args.root)
    X = external(args.root)
    out = dict(row_level=R, source_grouped=S, external=X,
               followup=dict(jobs={k: {kk: vv for kk, vv in v.items() if not kk.startswith("_") and kk != "curve"}
                                   for k, v in jobs.items()},
                             summary=summ, pairs=pairs, errors=errs))
    with open(args.out, "w") as fh:
        json.dump(strip(out), fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
