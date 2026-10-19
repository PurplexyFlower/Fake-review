"""Render result figures from numbers.json and the per-run outputs.

Usage:
    python analysis/make_figures.py --numbers analysis/numbers.json --out analysis/figures
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

# Reference categorical slots (blue, orange, aqua) and chrome.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#898781"
INK, INK2 = "#0b0b0b", "#52514e"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)
SEEDS = (1998, 7, 42)
GEN_LABEL = {"gpt2": "GPT-2 era", "lfm": "LFM2.5", "deepseek": "DeepSeek V4", "glm": "GLM-5.2"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
    "pdf.fonttype": 42,
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    fig.savefig(os.path.join(out, f"{name}.png"))
    plt.close(fig)


def pct(x):
    return 100.0 * x


# ------------------------------------------------------------------ data
def followup_logs(root):
    by_seed = os.path.join(root, "results", "source_grouped_followup_summary", "by_seed.csv")
    logs = {}
    with open(by_seed, newline="") as fh:
        for row in csv.DictReader(fh):
            jid = row["job_id"]
            if not jid.startswith("pair_"):
                continue
            job_dir = os.path.join(root, "results", "source_grouped_followup", jid)
            if os.path.isdir(job_dir):
                logs[jid] = json.load(open(os.path.join(job_dir, "log_history.json")))
                continue
            with zipfile.ZipFile(row["archive"]) as z:
                name = [n for n in z.namelist() if n.endswith(f"source_grouped_rebuild/{jid}/log_history.json")][0]
                logs[jid] = json.loads(z.read(name))
    return logs


def curve(lh):
    pts = sorted((int(x["step"]), x["eval_accuracy"], x["eval_loss"]) for x in lh if "eval_accuracy" in x)
    return np.array(pts)


def load_probs(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return (np.array([r["true"] == "CG" for r in rows], dtype=int),
            np.array([float(r["p_CG"]) for r in rows]))


# --------------------------------------------------------------- figures
def fig_matched(N, out):
    R = N["row_level"]["matched"]
    S = N["source_grouped"]["directional"]
    gens = ["gpt2", "lfm", "deepseek", "glm"]
    x = np.arange(len(gens))
    d = 0.12
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    rm = [pct(R[g]["summary"]["accuracy"]["mean"]) for g in gens]
    rs = [pct(R[g]["summary"]["accuracy"]["sd"]) for g in gens]
    ax.errorbar(x - d, rm, yerr=rs, fmt="o", color=BLUE, ms=6, capsize=3, lw=1.5,
                label="Row-level partition (four domains)")
    sm = [pct(S[f"{g}->{g}"]["summary"]["accuracy"]["mean"]) for g in gens[1:]]
    ss = [pct(S[f"{g}->{g}"]["summary"]["accuracy"]["sd"]) for g in gens[1:]]
    ax.errorbar(x[1:] + d, sm, yerr=ss, fmt="s", color=ORANGE, ms=6, capsize=3, lw=1.5,
                label="Source-grouped partition (contemporary)")
    for i in range(len(gens)):
        ax.annotate(f"{rm[i]:.3f}%", (x[i] - d, rm[i]), textcoords="offset points",
                    xytext=(-8, -3), ha="right", fontsize=7.5, color=INK2)
    for i in range(1, len(gens)):
        ax.annotate(f"{sm[i-1]:.3f}%", (x[i] + d, sm[i-1] - ss[i-1]), textcoords="offset points",
                    xytext=(0, -6), ha="center", va="top", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [GEN_LABEL[g] for g in gens])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(98.0, 100.1)
    ax.set_xlim(-0.8, 3.8)
    ax.legend(loc="lower right")
    save(fig, out, "figure2")


def fig_same_split(N, out):
    F = N["followup"]["summary"]
    rows = [("tfidf_deepseek_logreg", "TF-IDF logistic regression", GRAY),
            ("tfidf_deepseek_linsvm", "TF-IDF LinearSVC", GRAY),
            ("roberta", "RoBERTa-base", AQUA),
            ("deberta", "DeBERTa-v3-base", AQUA),
            ("modernbert", "ModernBERT-large", AQUA),
            ("pair_standard", "Qwen standard QLoRA", ORANGE),
            ("pair_rs", "Qwen Rs-QLoRA", BLUE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(rows))[::-1]
    for yi, (k, lab, c) in zip(y, rows):
        m, s = pct(F[k]["accuracy"]["mean"]), pct(F[k]["accuracy"]["sd"])
        a.errorbar(m, yi, xerr=s if s > 0 else None, fmt="o", color=c, ms=6, capsize=3, lw=1.5)
        a.annotate(f"{m:.3f}", (m + max(s, 0.0), yi), textcoords="offset points", xytext=(5, -3),
                   fontsize=7.5, color=INK2)
    a.set_yticks(y, [r[1] for r in rows])
    a.set_xlabel("Test accuracy (%)")
    a.set_xlim(98.7, 100.0)
    a.set_title("(a) Matched accuracy", loc="left")
    # (b) accuracy against peak allocated training memory (GPU models only)
    for k, lab, c in rows[2:]:
        mem = F[k]["peak_alloc_gib"]["mean"]
        acc = pct(F[k]["accuracy"]["mean"])
        b.scatter(mem, acc, s=46, color=c, zorder=3, edgecolor="white", linewidth=1.2)
        off = {"pair_rs": (6, 4), "pair_standard": (6, -9), "roberta": (6, -9), "deberta": (-6, 6),
               "modernbert": (-6, 6)}[k]
        b.annotate(lab.replace("Qwen s", "S").replace("Qwen ", ""), (mem, acc), textcoords="offset points", xytext=off,
                   ha="left" if off[0] > 0 else "right", fontsize=7.5, color=INK2)
    b.set_xscale("log")
    b.set_xticks([2, 4, 8, 16], ["2", "4", "8", "16"])
    b.set_xlim(1.5, 22)
    b.set_ylim(99.30, 99.70)
    b.set_xlabel("Peak allocated memory (GiB, log scale)")
    b.set_ylabel("Mean test accuracy (%)")
    b.set_title("(b) Accuracy versus training memory", loc="left")
    fig.tight_layout(w_pad=2.5)
    save(fig, out, "figure3")


def heat(ax, M, labels_r, labels_c, sd=None, vmax=100):
    ax.imshow(M, cmap=SEQ_CMAP, vmin=0, vmax=vmax, aspect="equal")
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            col = "white" if v > 55 else INK
            txt = f"{v:.1f}" if sd is None else f"{v:.1f}\n±{sd[i, j]:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
    ax.set_xticks(range(len(labels_c)), labels_c)
    ax.set_yticks(range(len(labels_r)), labels_r)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)


def fig_directional(N, out):
    R = np.array(N["row_level"]["directional"]["rs_recall"]) * 100
    S = N["source_grouped"]["directional"]
    gens = ["lfm", "deepseek", "glm"]
    SM = np.array([[pct(S[f"{a}->{b}"]["summary"]["recall_CG"]["mean"]) for b in gens] for a in gens])
    SS = np.array([[pct(S[f"{a}->{b}"]["summary"]["recall_CG"]["sd"]) for b in gens] for a in gens])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[4, 3.2], wspace=0.45))
    lab4 = ["GPT-2", "LFM2.5", "DeepSeek", "GLM-5.2"]
    heat(a, R, lab4, lab4)
    a.set_title("(a) Row-level partition, four domains", loc="left")
    a.set_xlabel("Evaluation generator")
    a.set_ylabel("Training generator")
    heat(b, SM, lab4[1:], lab4[1:], sd=SS)
    b.set_title("(b) Source-grouped partition", loc="left")
    b.set_xlabel("Evaluation generator")
    sm = plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=[a, b], fraction=0.025, pad=0.02)
    cb.set_label("CG recall (%)")
    cb.outline.set_visible(False)
    save(fig, out, "figure4")


def fig_logo(N, out):
    S = N["source_grouped"]
    gens = ["lfm", "deepseek", "glm"]
    short = {"lfm": "LFM2.5", "deepseek": "DeepSeek", "glm": "GLM-5.2"}
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    w = 0.27
    ticks, labels = [], []
    for i, tg in enumerate(gens):
        others = [g for g in gens if g != tg]
        vals = [(short[o], S["directional"][f"{o}->{tg}"]["summary"]["recall_CG"], "#9ec5f4") for o in others]
        vals.append(("Pooled", S["logo"][tg]["summary"]["recall_CG"], BLUE))
        for k, (lab, d, c) in enumerate(vals):
            xk = i + (k - 1) * w
            m, s_ = pct(d["mean"]), pct(d["sd"])
            ax.bar(xk, m, width=w - 0.03, color=c, yerr=s_, capsize=2.5, error_kw=dict(lw=1, ecolor=INK2))
            ax.text(xk, m + s_ + 1.5, f"{m:.1f}", ha="center", fontsize=7.5, color=INK)
            ticks.append(xk)
            labels.append(lab)
        ax.text(i, -17, f"Held-out {GEN_LABEL[tg]}", ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_xticks(ticks, labels, fontsize=6.6)
    ax.set_ylabel("CG recall on held-out generator (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#9ec5f4", label="Trained on one other generator"),
                       Patch(color=BLUE, label="Pooled training on both other generators")],
              loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
    save(fig, out, "figure5")


def fig_paired(N, out):
    R = N["row_level"]["paired"]
    F = N["followup"]["jobs"]
    groups = []
    for s in SEEDS:
        rs = [d for d in R["rs"]["per_run"] if d["seed"] == s][0]
        st = [d for d in R["standard"]["per_run"] if d["seed"] == s][0]
        groups.append((f"{s}", rs, st))
    for s in SEEDS:
        groups.append((f"{s}", F[f"pair_rs_s{s}"], F[f"pair_standard_s{s}"]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    x = np.arange(len(groups))
    w = 0.38
    for ax, key, lab in ((a, "executed_steps", "Executed optimizer steps"),
                         (b, "runtime_min", "Training wall time (min)")):
        ax.bar(x - w / 2, [g[1][key] for g in groups], width=w - 0.03, color=BLUE, label="Rs-QLoRA")
        ax.bar(x + w / 2, [g[2][key] for g in groups], width=w - 0.03, color=ORANGE, label="Standard QLoRA")
        ax.set_xticks(x, [g[0] for g in groups], fontsize=7.5)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
        ax.axvline(2.5, color=AXIS, lw=0.8, ls="--")
        ymax = ax.get_ylim()[1]
        ax.text(1, -0.13, "Row-level partition", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.2)
        ax.text(4, -0.13, "Source-grouped partition", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=8.2)
    a.set_title("(a) Steps executed before early stopping", loc="left")
    b.set_title("(b) Wall time of the training call", loc="left")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, out, "figure6")


def fig_traj(N, logs, out, which, name, ylab, ylim=None):
    R = N["row_level"]["paired"]["curves_s1998"]
    F = N["followup"]["jobs"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=False)
    panels = [("(a) Row-level, seed 1998", np.array(R["rs"]), np.array(R["standard"]),
               [d for d in N["row_level"]["paired"]["rs"]["per_run"] if d["seed"] == 1998][0]["best_step"],
               [d for d in N["row_level"]["paired"]["standard"]["per_run"] if d["seed"] == 1998][0]["best_step"])]
    for k, s in enumerate(SEEDS):
        panels.append((f"({'bcd'[k]}) Source-grouped, seed {s}", curve(logs[f"pair_rs_s{s}"]),
                       curve(logs[f"pair_standard_s{s}"]), F[f"pair_rs_s{s}"]["best_step"],
                       F[f"pair_standard_s{s}"]["best_step"]))
    col = 1 if which == "acc" else 2
    for ax, (title, rs, st, brs, bst) in zip(axes.ravel(), panels):
        yr = rs[:, col] * (100 if which == "acc" else 1)
        ys = st[:, col] * (100 if which == "acc" else 1)
        ax.plot(rs[:, 0], yr, "-o", color=BLUE, ms=3, lw=1.6, label="Rs-QLoRA, r = 64")
        ax.plot(st[:, 0], ys, "-s", color=ORANGE, ms=3, lw=1.6, label="Standard QLoRA, r = 64")
        ax.axvline(brs, color=BLUE, lw=0.9, ls="--")
        ax.axvline(bst, color=ORANGE, lw=0.9, ls="--")
        ax.set_title(title, loc="left")
        if which == "acc":
            ax.set_ylim(88, 100.5)
        else:
            ax.set_ylim(0, 0.32)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(ylab)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save(fig, out, name)


def fig_rank(N, out):
    G = N["row_level"]["rank_grid_gpt2"]
    FR = N["source_grouped"]["fixed_scale_rank"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ranks = [8, 16, 32, 64]
    for scal, c, lab, mk in (("standard", ORANGE, "Standard QLoRA", "s"), ("rs", BLUE, "Rs-QLoRA", "o")):
        m = [pct(G[f"{scal}_r{r}"]["accuracy"]["mean"]) for r in ranks]
        s = [pct(G[f"{scal}_r{r}"]["accuracy"]["sd"]) for r in ranks]
        a.errorbar(ranks, m, yerr=s, fmt=f"-{mk}", color=c, ms=5, capsize=3, lw=1.8, label=lab)
    a.set_xscale("log", base=2)
    a.set_xticks(ranks, [str(r) for r in ranks])
    a.set_xlabel("LoRA rank r (α = 2r)")
    a.set_ylabel("Test accuracy (%)")
    a.set_title("(a) Joint grid, original GPT-2-era release", loc="left", fontsize=8.6)
    a.legend(loc="lower right")
    keys = ["r16_a64", "r36_a96", "r64_a128"]
    rr = [FR[k]["rank"] for k in keys]
    m = [pct(FR[k]["summary"]["accuracy"]["mean"]) for k in keys]
    s = [pct(FR[k]["summary"]["accuracy"]["sd"]) for k in keys]
    b.errorbar(rr, m, yerr=s, fmt="-o", color=BLUE, ms=5, capsize=3, lw=1.8)
    for r_, m_, s_, k in zip(rr, m, s, keys):
        b.annotate(f"α = {FR[k]['alpha']}\n{m_:.3f}%", (r_, m_ - s_), textcoords="offset points",
                   xytext=(0, -5), ha="center", va="top", fontsize=7.2, color=INK2)
    b.set_xscale("log", base=2)
    b.set_xticks(rr, [str(r) for r in rr])
    b.set_ylim(99.30, 99.75)
    b.set_xlim(12, 82)
    b.set_xlabel("LoRA rank r (α/√r = 16 fixed)")
    b.set_ylabel("Test accuracy (%)")
    b.set_title("(b) Fixed multiplier, source-grouped DeepSeek", loc="left", fontsize=8.6)
    fig.tight_layout()
    save(fig, out, "figure9")


def fig_memory(N, out):
    G = N["row_level"]["rank_grid_gpt2"]
    FR = N["source_grouped"]["fixed_scale_rank"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ranks = [8, 16, 32, 64]
    mg = [G[f"rs_r{r}"]["peak_alloc_gib"]["mean"] for r in ranks]
    ax.plot(ranks, mg, "-o", color=BLUE, ms=5, label="Joint grid (α = 2r), original GPT-2-era release, RTX 5090 Laptop")
    keys = ["r16_a64", "r36_a96", "r64_a128"]
    rr = [FR[k]["rank"] for k in keys]
    mf = [FR[k]["summary"]["peak_alloc_gib"]["mean"] for k in keys]
    ax.plot(rr, mf, "-s", color=ORANGE, ms=5, label="Fixed multiplier (α/√r = 16), DeepSeek, H100")
    for r_, v in zip(ranks, mg):
        ax.annotate(f"{v:.2f}", (r_, v), textcoords="offset points", xytext=(0, -13), ha="center",
                    fontsize=7.2, color=INK2)
    for r_, v in zip(rr, mf):
        ax.annotate(f"{v:.2f}", (r_, v), textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=7.2, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 36, 64], ["8", "16", "32", "36", "64"])
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Peak PyTorch-allocated memory (GiB)")
    ax.set_ylim(1.6, 2.15)
    ax.legend(loc="upper left")
    save(fig, out, "figure10")


def fig_errors(N, out):
    Rb = N["row_level"]["errors_canonical"]["length_bins"]
    Sb = N["followup"]["errors"]["pooled"]["length_bins"]
    bins = ["<8", "8-14", "15-24", "25-49", "50-99", "100+"]
    x = np.arange(len(bins))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    rv = [pct(Rb[b]["rate"]) for b in bins]
    sv = [pct(Sb[b]["rate"]) for b in bins]
    ax.bar(x - w / 2, rv, width=w - 0.03, color=BLUE, label="Row-level, seed 1998 (n = 5,660)")
    ax.bar(x + w / 2, sv, width=w - 0.03, color=ORANGE,
           label="Source-grouped, three seeds (n = 16,974)")
    for xi, v in zip(x - w / 2, rv):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7, color=INK2)
    for xi, v in zip(x + w / 2, sv):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [b.replace("-", "–") for b in bins])
    ax.set_xlabel("Review length (words)")
    ax.set_ylabel("Error rate (%)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 2.25)
    ax.legend(loc="upper right", fontsize=7.5)
    save(fig, out, "figure11")


def fig_prevalence(N, root, out):
    gens = ["lfm", "deepseek", "glm"]
    prevs = [0.001, 0.005, 0.01, 0.05, 0.10]
    cols = [SEQ[3], SEQ[5], SEQ[7], SEQ[9], SEQ[12]]
    grid = np.linspace(0.5, 1.0, 501)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    rdir = os.path.join(root, "results")
    import glob
    for ax, tg in zip(axes, gens):
        runs = sorted(glob.glob(os.path.join(rdir, f"sgpwo{tg}_rs_r64_s*")))
        rates = []
        for r in runs:
            y, p = load_probs(os.path.join(r, f"source_grouped_eval_{tg}.csv"))
            fpr, tpr, _ = roc_curve(y, p)
            pred = p >= 0.5
            t05 = (pred[y == 1].mean(), pred[y == 0].mean())
            rates.append((fpr, tpr, t05))
        for pi, c in zip(prevs, cols):
            curves = []
            for fpr, tpr, _ in rates:
                with np.errstate(invalid="ignore", divide="ignore"):
                    prec = np.where(pi * tpr + (1 - pi) * fpr > 0,
                                    pi * tpr / (pi * tpr + (1 - pi) * fpr), 1.0)
                interp = np.array([prec[tpr >= g].max() if (tpr >= g).any() else np.nan for g in grid])
                curves.append(interp)
            ax.plot(grid * 100, np.nanmean(curves, axis=0) * 100, color=c, lw=1.6,
                    label=f"π = {pi * 100:g}%")
            r05 = np.mean([t[2][0] for t in rates])
            p05 = np.mean([pi * t[2][0] / (pi * t[2][0] + (1 - pi) * t[2][1]) for t in rates])
            ax.plot(r05 * 100, p05 * 100, "o", color=c, ms=5, mec="white", mew=1.0, zorder=4)
        ax.set_title(f"Held-out {GEN_LABEL[tg]}", loc="left")
        ax.set_xlabel("CG recall (%)")
        ax.set_xlim(50, 100.5)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("Expected alert precision (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "figure12")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--numbers", default=os.path.join(here, "numbers.json"))
    ap.add_argument("--out", default=os.path.join(here, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    N = json.load(open(args.numbers))
    logs = followup_logs(args.root)
    fig_matched(N, args.out)
    fig_same_split(N, args.out)
    fig_directional(N, args.out)
    fig_logo(N, args.out)
    fig_paired(N, args.out)
    fig_traj(N, logs, args.out, "acc", "figure7", "Validation accuracy (%)")
    fig_traj(N, logs, args.out, "loss", "figure8", "Validation loss")
    fig_rank(N, args.out)
    fig_memory(N, args.out)
    fig_errors(N, args.out)
    fig_prevalence(N, args.root, args.out)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
