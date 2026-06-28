"""Overfitting analysis across runs (R1-4: rs vs standard 'instability' = overfitting).

Prefers log_history.json (full trajectory through early-stop). Falls back to the
best-checkpoint trainer_state.json (truncated at best step) for older runs.
Reports per scaling x rank: train loss at best, min eval_loss, eval_loss at the
accuracy-best step, the train/val gap, and (if full history) post-peak val-loss rise.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_history(run_dir):
    full = run_dir / "log_history.json"
    if full.exists():
        return json.loads(full.read_text()), True
    ts = run_dir / "trainer_state.json"
    if ts.exists():
        return json.loads(ts.read_text())["log_history"], False
    return None, False


def analyze(hist):
    train = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    evals = [(h["step"], h["eval_loss"], h.get("eval_accuracy"))
             for h in hist if "eval_loss" in h]
    if not evals:
        return None
    min_eval = min(e[1] for e in evals)
    best_acc_step = max(evals, key=lambda e: (e[2] or 0))[0]
    eval_at_best = next(e[1] for e in evals if e[0] == best_acc_step)
    final_eval = evals[-1][1]
    train_at_best = min((t for t in train if t[0] <= best_acc_step),
                        key=lambda t: abs(t[0] - best_acc_step), default=(0, None))[1]
    return {
        "train_at_best": train_at_best,
        "min_eval": min_eval,
        "eval_at_best": eval_at_best,
        "gap_at_best": (eval_at_best - train_at_best) if train_at_best else None,
        "post_peak_rise": final_eval - min_eval,   # only meaningful w/ full history
        "best_acc_step": best_acc_step,
    }


cells = defaultdict(list)
for d in sorted(RESULTS.iterdir()):
    if not d.is_dir() or not (d / "config.json").exists():
        continue
    cfg = json.loads((d / "config.json").read_text())
    hist, is_full = load_history(d)
    if hist is None:
        continue
    a = analyze(hist)
    if a:
        a["full"] = is_full
        cells[(cfg["scaling"], cfg["rank"])].append(a)

hdr = f"{'cell':<16}{'n':>2} {'trn@best':>9}{'min_val':>9}{'val@best':>9}{'gap':>8}{'postrise':>9}{'beststep':>9}{'full':>6}"
print(hdr)
print("-" * len(hdr))
for (scaling, rank) in sorted(cells, key=lambda k: (k[0], k[1])):
    runs = cells[(scaling, rank)]
    def avg(k):
        vals = [r[k] for r in runs if r[k] is not None]
        return st.mean(vals) if vals else float("nan")
    nfull = sum(r["full"] for r in runs)
    print(f"{scaling+'/r'+str(rank):<16}{len(runs):>2} "
          f"{avg('train_at_best'):>9.4f}{avg('min_eval'):>9.4f}"
          f"{avg('eval_at_best'):>9.4f}{avg('gap_at_best'):>8.4f}"
          f"{avg('post_peak_rise'):>9.4f}{avg('best_acc_step'):>9.0f}{nfull:>5}/{len(runs)}")

print("\nLegend: trn@best=train loss at peak; min_val=lowest eval_loss; "
      "val@best=eval_loss at peak-accuracy step;\n  gap=val@best-trn@best (overfit "
      "margin); postrise=final-min eval_loss (needs full history);\n  lower gap & "
      "lower postrise = less overfitting. 'full' = runs with untruncated history.")
