"""D2: download + normalize external review corpora into dataset/external/*.csv
(columns: text_, label) for zero-shot eval with runs/eval_external.py.

Needs internet on the instance. Each source is attempted independently and
failures are non-fatal (a clear manual-download hint is printed instead).

  ott_deceptive.csv  : Ott et al. deceptive-opinion-spam (hotel). deceptive->CG,
                       truthful->OR. Cross-domain transfer probe.
  amazon_genuine.csv : sample of real human Amazon reviews (all OR). False-
                       positive-rate probe on out-of-distribution genuine text.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "dataset" / "external"
# NOTE: hard-coded mirror URLs are unreliable (they 404). We only attempt HF
# datasets; if unavailable, place a normalized text_/label CSV manually. This
# external-corpus path is now secondary — the modern-AI-fake study (real OR vs
# Qwen-generated fakes) is the primary external-validity evidence.


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text_", "label"])
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {len(rows)} -> {path}")


def prepare_ott():
    out = EXT / "ott_deceptive.csv"
    # 1) try HF datasets
    try:
        from datasets import load_dataset
        for name in ("deceptive-opinion", "ott_deceptive"):
            try:
                ds = load_dataset(name, split="train")
                rows = [{"text_": r["text"],
                         "label": "CG" if str(r.get("deceptive", r.get("label")))
                         .lower().startswith("decep") else "OR"} for r in ds]
                if rows:
                    write_csv(out, rows); return
            except Exception:
                continue
    except Exception:
        pass
    print("  [SKIP] Ott corpus not auto-available via HF. To use it, place a "
          "normalized text_/label CSV (deceptive->CG, truthful->OR) at", out)


def prepare_amazon(n=2000):
    out = EXT / "amazon_genuine.csv"
    try:
        from datasets import load_dataset
        ds = load_dataset("amazon_polarity", split="test", streaming=True)
        rows = []
        for r in ds:
            t = (r.get("content") or r.get("text") or "").strip()
            if 5 <= len(t.split()) <= 200:
                rows.append({"text_": t, "label": "OR"})
            if len(rows) >= n:
                break
        if rows:
            write_csv(out, rows); return
    except Exception as e:
        print(f"  amazon_polarity failed: {e}")
    print("  [SKIP] genuine-Amazon sample unavailable (needs HF datasets + net).")


def main():
    EXT.mkdir(parents=True, exist_ok=True)
    print(">> Ott deceptive-opinion corpus"); prepare_ott()
    print(">> genuine Amazon sample"); prepare_amazon()


if __name__ == "__main__":
    main()
