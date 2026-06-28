"""Generate modern AI-fake reviews with Claude Opus 4.8 via the Anthropic API.
Same grounded prompt as the DeepSeek/LFM generators (so the only variable is the
model), producing a 4th generator for the cross-gen matrix.

  python runs/gen_fakes_opus.py --limit 200      # quick subset
  python runs/gen_fakes_opus.py                   # all real-OR reviews

Key (CLAUDE_API_KEY) is read from .env. NOTE: Opus 4.8 rejects temperature/top_p
(400) — diversity comes from grounding + per-draft angle rotation. Output ->
dataset/sota_fakes_opus.csv (resumable; build_modern_dataset.py --fakes it).
"""
import argparse
import csv
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "fake reviews dataset.csv"
OUT = ROOT / "dataset" / "sota_fakes_opus.csv"
COLS = ["src_idx", "category", "rating", "label", "text_", "gen_model"]
MODEL = "claude-opus-4-8"

SYSTEM = ("You write realistic, concise Amazon-style customer product reviews. "
          "Output ONLY the review text — no preamble, no surrounding quotes, no "
          "title, no rating line, no markdown.")
# Same angle set as the DeepSeek/LFM runs (keep identical for a fair generator).
ANGLES = ["overall experience", "a specific feature you liked or disliked",
          "value for the price", "how it compares to expectations",
          "shipping/packaging and first impressions", "durability after some use",
          "who you'd recommend it to"]
client = None
_lock = threading.Lock()


def load_env(path=ROOT / ".env"):
    if path.exists():
        for ln in open(path, encoding="utf-8"):
            m = re.match(r'\s*([A-Za-z_]+)\s*=\s*["\']?([^"\'\r\n]*)', ln)
            if m:
                os.environ[m.group(1)] = m.group(2).strip()


def nice(c):
    return c[:-2].replace("_", " ") if c.endswith("_5") else c.replace("_", " ")


def clean(text):
    text = re.sub(r'^["\'`\s]+', '', text)
    text = re.sub(r'^(sure[,!]?|here(\'s| is)[^:]*:|review:)\s*', '', text, flags=re.I)
    text = re.sub(r'["\'`\s]+$', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


def gen_one(src_idx, cat, rating, real_text):
    target_len = max(4, min(len(real_text.split()), 120))
    rnd = random.Random(src_idx)
    for attempt in range(4):
        angle = rnd.choice(ANGLES)
        user = (f"Here is a real customer review of a product in the {nice(cat)} "
                f"category:\n\"{real_text[:400]}\"\n\n"
                f"Write a NEW, different {int(float(rating))}-star review for the "
                f"SAME product as if you were another genuine customer. Emphasise "
                f"{angle}. About {target_len} words. Natural, specific, casual. Do "
                f"NOT copy phrasing from the review above. Output only the review text.")
        try:
            m = client.messages.create(
                model=MODEL, max_tokens=300, system=SYSTEM,
                messages=[{"role": "user", "content": user}])
            if m.stop_reason == "refusal":
                continue
            raw = next((b.text for b in m.content if b.type == "text"), "")
            txt = clean(raw)
            if 3 <= len(txt.split()) <= 130 and jaccard(txt, real_text) < 0.6:
                return {"src_idx": src_idx, "category": cat, "rating": rating,
                        "label": "CG", "text_": txt, "gen_model": MODEL}
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def real_or():
    return [(i, r["category"], r["rating"], r["text_"])
            for i, r in enumerate(csv.DictReader(open(DATA, encoding="utf-8")))
            if r["label"] == "OR"]


def done():
    if not OUT.exists():
        return set()
    return {int(r["src_idx"]) for r in csv.DictReader(open(OUT, encoding="utf-8"))}


def main():
    global client
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all real-OR reviews")
    args = ap.parse_args()

    load_env()
    key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("no CLAUDE_API_KEY in .env")
    client = anthropic.Anthropic(api_key=key, max_retries=5)

    rows = real_or()
    random.Random(1998).shuffle(rows)          # category-balanced partial runs
    if args.limit:
        rows = rows[:args.limit]
    have = done()
    todo = [r for r in rows if r[0] not in have]
    print(f"real OR={len(rows)}  done={len(have & {r[0] for r in rows})}  "
          f"to generate={len(todo)}  model={MODEL}")

    new_file = not OUT.exists()
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new_file:
        w.writeheader(); fh.flush()

    n, ok, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen_one, *r) for r in todo]
        for fut in as_completed(futs):
            res = fut.result(); n += 1
            if res:
                ok += 1
                with _lock:
                    w.writerow(res); fh.flush()
            if n % 25 == 0:
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)} ok={ok} ({ok/n*100:.0f}%) {rate:.2f}/s "
                      f"eta={(len(todo)-n)/max(rate,1e-9)/60:.0f}min", flush=True)
    fh.close()
    print(f"done. {len(done())} opus fakes -> {OUT}")


if __name__ == "__main__":
    main()
