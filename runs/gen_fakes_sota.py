"""Generate MODERN AI fake reviews with a SOTA local model (Qwen3.6-27B via the
LM Studio OpenAI endpoint), distribution-matched to the dataset's real human
reviews. New research direction: replace the obsolete GPT-2-era "computer-
generated" half of the Kaggle set with genuinely modern LLM fakes, then train a
SOTA detector on  real-human (OR)  vs  modern-AI-fake (CG).

Each generated fake is conditioned on a REAL human review's (category, rating,
word-length) so category/rating/length marginals match the genuine half exactly
and cannot be exploited as trivial shortcuts.

Resumable: appends to dataset/sota_fakes.csv and skips src_idx already present.

  python runs/gen_fakes_sota.py                 # all real-OR reviews
  python runs/gen_fakes_sota.py --limit 200     # quick subset
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

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "fake reviews dataset.csv"
OUT = ROOT / "dataset" / "sota_fakes.csv"          # overridden by --out
COLS = ["src_idx", "category", "rating", "label", "text_", "gen_model"]

client = None          # built in main() from CLI args
EXTRA_BODY = None       # provider-specific (e.g. DeepSeek thinking-disable)


def load_env(path=ROOT / ".env"):
    if path.exists():
        for ln in open(path, encoding="utf-8"):
            m = re.match(r'\s*([A-Za-z_]+)\s*=\s*["\']?([^"\'\r\n]*)', ln)
            if m:  # override + strip \r (a stray CRLF makes the key invalid -> 401)
                os.environ[m.group(1)] = m.group(2).strip()
# Plain instruct models: output the review directly.
SYS_PLAIN = ("You write realistic, concise Amazon-style customer product reviews. "
             "Output ONLY the review text — no preamble, no surrounding quotes, no "
             "title, no rating line, no markdown.")
# Reasoning models (e.g. qwen3.5-35b-a3b) keep thinking; we can't disable it via
# the API, so we force a clearly delimited final answer and extract it.
SYS_REASON = ("You write realistic Amazon-style customer reviews. You may think "
              "briefly, but you MUST end your response with the final review on its "
              "own line starting with 'FINAL: ' followed by only the review text "
              "(no quotes, no word count, no extra commentary after it).")
_lock = threading.Lock()


def nice(cat):
    return cat[:-2].replace("_", " ") if cat.endswith("_5") else cat.replace("_", " ")


def _tidy(text):
    text = re.sub(r'^["\'`\s]+', '', text)
    text = re.sub(r'^(sure[,!]?|here(\'s| is)[^:]*:|review:)\s*', '', text, flags=re.I)
    text = re.sub(r'\(\s*\d+\s*words?\s*\)', '', text, flags=re.I)   # "(25 words)"
    text = re.sub(r'\(\d+\)', '', text)                              # draft word-counters
    text = re.sub(r'["\'`\s]+$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_review(raw, reasoning):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
    if not reasoning:
        return _tidy(raw)
    # reasoning path: prefer the explicit FINAL: marker (last occurrence)
    m = list(re.finditer(r'FINAL:\s*', raw, flags=re.I))
    if m:
        return _tidy(raw[m[-1].end():].splitlines()[0] if "\n" in raw[m[-1].end():]
                     else raw[m[-1].end():])
    # fallback: last prose-looking paragraph (no numbered list / bold / "attempt")
    paras = [p.strip() for p in re.split(r'\n\s*\n', raw) if p.strip()]
    for p in reversed(paras):
        if (len(p.split()) >= 6 and not re.match(r'^\s*(\d+[\.\)]|\*|#|attempt|draft)',
                                                  p, flags=re.I)
                and "thinking process" not in p.lower()):
            return _tidy(p)
    return ""


def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


# different reviewers emphasise different things -> diversify the angle per draft
ANGLES = ["overall experience", "a specific feature you liked or disliked",
          "value for the price", "how it compares to expectations",
          "shipping/packaging and first impressions", "durability after some use",
          "who you'd recommend it to"]


def gen_one(model, src_idx, cat, rating, real_text, reasoning):
    target_len = max(4, min(len(real_text.split()), 120))
    sys = SYS_REASON if reasoning else SYS_PLAIN
    rnd = random.Random(src_idx)
    for attempt in range(5):
        angle = rnd.choice(ANGLES)
        # ground on the real review so the fake is product-specific and varied,
        # but demand a DIFFERENT review (not a paraphrase)
        user = (
            f"Here is a real customer review of a product in the {nice(cat)} "
            f"category:\n\"{real_text[:400]}\"\n\n"
            f"Write a NEW, different {int(float(rating))}-star review for the SAME "
            f"product as if you were another genuine customer. Emphasise {angle}. "
            f"About {target_len} words. Natural, specific, casual. Do NOT copy "
            f"phrasing from the review above. Output only the review text.")
        mt = (800 if reasoning else int(target_len * 4) + 100)
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": user}],
                temperature=round(0.85 + 0.06 * attempt, 2), top_p=0.95,
                max_tokens=mt, timeout=240,
                **({"extra_body": EXTRA_BODY} if EXTRA_BODY else {}))
            txt = extract_review(r.choices[0].message.content or "", reasoning)
            # length cap proportional to the ask: models overshoot "about N words"
            # by ~20-30%; a flat cap of 130 permanently rejected every long source
            if (3 <= len(txt.split()) <= max(130, int(target_len * 1.5))
                    and jaccard(txt, real_text) < 0.6):  # not a near-copy
                return {"src_idx": src_idx, "category": cat, "rating": rating,
                        "label": "CG", "text_": txt, "gen_model": model}
        except Exception:
            pass
    return None


def load_real_or():
    rows = []
    for i, r in enumerate(csv.DictReader(open(DATA, encoding="utf-8"))):
        if r["label"] == "OR":
            rows.append((i, r["category"], r["rating"], r["text_"]))
    return rows


def done_set():
    if not OUT.exists():
        return set()
    return {int(r["src_idx"]) for r in csv.DictReader(open(OUT, encoding="utf-8"))}


def main():
    global client, EXTRA_BODY, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-35b-a3b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all real-OR reviews")
    ap.add_argument("--sample", type=int, default=0,
                    help="random sample N across all categories (testing diversity)")
    ap.add_argument("--reasoning", choices=["auto", "on", "off"], default="auto",
                    help="reasoning model -> force+extract a FINAL: review line")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--api-key-env", default=None,
                    help="env var holding the API key (e.g. DEEPSEEK_API_KEY)")
    ap.add_argument("--no-think", action="store_true",
                    help="disable reasoning via extra_body (DeepSeek v4)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    load_env()
    api_key = (os.environ.get(args.api_key_env) if args.api_key_env else "lm-studio")
    client = OpenAI(base_url=args.base_url, api_key=api_key)
    if args.no_think:
        EXTRA_BODY = {"thinking": {"type": "disabled"}}
    OUT = Path(args.out)
    # thinking disabled => model answers directly => no FINAL-marker extraction
    reasoning = (not args.no_think) and (
        args.reasoning == "on" or
        (args.reasoning == "auto" and
         any(k in args.model.lower() for k in ("qwen3", "glm", "think"))))

    real = load_real_or()
    random.Random(1998).shuffle(real)   # category-balanced partial runs
    if args.sample:
        real = real[:args.sample]
    elif args.limit:
        real = real[:args.limit]
    done = done_set()
    todo = [r for r in real if r[0] not in done]
    print(f"real OR={len(real)}  already done={len(done & {r[0] for r in real})}  "
          f"to generate={len(todo)}  model={args.model}  reasoning={reasoning}")

    new_file = not OUT.exists()
    fh = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new_file:
        w.writeheader(); fh.flush()

    done_n, ok_n, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen_one, args.model, *r, reasoning) for r in todo]
        for fut in as_completed(futs):
            res = fut.result()
            done_n += 1
            if res:
                ok_n += 1
                with _lock:
                    w.writerow(res); fh.flush()
            if done_n % 25 == 0:
                rate = done_n / (time.time() - t0)
                print(f"  {done_n}/{len(todo)}  ok={ok_n} ({ok_n/done_n*100:.0f}%)  "
                      f"{rate:.2f}/s  eta={(len(todo)-done_n)/max(rate,1e-9)/60:.0f}min",
                      flush=True)
    fh.close()
    total = len(done_set())
    print(f"done. sota_fakes.csv now has {total} fakes -> {OUT}")


if __name__ == "__main__":
    main()
