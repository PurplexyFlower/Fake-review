"""D3: generate MODERN-LLM product reviews via a local LM Studio OpenAI endpoint.

Tests the reviewer's point that the Kaggle 'computer-generated' reviews may be
GPT-2-era artifacts rather than modern fakes. We generate realistic reviews with
a current small instruct model (LFM2.5-350M) across the same 10 categories, then
(separately) zero-shot the headline detector on them: a LOW detection rate means
the detector learned generator-specific artifacts, not general machine-text.

Output: dataset/llm_generated_reviews.csv  (category, rating, label, text_, gen_model)
Honest either way. No external paid API; everything local.

  python runs/gen_llm_reviews.py --n-per-cat 300 --model lfm2.5-350m
"""
import argparse
import csv
import json
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "llm_generated_reviews.csv"
ENDPOINT = "http://localhost:1234/v1/chat/completions"

CATEGORIES = [
    "Home_and_Kitchen_5", "Books_5", "Pet_Supplies_5", "Electronics_5",
    "Sports_and_Outdoors_5", "Tools_and_Home_Improvement_5",
    "Clothing_Shoes_and_Jewelry_5", "Toys_and_Games_5", "Movies_and_TV_5",
    "Kindle_Store_5",
]
# realistic product hints per category so reviews aren't all generic
HINTS = {
    "Home_and_Kitchen_5": ["blender", "cutting board", "coffee maker", "bed sheets",
                           "knife set", "storage containers", "vacuum", "cookware"],
    "Books_5": ["a novel", "a cookbook", "a biography", "a thriller",
                "a self-help book", "a history book", "a children's book"],
    "Pet_Supplies_5": ["dog food", "a cat toy", "a leash", "a pet bed",
                       "a grooming brush", "an aquarium filter", "training treats"],
    "Electronics_5": ["headphones", "a phone charger", "a bluetooth speaker",
                      "an HDMI cable", "a webcam", "a power bank", "a smart plug"],
    "Sports_and_Outdoors_5": ["a yoga mat", "a water bottle", "a tent",
                             "running shoes", "a bike pump", "dumbbells", "a backpack"],
    "Tools_and_Home_Improvement_5": ["a drill", "a tape measure", "a wrench set",
                                    "LED bulbs", "a ladder", "a paint roller", "screws"],
    "Clothing_Shoes_and_Jewelry_5": ["a jacket", "running shoes", "a watch",
                                    "a dress", "a necklace", "jeans", "a handbag"],
    "Toys_and_Games_5": ["a board game", "a building set", "a puzzle",
                        "an action figure", "a plush toy", "a remote-control car"],
    "Movies_and_TV_5": ["a movie", "a TV series box set", "a documentary",
                       "a classic film", "an animated film"],
    "Kindle_Store_5": ["an ebook romance", "an ebook mystery", "an ebook fantasy",
                      "a non-fiction ebook", "a short-story collection"],
}
RATINGS, RATING_W = [5, 4, 3, 2, 1], [0.50, 0.22, 0.13, 0.08, 0.07]

# Short system prompt: weak models echo long instruction lists, so keep it minimal.
SYSTEM = "You write realistic, concise Amazon-style customer reviews."

# fragments that mean the model leaked the instructions instead of reviewing
LEAK = re.compile(
    r"natural sentences|casual tone|no preamble|no quotation|no rating|no title|"
    r"specific and believable|sample review|here'?s (a|the)|as an ai|"
    r"\bstar review\b|^title:|^review:|^sure[,!]|preamble|"
    r"glad to help|happy to help|hope (this|that) helps|let me know", re.I)
MIN_WORDS, MAX_WORDS = 6, 70

_lock = threading.Lock()


def nice(cat):
    return cat[:-2].replace("_", " ") if cat.endswith("_5") else cat.replace("_", " ")


def clean(text):
    text = text.strip()
    # strip wrapping quotes and common boilerplate prefixes/suffixes
    text = re.sub(r'^["\'`\s]+', '', text)
    text = re.sub(r'^(review|here(\'s| is)[^:]*|sure[,!]?|title)\s*:?\s*', '',
                  text, flags=re.I)
    text = re.sub(r'\s*\(?\d\s*-?\s*star[s]?\b.*$', '', text, flags=re.I)  # trailing "1-star"
    text = re.sub(r'\s*\((?:no [^)]*)\)\s*$', '', text, flags=re.I)        # "(no rating...)"
    text = re.sub(r'\s*(overall satisfaction|rating|score)\s*:?\s*\d.*$', '',
                  text, flags=re.I)                                        # "Rating: 3/5"
    text = re.sub(r'\s*\b\d\s*/\s*5\b.*$', '', text)                       # trailing "3/5"
    text = re.sub(r'\s*\([^)]*$', '', text)                                # unclosed "(4-"
    text = re.sub(r'["\'`\s]+$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # if the model emitted a leading instruction-echo sentence, drop it and keep the rest
    if LEAK.search(text):
        parts = re.split(r'(?<=[.!?])\s+', text)
        parts = [p for p in parts if not LEAK.search(p)]
        text = ' '.join(parts).strip()
    return text


def accept(text):
    n = len(text.split())
    return MIN_WORDS <= n <= MAX_WORDS and not LEAK.search(text)


def gen_one(model, cat, rating, seed):
    rnd = random.Random(seed)
    hint = rnd.choice(HINTS[cat])
    # vary seed/temperature across retries so a bad slot can recover
    for attempt in range(5):
        user = (f"Write a short {rating}-star customer review for {hint} "
                f"(category: {nice(cat)}). 1-3 sentences, under 40 words, "
                f"sound like a genuine buyer. Output only the review text.")
        body = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "temperature": round(0.7 + 0.1 * attempt, 2),
            "max_tokens": 110, "top_p": 0.95, "seed": seed + attempt * 9973,
        }).encode()
        req = urllib.request.Request(ENDPOINT, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            txt = clean(out["choices"][0]["message"]["content"])
            if accept(txt):
                return {"category": cat, "rating": float(rating), "label": "CG",
                        "text_": txt, "gen_model": model}
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=300)
    ap.add_argument("--model", default="lfm2.5-350m")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rng = random.Random(1998)
    jobs = []
    sid = 0
    for cat in CATEGORIES:
        for _ in range(args.n_per_cat):
            rating = rng.choices(RATINGS, RATING_W)[0]
            jobs.append((cat, rating, sid)); sid += 1
    rng.shuffle(jobs)
    print(f"generating {len(jobs)} reviews with {args.model}, {args.workers} workers")

    rows, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, args.model, c, r, s): (c, r, s)
                for (c, r, s) in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if res:
                rows.append(res)
            if done % 100 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(jobs)}  ok={len(rows)}  "
                      f"{rate:.1f}/s  eta={ (len(jobs)-done)/max(rate,1e-6)/60:.1f}min")

    # dedup exact-equal text
    seen, uniq = set(), []
    for r in rows:
        k = r["text_"].lower()
        if k not in seen:
            seen.add(k); uniq.append(r)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category", "rating", "label",
                                          "text_", "gen_model"])
        w.writeheader(); w.writerows(uniq)
    from collections import Counter
    print(f"\nwrote {len(uniq)} unique reviews -> {OUT}")
    print("per-category:", dict(Counter(r["category"] for r in uniq)))
    print("ratings:", dict(Counter(r["rating"] for r in uniq)))
    lens = [len(r["text_"].split()) for r in uniq]
    print(f"word-length: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")


if __name__ == "__main__":
    main()
