"""Experiment: find a prompt that makes DeepSeek fakes statistically indistinguishable
from real human reviews. For each prompt variant: generate N grounded fakes, then
measure TF-IDF separability vs a held-out set of real reviews (5-fold CV accuracy;
lower = more human-like). Prints the top 'tells' giving each variant away.
"""
import csv, os, re, random, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from openai import OpenAI
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
from build_modern_dataset import normalize_text  # noqa

for ln in open(ROOT / ".env", encoding="utf-8"):
    m = re.match(r'\s*([A-Z_]+)\s*=\s*["\']?([^"\'\n]+)', ln)
    if m: os.environ.setdefault(m.group(1), m.group(2))
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
MODEL = "deepseek-v4-pro"
N = 300

def nice(c): return c[:-2].replace("_", " ") if c.endswith("_5") else c
rows = [r for r in csv.DictReader(open(ROOT/"dataset"/"fake reviews dataset.csv", encoding="utf-8")) if r["label"]=="OR"]
rng = random.Random(1998); rng.shuffle(rows)
ground = rows[:N]                 # seed reviews to ground fakes on
real_cls = [normalize_text(r["text_"]) for r in rows[N:2*N]]   # the 'real' class
by_cat = {}
for r in rows[2*N:]:
    by_cat.setdefault(r["category"], []).append(r["text_"])

def examples(cat, k=3):
    pool = by_cat.get(cat, []); return random.sample(pool, min(k, len(pool)))

# ---------- prompt variants: (system, user) builders ----------
def v0_baseline(cat, rating, tlen, real):
    return ("You write realistic Amazon-style customer reviews. Output only the review text, no preamble.",
            f"Write a believable {rating}-star review for a product in the {nice(cat)} category. "
            f"About {tlen} words. Natural, specific, casual. Output only the review.")

def v4_rewrite(cat, rating, tlen, real):
    return ("You imitate a sample review's exact register: its length, vocabulary level, grammar quality, "
            "punctuation habits and imperfections. You are NOT an eloquent writer. Output only the review.",
            f"Here is a real customer review:\n\"{real}\"\n\nWrite a DIFFERENT {rating}-star review for "
            f"another {nice(cat)} product. Copy this example's style as closely as possible: same length "
            f"(~{tlen} words), same plain words, same casual/imperfect grammar, typos and capitalization. "
            f"Do NOT be more polished, articulate, or detailed than the example. Output only the review.")

def v5_rewrite_messy(cat, rating, tlen, real):
    return ("You rewrite in the voice of an ordinary shopper, not a writer. Keep it as plain and imperfect "
            "as the sample. BANNED: em-dashes, semicolons, and the words honestly, perfectly, solid, "
            "game-changer, elevate, overall, seamless, sturdy. Output only the review.",
            f"Sample real review:\n\"{real}\"\n\nWrite a new {rating}-star review for a different "
            f"{nice(cat)} product in the same plain, slightly sloppy voice, about {tlen} words. Match the "
            f"sample's simplicity, do not improve its grammar. Output only the review.")

def v6_heavy_fewshot(cat, rating, tlen, real):
    shots = "\n".join(f'- "{e}"' for e in examples(cat, 6))
    return ("You generate a review that is statistically indistinguishable from a set of real ones: same "
            "plainness, brevity, typos, caps, and word choices. Avoid polished or 'AI' phrasing. Output only the review.",
            f"Real {nice(cat)} reviews:\n{shots}\n\nWrite ONE more {rating}-star review for a different "
            f"{nice(cat)} product that would be impossible to tell apart from the list above. "
            f"About {tlen} words. Output only the review.")

def v8_rewrite_exact(cat, rating, tlen, real):
    return ("You reproduce the EXACT register of a sample review: if it starts lowercase you start lowercase; "
            "if it has typos, missing punctuation, ALL-CAPS words or run-ons, you include similar imperfections; "
            "if it is blunt and short you are blunt and short. Never sound like an AI assistant. Output only the review.",
            f"Sample:\n\"{real}\"\n\nWrite a different {rating}-star review for another {nice(cat)} product that "
            f"a detector could not distinguish from the sample's author. About {tlen} words. Mirror its "
            f"imperfections. Output only the review.")

def v9_spin(cat, rating, tlen, real):
    return ("You lightly edit a real review so it describes a DIFFERENT product, changing as little as "
            "possible. Keep the original's exact tone, length, grammar, typos, capitalization and "
            "punctuation. Output only the edited review.",
            f"Original review:\n\"{real}\"\n\nRewrite it as a {rating}-star review for a different "
            f"{nice(cat)} product. Change ONLY the product specifics and a few words. Keep everything "
            f"else the same. Output only the review.")

def v11_match_punct(cat, rating, tlen, real):
    return ("You write a review in the same voice as a sample, using the SAME level of capitalization, "
            "apostrophes and punctuation as the sample - do not be more OR less casual than it. Plain "
            "words, no AI phrasing. Output only the review.",
            f"Sample:\n\"{real}\"\n\nWrite a different {rating}-star review for another {nice(cat)} "
            f"product, matching the sample's exact register and ~{tlen} words. Output only the review.")

VARIANTS = {"v8_rewrite_exact": v8_rewrite_exact, "v11_match_punct": v11_match_punct,
            "v9_spin": v9_spin}

def gen_one(builder, r):
    cat, rating = r["category"], str(int(float(r["rating"])))
    tlen = max(4, min(len(r["text_"].split()), 120))
    sysmsg, usr = builder(cat, rating, tlen, r["text_"])
    try:
        resp = client.chat.completions.create(model=MODEL,
            messages=[{"role":"system","content":sysmsg},{"role":"user","content":usr}],
            temperature=round(random.uniform(0.85, 1.2), 2), max_tokens=400,
            extra_body={"thinking":{"type":"disabled"}})
        t = normalize_text(resp.choices[0].message.content or "")
        return (t, r["text_"]) if len(t.split())>=3 else None
    except Exception:
        return None

def separability(fakes):
    texts = fakes + real_cls[:len(fakes)]
    y = np.array([1]*len(fakes) + [0]*len(fakes))
    wv = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=2)
    cv = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3,5), min_df=2)
    X = hstack([wv.fit_transform(texts), cv.fit_transform(texts)]).tocsr()
    acc = cross_val_score(LogisticRegression(C=4.0, max_iter=2000), X, y, cv=5).mean()
    lr = LogisticRegression(C=4.0, max_iter=2000).fit(X, y)
    names = np.array(list(wv.get_feature_names_out()) + ["c:"+c for c in cv.get_feature_names_out()])
    tells = [names[i] for i in np.argsort(lr.coef_[0])[-8:][::-1]]
    return acc, tells

def jaccard(a, b):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)

print(f"N={N} per variant; real class = {len(real_cls)} held-out human reviews")
print("(copy_jac = mean word-overlap with the source review; high => near-plagiarism)\n")
for name, builder in VARIANTS.items():
    with ThreadPoolExecutor(max_workers=24) as ex:
        pairs = [f for f in ex.map(lambda r: gen_one(builder, r), ground) if f]
    fakes = [t for t, s in pairs]
    copy_jac = float(np.mean([jaccard(t, s) for t, s in pairs]))
    acc, tells = separability(fakes)
    print(f"{name:18} sep_acc={acc*100:5.2f}%  copy_jac={copy_jac:.2f}  (n={len(fakes)})  tells={tells[:6]}")
