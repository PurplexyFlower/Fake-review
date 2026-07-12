# Dataset generation and provenance

This document records how the modern fake-review datasets were produced, what
can be reproduced exactly, and what information was not captured at generation
time. The machine-readable source of truth is
[`dataset/provenance.json`](dataset/provenance.json).

## Scope

The study uses the genuine (`OR`) half of the Kaggle Fake Reviews Dataset as the
human source and replaces its historical machine-generated half with grounded
reviews from three modern generators:

| Dataset | Request model recorded in raw rows | Raw CG rows | Final OR / CG | Final rows |
|---|---|---:|---:|---:|
| LFM | `lfm2.5-1.2b-instruct` | 20,216 | 20,212 / 20,212 | 40,424 |
| DeepSeek | `deepseek-v4-pro` | 20,216 | 20,212 / 20,212 | 40,424 |
| GLM | `glm-5.2` | 20,216 | 20,208 / 20,208 | 40,416 |

The four- and eight-example reductions occur during exact/cross-class
deduplication in `runs/build_modern_dataset.py`.

## Human source

- Local artifact: `dataset/fake reviews dataset.csv`
- SHA-256: `d2581f5436ae92a8106885988fd5381ea562cc366a57dcffb942c1090761282a`
- Genuine source rows selected: 20,216 rows with `label == "OR"`
- Upstream page: <https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset>

The Hub dataset cards currently declare CC-BY-4.0. Downstream users remain
responsible for checking the upstream Kaggle dataset terms and any product-review
content restrictions.

## Exact generation protocol

Implementation: `runs/gen_fakes_sota.py`.

For each genuine review, the generator receives its category, integer star
rating, and the first 400 characters of its text. Target length is the genuine
review's word count clamped to 4–120 words.

Plain system prompt:

```text
You write realistic, concise Amazon-style customer product reviews. Output ONLY
the review text — no preamble, no surrounding quotes, no title, no rating line,
no markdown.
```

User-prompt template:

```text
Here is a real customer review of a product in the {category} category:
"{first 400 characters of the genuine review}"

Write a NEW, different {rating}-star review for the SAME product as if you were
another genuine customer. Emphasise {angle}. About {target_length} words.
Natural, specific, casual. Do NOT copy phrasing from the review above. Output
only the review text.
```

`angle` is selected deterministically from:

1. overall experience;
2. a specific feature liked or disliked;
3. value for the price;
4. comparison with expectations;
5. shipping/packaging and first impressions;
6. durability after some use;
7. intended recipient or recommendation.

Generation and acceptance settings:

- genuine-source order shuffled with Python seed 1998;
- angle RNG seeded independently with the source-row index;
- up to five attempts per source review;
- temperature `0.85 + 0.06 * attempt` (`0.85` through `1.09`);
- `top_p = 0.95`;
- no API decoding seed was supplied;
- plain-response maximum tokens: `4 * target_length + 100`;
- accepted length: at least 3 words and no more than
  `max(130, 1.5 * target_length)` in the current implementation;
- source/generated word-set Jaccard similarity must be below 0.6;
- leading boilerplate, quotes, counters, and repeated whitespace are removed;
- failures are retried, but historical per-attempt responses and exception
  details were not retained.

DeepSeek and GLM were invoked with provider reasoning disabled and treated as
plain-response models. LFM was served locally through an OpenAI-compatible LM
Studio endpoint and also followed the plain-response path.

## Provider-specific records

### LFM

- Request model: `lfm2.5-1.2b-instruct`
- Serving path: local LM Studio OpenAI-compatible endpoint
- Raw file: `dataset/sota_fakes.csv`
- Raw SHA-256: `8bfe7eb10be6f815eabdf0e8169b35adac1db7c5e1df540b9ef7fd641b2a6270`
- Final file: `dataset/modern_reviews_lfm.csv`
- Final SHA-256: `6a0f516f0fee535f3c3fc31fc613dba484684e9f44f89db36b57655382c201e7`
- Artifact commit: `a1a77c419a59658f6cf06ea5c27e4f0bdab25229`
- Hub: <https://huggingface.co/datasets/Flowerly/modern-fake-reviews-lfm>

The exact LFM Hub checkpoint, quantization, LM Studio version, and backend build
were not recorded. The committed raw generations are therefore the canonical
artifact.

### DeepSeek

- Request model: `deepseek-v4-pro`
- Provider endpoint: `https://api.deepseek.com`
- Reasoning: disabled through `{"thinking": {"type": "disabled"}}`
- Raw file: `dataset/sota_fakes_deepseek.csv`
- Raw SHA-256: `a47568d219a38f5cc16207269d9d855ae370405fd703ae5c7d27e2131a019cc1`
- Final file: `dataset/modern_reviews_deepseek.csv`
- Final SHA-256: `c21b3a596101d488954058f56d72f35512a34dac1526ef8cbbd3d307fb005783`
- Artifact commit: `a1a77c419a59658f6cf06ea5c27e4f0bdab25229`
- Hub: <https://huggingface.co/datasets/Flowerly/modern-fake-reviews>

`deepseek-v4-pro` is a provider alias. A provider-side immutable revision or
snapshot identifier was not returned or retained.

### GLM

- Request model: `glm-5.2`
- Provider endpoint: `https://api.z.ai/api/paas/v4/`
- Reasoning: disabled through `{"thinking": {"type": "disabled"}}`
- Raw file: `dataset/sota_fakes_glm.csv`
- Raw SHA-256: `4d1b96580a2718e0602fe633a6c6b593da24880030772a87ec5c620c09048664`
- Final file: `dataset/modern_reviews_glm.csv`
- Final SHA-256: `1519dedd45a8bd2d04c12a201ad488b7c6a919122f2d9bc86d69955f49a2ae41`
- Final artifact commit: `f728f090cd6436f161134915c64f8099f705b8b6`
- Hub: <https://huggingface.co/datasets/Flowerly/modern-fake-reviews-glm>

The GLM corpus was topped up to 20,112 rows and then completed with 104 final
rows after replacing a flat 130-word acceptance ceiling with the proportional
ceiling documented above. Row-level attempt/version metadata was not retained.
`glm-5.2` is a provider alias, and its immutable provider revision is unknown.

## Dataset assembly

Implementation: `runs/build_modern_dataset.py`.

1. Select all genuine `OR` reviews from the source CSV.
2. Normalize both classes to one ASCII punctuation convention (NFKD plus an
   explicit smart-quote/dash/ellipsis map).
3. Remove empty generated text, exact generated duplicates, and generated text
   exactly matching a human review after case folding.
4. Balance by subsampling the human class with seed 1998 when necessary.
5. Combine both classes and shuffle with seed 1998.
6. Write `category,rating,label,text_`.

Hub publication uses an 80%/6%/14% train/validation/test split. It first applies
`train_test_split(test_size=0.2, seed=1998)` and then splits that 20% remainder
with `test_size=0.7, seed=1998`.

## Reproduction levels

- **Exact artifacts:** clone the recorded Git commits or verify the SHA-256
  values above. These bytes are reproducible.
- **Exact preprocessing:** rerunning the committed assembly code on the recorded
  raw artifacts is deterministic.
- **Statistically comparable regeneration:** rerun the prompt and sampling
  protocol against the named models.
- **Byte-identical regeneration:** not possible because API decoding was
  unseeded, provider aliases are mutable, and the exact LFM serving build was not
  retained.

This boundary is intentional and must remain explicit in the paper and dataset
cards.
