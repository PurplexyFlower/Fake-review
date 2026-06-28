"""E2: inference latency / throughput / peak VRAM for the headline Rs-QLoRA r64.

Loads the saved adapter on the 4-bit base, runs over the real test set:
 - batch-1 latency p50/p95 over N samples
 - throughput at batch 64
 - peak inference VRAM (allocated / reserved)
Writes results/inference_bench.json.
"""
import pyarrow  # noqa: F401
import json
import time
from pathlib import Path

import numpy as np

from _paths import headline_run_dir

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "fake reviews dataset.csv"
ADAPTER = headline_run_dir() / "adapter"
SPLIT_SEED = 1998
N_LATENCY = 1000


def main():
    import torch
    from datasets import load_dataset
    from unsloth import FastModel
    from transformers import AutoModelForSequenceClassification

    full = load_dataset("csv", data_files=str(DATA))["train"]
    sp = full.train_test_split(test_size=0.2, seed=SPLIT_SEED)
    sp2 = sp["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)
    test = sp2["test"]
    texts = test["text_"]

    model, tokenizer = FastModel.from_pretrained(
        model_name=str(ADAPTER), auto_model=AutoModelForSequenceClassification,
        max_seq_length=512, dtype=None, num_labels=2, load_in_4bit=True,
        full_finetuning=False,
    )
    FastModel.for_inference(model)
    model.eval()
    dev = next(model.parameters()).device

    def run_batch(batch_texts):
        enc = tokenizer(batch_texts, truncation=True, max_length=512,
                        padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            model(**enc)

    # warmup
    for _ in range(5):
        run_batch([texts[0]])
    torch.cuda.synchronize()

    # batch-1 latency
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for i in range(min(N_LATENCY, len(texts))):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run_batch([texts[i]])
        torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    peak_alloc_b1 = torch.cuda.max_memory_allocated() / 1024**3
    peak_resv_b1 = torch.cuda.max_memory_reserved() / 1024**3

    # throughput at batch 64
    torch.cuda.reset_peak_memory_stats()
    bs = 64
    n = min(64 * 20, len(texts))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(0, n, bs):
        run_batch(texts[i:i + bs])
    torch.cuda.synchronize()
    thr = n / (time.perf_counter() - t0)
    peak_alloc_b64 = torch.cuda.max_memory_allocated() / 1024**3
    peak_resv_b64 = torch.cuda.max_memory_reserved() / 1024**3

    out = {
        "n_latency": int(len(lat)),
        "latency_ms_batch1": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "mean": float(lat.mean()),
        },
        "throughput_samples_per_s_batch64": float(thr),
        "peak_vram_gb": {
            "batch1_alloc": float(peak_alloc_b1), "batch1_reserved": float(peak_resv_b1),
            "batch64_alloc": float(peak_alloc_b64), "batch64_reserved": float(peak_resv_b64),
        },
        "quant": "4-bit NF4 base + rs-LoRA r64 adapter",
    }
    (ROOT / "results" / "inference_bench.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
