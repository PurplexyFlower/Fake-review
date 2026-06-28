"""Phase 5: training peak-VRAM vs batch size (R1-8 'batch size dependence').

Re-creates the headline rs/r64 training setup and runs a few forward+backward
steps at batch {8,16,32,64}, recording peak allocated / reserved VRAM each time.
Writes results/vram_batch.json. Same frozen-protocol model/optimizer as train_run.
"""
import pyarrow  # noqa: F401
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "fake reviews dataset.csv"
SPLIT_SEED = 1998
BATCHES = [8, 16, 32, 64]


def main():
    import torch
    from datasets import load_dataset
    from unsloth import FastModel, is_bfloat16_supported
    from transformers import AutoModelForSequenceClassification
    import bitsandbytes as bnb

    full = load_dataset("csv", data_files=str(DATA))["train"]
    sp = full.train_test_split(test_size=0.2, seed=SPLIT_SEED)
    texts = sp["train"]["text_"]
    labels = [1 if x == "CG" else 0 for x in sp["train"]["label"]]

    results = {}
    for bs in BATCHES:
        torch.cuda.empty_cache()
        model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit",
            auto_model=AutoModelForSequenceClassification,
            max_seq_length=512, dtype=None, num_labels=2,
            full_finetuning=False, load_in_4bit=True,
        )
        model = FastModel.get_peft_model(
            model, r=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=128, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=1998,
            use_rslora=True, task_type="SEQ_CLS",
        )
        model.train()
        dev = next(model.parameters()).device
        opt = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad], lr=5e-6)

        torch.cuda.reset_peak_memory_stats()
        for step in range(4):
            sl = slice(step * bs, step * bs + bs)
            enc = tokenizer(list(texts[sl]), truncation=True, max_length=512,
                            padding=True, return_tensors="pt").to(dev)
            enc["labels"] = torch.tensor(labels[sl]).to(dev)
            out = model(**enc)
            out.loss.backward()
            opt.step(); opt.zero_grad()
        results[bs] = {
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        }
        print(f"batch {bs:3d}: alloc={results[bs]['peak_alloc_gb']:.2f} GB  "
              f"reserved={results[bs]['peak_reserved_gb']:.2f} GB")
        del model, opt
        torch.cuda.empty_cache()

    payload = {"note": "rs/r64 frozen protocol, 4-bit base, grad-ckpt unsloth, "
                       "adamw_8bit, bf16=%s, per-device batch only (accum=1)"
                       % is_bfloat16_supported(),
               "by_batch": results}
    (ROOT / "results" / "vram_batch.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
