import pyarrow  # noqa: F401  (must precede unsloth/torch on Windows)

import numpy as np
import torch
from datasets import load_dataset
from unsloth import FastModel
from transformers import AutoModelForSequenceClassification

seed_value = 1998
id2label = {0: "OR", 1: "CG"}
label2id = {"OR": 0, "CG": 1}

# --- Rebuild the EXACT test split from the notebook (same ops, same seed) ---
dataset = load_dataset("csv", data_files="dataset/fake reviews dataset.csv")
dataset = dataset["train"].train_test_split(test_size=0.2, seed=seed_value)
dataset_test = dataset["test"].train_test_split(test_size=0.7, seed=seed_value)
test = dataset_test["test"]               # 5661 rows, matches the notebook
print(f"Test rows: {len(test)}")

# --- Load the saved best model (== checkpoint-1300) ---
model, tokenizer = FastModel.from_pretrained(
    model_name="model",
    auto_model=AutoModelForSequenceClassification,
    max_seq_length=512,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    load_in_4bit=True,
    full_finetuning=False,
)
FastModel.for_inference(model)
model.eval()
device = model.device

texts = test["text_"]
true_ids = np.array([label2id[l] for l in test["label"]])

# --- Batched inference ---
preds, confs, cg_probs = [], [], []
B = 64
with torch.no_grad():
    for i in range(0, len(texts), B):
        batch = texts[i:i + B]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        logits = model(**enc).logits.float()
        prob = torch.softmax(logits, dim=-1)
        p = prob.argmax(dim=-1)
        preds.extend(p.cpu().tolist())
        confs.extend(prob.max(dim=-1).values.cpu().tolist())
        cg_probs.extend(prob[:, 1].cpu().tolist())

preds = np.array(preds)
acc = (preds == true_ids).mean()
tp = int(((preds == 1) & (true_ids == 1)).sum())
tn = int(((preds == 0) & (true_ids == 0)).sum())
fp = int(((preds == 1) & (true_ids == 0)).sum())
fn = int(((preds == 0) & (true_ids == 1)).sum())
print(f"\nAccuracy: {acc:.4f}   errors: {(preds != true_ids).sum()} / {len(preds)}")
print(f"Confusion [[OR->OR {tn}, OR->CG {fp}], [CG->OR {fn}, CG->CG {tp}]]")

# --- Collect and save the misclassified rows ---
import csv
wrong = np.where(preds != true_ids)[0]
out = "misclassified_test.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["row", "true", "pred", "pred_conf", "p_CG", "category", "rating", "text"])
    for r in wrong:
        w.writerow([int(r), id2label[true_ids[r]], id2label[preds[r]],
                    f"{confs[r]:.4f}", f"{cg_probs[r]:.4f}",
                    test["category"][int(r)], test["rating"][int(r)],
                    texts[int(r)]])
print(f"\nWrote {len(wrong)} misclassified rows to {out}\n")

# --- Print them, hardest-first (highest confidence in the wrong answer) ---
order = sorted(wrong, key=lambda r: confs[r], reverse=True)
for n, r in enumerate(order, 1):
    t = texts[int(r)].replace("\n", " ")
    if len(t) > 240:
        t = t[:240] + "…"
    print(f"{n:>2}. true={id2label[true_ids[r]]} pred={id2label[preds[r]]} "
          f"conf={confs[r]:.3f} p_CG={cg_probs[r]:.3f}\n    {t}\n")
