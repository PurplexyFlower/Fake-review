"""Non-result GPU smoke check: forward/backward and adapter save/reload.

Synthetic examples are not an experiment and never enter the result ledger.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow  # must precede CUDA imports
from unsloth import FastModel
import torch
from transformers import AutoModelForSequenceClassification, set_seed

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/source_grouped_preflight'
BASE_REVISION = '10413c288cb9629acdf60b3e0229f3ba75efe413'


def main() -> None:
    set_seed(1998)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name='unsloth/qwen2.5-0.5b-instruct-unsloth-bnb-4bit',
        auto_model=AutoModelForSequenceClassification,
        max_seq_length=512, dtype=None, num_labels=2, load_in_4bit=True,
        full_finetuning=False, id2label={0: 'OR', 1: 'CG'},
        label2id={'OR': 0, 'CG': 1},
        revision=BASE_REVISION,
    )
    model = FastModel.get_peft_model(
        model, r=64, lora_alpha=128, lora_dropout=0, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        use_gradient_checkpointing='unsloth', random_state=1998,
        use_rslora=True, loftq_config=None, task_type='SEQ_CLS',
    )
    model.peft_config['default'].revision = BASE_REVISION
    batch = tokenizer(['The item arrived late but works well.',
                       'The product exceeded expectations and is easy to use.'],
                      truncation=True, max_length=512, padding=True,
                      return_tensors='pt').to(next(model.parameters()).device)
    labels = torch.tensor([0, 1], device=next(model.parameters()).device)
    model.train()
    loss = model(**batch, labels=labels).loss
    assert torch.isfinite(loss).item(), 'Nonfinite forward loss'
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.requires_grad and parameter.grad is not None]
    assert gradients, 'No adapter/head gradients'
    assert all(torch.isfinite(grad).all().item() for grad in gradients), 'Nonfinite gradients'
    loss_value = float(loss.detach())
    model.zero_grad(set_to_none=True)
    FastModel.for_inference(model)
    model.eval()
    with torch.no_grad():
        reference = model(**batch).logits.float().cpu()
    adapter = OUTPUT / 'adapter'
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    base_revision = getattr(model.config, '_commit_hash', None)
    del model
    torch.cuda.empty_cache()
    restored, _ = FastModel.from_pretrained(
        model_name=str(adapter), auto_model=AutoModelForSequenceClassification,
        max_seq_length=512, dtype=None, num_labels=2, load_in_4bit=True,
        full_finetuning=False,
    )
    FastModel.for_inference(restored)
    restored.eval()
    with torch.no_grad():
        actual = restored(**batch).logits.float().cpu()
    difference = float((reference - actual).abs().max())
    assert torch.allclose(reference, actual, atol=0.02, rtol=0.02), (
        f'Adapter reload changed logits: {difference}')
    record = {'synthetic_smoke_check': True, 'not_a_reported_experiment': True,
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'base_revision': base_revision, 'loss': loss_value,
              'adapter_reload_max_logit_difference': difference}
    (OUTPUT / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('SOURCE-GROUPED MODEL FORWARD/BACKWARD/SAVE/RELOAD VERIFIED')


if __name__ == '__main__':
    main()
