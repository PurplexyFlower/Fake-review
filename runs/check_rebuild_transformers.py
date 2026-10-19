"""Non-result synthetic GPU checks for the three pinned encoder baselines."""
import json
import tempfile
from pathlib import Path
import pyarrow  # noqa: F401
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from rebuild_common import MODELS, OUT, save_json


def main():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    receipts = []
    for name, model_id, revision, batch_size in MODELS:
        set_seed(1998)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        extra = {'attn_implementation': 'sdpa'} if model_id.startswith('answerdotai/') else {}
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, num_labels=2, **extra).cuda()
        inputs = tokenizer(['A synthetic review example. ' * 200] * batch_size,
            padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
        labels = torch.arange(batch_size, device='cuda') % 2
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**inputs, labels=labels).loss
        assert torch.isfinite(loss), f'nonfinite loss: {name}'
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients), name
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            reference = model(**{key: value[:2] for key, value in inputs.items()}).logits.cpu()
        control = OUT / 'control'
        control.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='synthetic_encoder_', dir=control) as temporary:
            model.save_pretrained(temporary, safe_serialization=True)
            del model
            torch.cuda.empty_cache()
            restored = AutoModelForSequenceClassification.from_pretrained(temporary, **extra).cuda().eval()
            with torch.inference_mode():
                actual = restored(**{key: value[:2] for key, value in inputs.items()}).logits.cpu()
            assert torch.allclose(reference, actual, atol=1e-4, rtol=1e-4), name
            del restored
        receipts.append({'model': model_id, 'revision': revision, 'batch_size': batch_size,
            'sequence_length': inputs['input_ids'].shape[1], 'forward_backward_finite': True,
            'save_reload_verified': True, 'synthetic_peak_allocated_gib': peak})
        del inputs, gradients, loss, reference, actual
        torch.cuda.empty_cache()
        print('SYNTHETIC_ENCODER_CHECK_PASSED', name, flush=True)
    save_json(OUT / 'control/encoder_preflight.json', {
        'synthetic_smoke_check': True, 'not_a_reported_experiment': True,
        'gpu': torch.cuda.get_device_name(0), 'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
        'models': receipts}, exclusive=True)


if __name__ == '__main__':
    main()
