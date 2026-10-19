#!/usr/bin/env bash
# Dedicated isolated setup: never modifies the template's /venv/main.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"
command -v uv >/dev/null || { echo "uv is required" >&2; exit 2; }
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
uv pip install --constraint cloud/source_grouped_constraints.txt \
  'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
uv pip install --constraint cloud/source_grouped_constraints.txt \
  'transformers==4.56.2' 'peft==0.19.1' 'bitsandbytes==0.49.2' \
  'trl==0.22.2' 'unsloth==2026.6.3' 'unsloth_zoo==2026.6.3' \
  'datasets>=4,<5' 'accelerate>=1' scikit-learn scipy pyarrow \
  sentencepiece protobuf huggingface_hub hf_transfer
uv pip check
python - <<'PY'
import pyarrow
import unsloth
import torch
import transformers
import peft
import bitsandbytes
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 1, 'Expected one visible GPU'
assert torch.cuda.is_bf16_supported(), 'Expected BF16 support'
value = torch.ones((128, 128), device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(value @ value).all().item(), 'GPU matmul failed'
print('GPU:', torch.cuda.get_device_name(0))
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('transformers:', transformers.__version__, 'peft:', peft.__version__)
print('bitsandbytes:', bitsandbytes.__version__)
print('SOURCE-GROUPED GPU ENVIRONMENT VERIFIED')
PY
mkdir -p results
uv pip freeze > results/source_grouped_environment.lock.txt
