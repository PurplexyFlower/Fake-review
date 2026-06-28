#!/usr/bin/env bash
# Rs-QLoRA revision — environment setup on a fresh Linux GPU instance (H100/H200)
# via uv. Mirrors the pinned stack that works locally (torch<2.11 for unsloth).
#
#   bash cloud/setup.sh
#
# Gemma-3-270m is gated on the Hub: `export HF_TOKEN=hf_...` (and accept the
# license once) before running cloud/run_all.sh, or drop it from --models.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

# 1) uv
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 2) venv (Python 3.12 to match local)
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
PIP="uv pip install"

# 3) torch first, pinned, from the CUDA 12.8 index (unsloth needs torch<2.11)
$PIP "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128

# 4) core stack — versions matched to the frozen protocol
$PIP "transformers==4.56.2" "peft==0.19.1" "bitsandbytes==0.49.2" \
     "datasets>=4.0,<5" "accelerate>=1.0" "trl==0.22.2" \
     "scikit-learn" "scipy" "pyarrow" "sentencepiece" "protobuf" \
     "huggingface_hub" "hf_transfer" "openai"

# 5) unsloth + zoo, then RE-PIN the things it may have bumped
$PIP "unsloth==2026.6.3" "unsloth_zoo"
$PIP --no-deps --force-reinstall "torch==2.10.0" \
     --index-url https://download.pytorch.org/whl/cu128
$PIP --no-deps --force-reinstall "transformers==4.56.2"

# 6) verify
python - <<'PY'
import torch, transformers, peft, bitsandbytes as bnb
print("torch       ", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
print("transformers", transformers.__version__, "| peft", peft.__version__,
      "| bnb", bnb.__version__)
import unsloth  # noqa: F401
print("unsloth import OK")
PY
echo ">> setup complete. Next:  bash cloud/run_all.sh"
