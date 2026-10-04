#!/usr/bin/env bash
set -euo pipefail
TASK_REPO=/home/mlw0719/doit_reliability
TASK_STORAGE=/home/mlw0719/doit_reliability_storage
exec > >(tee -a "$TASK_STORAGE/setup.log") 2>&1
set -x
date -u
/home/mlw0719/.local/bin/uv venv --python /home/mlw0719/miniconda3/bin/python3.12 "$TASK_REPO/.venv"
/home/mlw0719/.local/bin/uv pip install --python "$TASK_REPO/.venv/bin/python" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
/home/mlw0719/.local/bin/uv pip install --python "$TASK_REPO/.venv/bin/python" numpy==1.26.4 diffusers==0.32.2 transformers==4.49.0 accelerate==1.4.0 huggingface_hub==0.36.0 safetensors==0.8.0 Pillow==12.3.0 scipy==1.17.1 h5py==3.14.0 pip
/home/mlw0719/.local/bin/uv pip check --python "$TASK_REPO/.venv/bin/python"
"$TASK_REPO/.venv/bin/python" - <<'PY'
import sys
import torch, torchvision, numpy, diffusers, transformers, accelerate, huggingface_hub, safetensors, PIL, scipy, h5py
print('Python:', sys.version)
for module in (torch, torchvision, numpy, diffusers, transformers, accelerate, huggingface_hub, safetensors, PIL, scipy, h5py):
    print(module.__name__, module.__version__)
print('CUDA build:', torch.version.cuda)
PY
"$TASK_REPO/.venv/bin/python" -m pip check
cp /home/mlw0719/.cache/emb_reader/sa_0_4_vit_l_14_linear.pth "$TASK_STORAGE/models/sa_0_4_vit_l_14_linear.pth"
date -u
