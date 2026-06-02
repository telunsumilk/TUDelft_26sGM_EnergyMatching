#!/bin/bash
# setup.sh — install dependencies for EnergyMatching on a cloud GPU instance.
#
# Usage:
#   bash setup.sh

set -euo pipefail

WORKDIR="/workspace"
mkdir -p "$WORKDIR"
REPO_DIR="$WORKDIR/EnergyMatching"
VENV_DIR="$WORKDIR/venv"

echo "==> Work directory : $WORKDIR"
echo "==> Repo directory : $REPO_DIR"
echo "==> Venv directory : $VENV_DIR"

# --------------------------------------------------------------------------- #
# 1. System packages
# --------------------------------------------------------------------------- #
echo "==> Installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    git tmux rsync wget curl unzip \
    libgl1 libglib2.0-0   # needed by some OpenCV/torchvision ops
apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# 2. Clone repository (skip if already present)
# --------------------------------------------------------------------------- #
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "==> Cloning EnergyMatching..."
    git clone git@github.com:telunsumilk/TUDelft_26sGM_EnergyMatching.git "$REPO_DIR"
else
    echo "==> Repo already present, pulling latest..."
    git -C "$REPO_DIR" pull --ff-only
fi

# --------------------------------------------------------------------------- #
# 3. Virtual environment
# --------------------------------------------------------------------------- #
echo "==> Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# --------------------------------------------------------------------------- #
# 4. Python dependencies
# --------------------------------------------------------------------------- #
echo "==> Installing Python packages..."
pip install --upgrade pip wheel
pip install -r "$REPO_DIR/requirements.txt"

# --------------------------------------------------------------------------- #
# 5. Verify CUDA is visible
# --------------------------------------------------------------------------- #
echo "==> CUDA check..."
python - <<'EOF'
import torch
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA    : {torch.version.cuda}")
print(f"  GPUs    : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"    GPU {i}: {props.name}  {props.total_memory // 1024**3} GB")
EOF

# --------------------------------------------------------------------------- #
# 6. Print activation reminder
# --------------------------------------------------------------------------- #
echo ""
echo "========================================="
echo "  Setup complete."
echo "  Activate the venv before training:"
echo "    source $VENV_DIR/bin/activate"
echo "  Then cd into the experiment:"
echo "    cd $REPO_DIR/experiments/genmodel"
echo "========================================="
