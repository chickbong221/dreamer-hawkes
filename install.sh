#!/usr/bin/env bash
# Installation script for DreamerV3 + ManiSkill + ManiSkill-HAB
# Tested on Linux with an NVIDIA GPU and CUDA 12.
# Run from the root of this repository: bash install.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 0. Conda environment ────────────────────────────────────────────────────
# Source conda's shell hook so that `conda activate` works inside this script.
# This is necessary because conda activate is only available in interactive
# shells by default; bash scripts need the hook loaded explicitly.
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda create -n dreamer python=3.11 -y
conda activate dreamer

# ── 1. Clone ManiSkill3 ─────────────────────────────────────────────────────
# ManiSkill/ is listed in .gitignore and must be cloned separately.
# We use the official main branch (v3.0.1).
if [ ! -d "$REPO_ROOT/ManiSkill" ]; then
  git clone https://github.com/haosulab/ManiSkill.git "$REPO_ROOT/ManiSkill"
else
  echo ">>> ManiSkill/ already exists, skipping clone."
fi

# ── 2. Clone ManiSkill-HAB ──────────────────────────────────────────────────
# mshab/ is listed in .gitignore and must be cloned separately.
if [ ! -d "$REPO_ROOT/mshab" ]; then
  git clone https://github.com/arth-shukla/mshab.git "$REPO_ROOT/mshab"
else
  echo ">>> mshab/ already exists, skipping clone."
fi

# ── 3. Apply modified TD-MPC2 baseline ──────────────────────────────────────
# This repo ships a patched tdmpc2/ folder at the root.  It replaces the
# stock copy that lives inside ManiSkill's baselines so that ManiSkill picks
# up the changes automatically.  After the copy the root folder is removed to
# keep the working tree clean.
TDMPC2_SRC="$REPO_ROOT/tdmpc2"
TDMPC2_DST="$REPO_ROOT/ManiSkill/examples/baselines/tdmpc2"

if [ -d "$TDMPC2_SRC" ]; then
  echo ">>> Replacing ManiSkill/examples/baselines/tdmpc2 with patched version..."
  rm -rf "$TDMPC2_DST"
  cp -r "$TDMPC2_SRC" "$TDMPC2_DST"
  echo ">>> tdmpc2/ moved into ManiSkill."
else
  echo ">>> tdmpc2/ not found at repo root — skipping (already applied?)."
fi

# ── 4. Install PyTorch (CUDA 12.6) ──────────────────────────────────────────
# See https://pytorch.org/get-started/locally/ for other CUDA versions.
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install wandb
pip install wandb[media]

# ── 5. Install ManiSkill3 ───────────────────────────────────────────────────
# Installing from the local clone ensures the exact version used in this repo.
# Do NOT use `pip install mani_skill` from PyPI — the version may differ.
pip install -e "$REPO_ROOT/ManiSkill"

# ── 6. Vulkan (required for rendering) ──────────────────────────────────────
# ManiSkill uses Vulkan for GPU-accelerated rendering.
# On a headless Linux server install the loader and ICD:
#
#   sudo apt-get install -y libvulkan1 vulkan-tools
#
# On a desktop Linux system Vulkan is usually already present.
# Full guide: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan
echo ""
echo ">>> ACTION REQUIRED: Make sure Vulkan is installed on your system."
echo "    Headless server: sudo apt-get install -y libvulkan1 vulkan-tools"
echo "    Verify with:     vulkaninfo --summary"
echo ""

# ── 7. Install DreamerV3 dependencies ───────────────────────────────────────
# JAX with CUDA 12, elements, portal, scope, etc.
# numpy<2 is pinned because DMLab and MineRL require it.
pip install -U -r "$REPO_ROOT/requirements.txt"

# ── 8. Install DreamerV3 package (this repo) ────────────────────────────────
pip install -e "$REPO_ROOT"

# ── 9. Install ManiSkill-HAB ────────────────────────────────────────────────
# Registers PickSubtaskTrain-v0, PlaceSubtaskTrain-v0, OpenSubtaskTrain-v0,
# CloseSubtaskTrain-v0, NavigateSubtaskTrain-v0, and SequentialTask-v0.
pip install -e "$REPO_ROOT/mshab"

# ── 10. Fix OpenCV (headless server compatibility) ───────────────────────────
# ManiSkill pulls in opencv-python which conflicts with headless rendering.
# Replace all opencv variants with the headless build.
python -m pip uninstall -y \
  opencv-python \
  opencv-contrib-python \
  opencv-python-headless \
  opencv-contrib-python-headless
python -m pip install --no-cache-dir opencv-python-headless

echo ""
echo ">>> Installation complete."
echo "    Next steps for ManiSkill-HAB tasks: download assets (see README.md)."
echo ""
