#!/usr/bin/env bash
# Sets up the 'behavior' conda environment for DreamerV3 + BEHAVIOR-1K (OmniGibson).
# Must be run from the root of this repository: bash install_behavior1k.sh
#
# Delegates environment creation, Isaac Sim, OmniGibson, and BDDL3 installation
# to BEHAVIOR-1K/setup.sh, then installs this repo on top.
#
# Flags forwarded to BEHAVIOR-1K/setup.sh:
#   --accept-nvidia-eula      Accept NVIDIA Isaac Sim EULA non-interactively
#   --accept-dataset-tos      Accept BEHAVIOR dataset license non-interactively
#   --dataset                 Download BEHAVIOR-1K assets (~50-200 GB)
#   --cuda-version VERSION    Default: 12.8

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEHAVIOR_ROOT="$REPO_ROOT/BEHAVIOR-1K"

# ── 0. Parse flags ───────────────────────────────────────────────────────────
ACCEPT_NVIDIA_EULA=false
ACCEPT_DATASET_TOS=false
DOWNLOAD_DATASET=false
CUDA_VERSION="12.8"

for arg in "$@"; do
  case $arg in
    --accept-nvidia-eula)  ACCEPT_NVIDIA_EULA=true ;;
    --accept-dataset-tos)  ACCEPT_DATASET_TOS=true ;;
    --dataset)             DOWNLOAD_DATASET=true ;;
    --cuda-version=*)      CUDA_VERSION="${arg#*=}" ;;
    *) echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

[ -d "$BEHAVIOR_ROOT" ] || { echo "ERROR: BEHAVIOR-1K/ not found. Clone it first:"; echo "  git clone https://github.com/StanfordVL/BEHAVIOR-1K.git $BEHAVIOR_ROOT"; exit 1; }

# ── 1. Run BEHAVIOR-1K/setup.sh ──────────────────────────────────────────────
# Creates the 'behavior' conda env, installs Isaac Sim 5.1 via pip,
# OmniGibson, and BDDL3.
echo ">>> Running BEHAVIOR-1K/setup.sh..."

SETUP_FLAGS="--new-env behavior --omnigibson --bddl --cuda-version $CUDA_VERSION --accept-conda-tos"
[ "$ACCEPT_NVIDIA_EULA"  = true ] && SETUP_FLAGS="$SETUP_FLAGS --accept-nvidia-eula"
[ "$ACCEPT_DATASET_TOS"  = true ] && SETUP_FLAGS="$SETUP_FLAGS --accept-dataset-tos"
[ "$DOWNLOAD_DATASET"    = true ] && SETUP_FLAGS="$SETUP_FLAGS --dataset"

cd "$BEHAVIOR_ROOT"
# shellcheck disable=SC2086
bash setup.sh $SETUP_FLAGS
cd "$REPO_ROOT"

# ── 2. Activate the new env ──────────────────────────────────────────────────
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate behavior

# ── 3. Install this repo (DreamerV3) into the behavior env ──────────────────
# Only the embodied framework and dreamerv3 package are needed.
# JAX is NOT installed here — OmniGibson and JAX both need GPU and conflict
# on CUDA runtime if jax.prealloc is not disabled. The jax.prealloc=false flag
# in configs.yaml handles this at runtime; install JAX separately if required.
echo ">>> Installing DreamerV3 package..."
pip install -e "$REPO_ROOT"

# ── 4. Set asset path ────────────────────────────────────────────────────────
# Point OmniGibson to the same data directory as ManiSkill to keep all sim
# assets in one place. Assets download automatically on first environment creation.
echo ""
echo ">>> ACTION REQUIRED: Set your asset path before training."
echo "    Add the following to your ~/.bashrc (same \$MS_ASSET_DIR as ManiSkill):"
echo ""
echo "      export OMNIGIBSON_ASSET_PATH=/mnt/data/\$USER/og_assets"
echo "      export OMNIGIBSON_HEADLESS=1   # required on headless servers"
echo ""
echo ">>> Installation complete."
echo "    To train: conda activate behavior && python -m dreamerv3.main --configs behavior1k --task behavior1k_picking_up_trash"
echo ""
