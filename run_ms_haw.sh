#!/bin/bash
#SBATCH --job-name=dreamerv3-ms-haw
#SBATCH --partition=main
#SBATCH --nodelist=worker-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/output/%x_%j.out
#SBATCH --error=/home/%u/output/%x_%j.err

echo "================================="
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "================================="

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamer

export NVIDIA_USERSPACE_VERSION=570.133.20
export NVIDIA_USERSPACE_DIR=$HOME/nvidia-userspace/NVIDIA-Linux-x86_64-${NVIDIA_USERSPACE_VERSION}

cd "$NVIDIA_USERSPACE_DIR"

ln -sf libGLX_nvidia.so.${NVIDIA_USERSPACE_VERSION} libGLX_nvidia.so.0
ln -sf libEGL_nvidia.so.${NVIDIA_USERSPACE_VERSION} libEGL_nvidia.so.0

cat > "$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json" <<EOF
{
    "file_format_version": "1.0.1",
    "ICD": {
        "library_path": "$NVIDIA_USERSPACE_DIR/libEGL_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
EOF

export LD_LIBRARY_PATH=$NVIDIA_USERSPACE_DIR:${LD_LIBRARY_PATH:-}
export VK_DRIVER_FILES=$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json
export VK_ICD_FILENAMES=$NVIDIA_USERSPACE_DIR/nvidia_icd_egl.json

vulkaninfo --summary

# Move to project directory
cd $HOME/projects/dreamer-hawkes

export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"

# Print initial GPU state
nvidia-smi

# Monitor GPU every 20 seconds in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

unset XLA_PYTHON_CLIENT_MEM_FRACTION

# ── Sparse reward runs (all tasks) ────────────────────────────────────────────

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_PushCube-v1 \
  --run.steps 1e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/PushCube-sparse \
  --env.maniskill.reward_mode sparse \
  --logger.wandb_name hawkes-dreamerv3-state-PushCube-sparse

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_PickCube-v1 \
  --run.steps 1e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/PickCube-sparse \
  --env.maniskill.reward_mode sparse \
  --logger.wandb_name hawkes-dreamerv3-state-PickCube-sparse

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_StackCube-v1 \
  --run.steps 4e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/StackCube-sparse \
  --env.maniskill.reward_mode sparse \
  --logger.wandb_name hawkes-dreamerv3-state-StackCube-sparse

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_PegInsertionSide-v1 \
  --run.steps 4e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/PegInsertionSide-sparse \
  --env.maniskill.reward_mode sparse \
  --logger.wandb_name hawkes-dreamerv3-state-PegInsertionSide-sparse

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_PushT-v1 \
  --run.steps 2e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/PushT-sparse \
  --env.maniskill.reward_mode sparse \
  --logger.wandb_name hawkes-dreamerv3-state-PushT-sparse

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_AnymalC-Reach-v1 \
  --run.steps 2e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/AnymalC-Reach-sparse \
  --env.maniskill.reward_mode sparse \
  --logger.wandb_name hawkes-dreamerv3-state-AnymalC-Reach-sparse

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_UnitreeG1TransportBox-v1 \
  --run.steps 2e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/UnitreeG1TransportBox-sparse \
  --env.maniskill.reward_mode sparse \
  --logger.wandb_name hawkes-dreamerv3-state-UnitreeG1TransportBox-sparse

# ── Dense reward runs (all tasks) ─────────────────────────────────────────────

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_PushCube-v1 \
  --run.steps 1e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/PushCube-dense \
  --env.maniskill.reward_mode normalized_dense \
  --logger.wandb_name hawkes-dreamerv3-state-PushCube-dense

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_PickCube-v1 \
  --run.steps 1e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/PickCube-dense \
  --env.maniskill.reward_mode normalized_dense \
  --logger.wandb_name hawkes-dreamerv3-state-PickCube-dense

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_StackCube-v1 \
  --run.steps 4e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/StackCube-dense \
  --env.maniskill.reward_mode normalized_dense \
  --logger.wandb_name hawkes-dreamerv3-state-StackCube-dense

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_PegInsertionSide-v1 \
  --run.steps 4e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/PegInsertionSide-dense \
  --env.maniskill.reward_mode normalized_dense \
  --logger.wandb_name hawkes-dreamerv3-state-PegInsertionSide-dense

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_PushT-v1 \
  --run.steps 2e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/PushT-dense \
  --env.maniskill.reward_mode normalized_dense \
  --logger.wandb_name hawkes-dreamerv3-state-PushT-dense

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_AnymalC-Reach-v1 \
  --run.steps 2e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/AnymalC-Reach-dense \
  --env.maniskill.reward_mode normalized_dense \
  --logger.wandb_name hawkes-dreamerv3-state-AnymalC-Reach-dense

python -m dreamerv3.main \
  --configs maniskill_state hawkes \
  --task maniskill_UnitreeG1TransportBox-v1 \
  --run.steps 2e6 \
  --logdir $HOME/logdir/maniskill_state/$TIMESTAMP/UnitreeG1TransportBox-dense \
  --env.maniskill.reward_mode normalized_dense \
  --logger.wandb_name hawkes-dreamerv3-state-UnitreeG1TransportBox-dense

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
