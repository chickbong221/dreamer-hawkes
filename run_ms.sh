#!/bin/bash
#SBATCH --job-name=dreamerv3-ms
#SBATCH --partition=main
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

# Anh Duong
source "$HOME/ENTER/etc/profile.d/conda.sh"
conda activate "$HOME/ENTER/envs/dreamer"

# Activate conda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate dreamer

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

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

unset XLA_PYTHON_CLIENT_MEM_FRACTION

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_PushCube-v1 \
  --logdir $HOME/logdir/maniskill/$TIMESTAMP \
  --logger.wandb_name dreamerv3-PushCube

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_PickCube-v1 \
  --logdir $HOME/logdir/maniskill/$TIMESTAMP \
  --logger.wandb_name dreamerv3-PickCube

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_StackCube-v1 \
  --logdir $HOME/logdir/maniskill/$TIMESTAMP \
  --logger.wandb_name dreamerv3-StackCube

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_PegInsertionSide-v1 \
  --logdir $HOME/logdir/maniskill/$TIMESTAMP \
  --logger.wandb_name dreamerv3-PegInsertionSide

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_PushT-v1 \
  --logdir $HOME/logdir/maniskill/$TIMESTAMP \
  --logger.wandb_name dreamerv3-PushT

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_AnymalC-Reach-v1 \
  --logdir $HOME/logdir/maniskill/$TIMESTAMP \
  --logger.wandb_name dreamerv3-AnymalC-Reach

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_UnitreeG1TransportBox-v1 \
  --logdir $HOME/logdir/maniskill/$TIMESTAMP \
  --logger.wandb_name dreamerv3-UnitreeG1TransportBox

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
