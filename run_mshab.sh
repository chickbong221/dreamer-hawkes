#!/bin/bash
#SBATCH --job-name=dreamerv3-mshab
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0
#SBATCH --output=/home/%u/mnt_data/output/%x_%j.out
#SBATCH --error=/home/%u/mnt_data/output/%x_%j.err

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
cd /home/$USERNAME/projects/dreamer-maniskill-hab

export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
export MS_ASSET_DIR=/mnt/data/$USERNAME

# Print initial GPU state
nvidia-smi

# Monitor GPU every 20 seconds in background
nvidia-smi -l 100 > /home/$USERNAME/mnt_data/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Generate timestamp properly
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

unset XLA_PYTHON_CLIENT_MEM_FRACTION

python -m dreamerv3.main \
  --configs maniskill_rgb mshab \
  --task maniskill_PlaceSubtaskTrain-v0 \
  --logdir /mnt/data/$USERNAME/logdir/maniskill/$TIMESTAMP \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task set_table \
  --logger.wandb_name dreamerv3-mshab-place-set-table

# python -m dreamerv3.main \
#   --configs maniskill_rgb mshab \
#   --task maniskill_OpenSubtaskTrain-v0 \
#   --logdir /mnt/data/$USERNAME/logdir/maniskill/$TIMESTAMP \
#   --env.maniskill.control_mode pd_joint_delta_pos \
#   --env.maniskill.mshab_task set_table \
#   --env.maniskill.mshab_obj kitchen_counter \
#   --logger.wandb_name dreamerv3-mshab-open-set-table-kitchen-counter

# python -m dreamerv3.main \
#   --configs maniskill_rgb mshab \
#   --task maniskill_OpenSubtaskTrain-v0 \
#   --logdir /mnt/data/$USERNAME/logdir/maniskill/$TIMESTAMP \
#   --env.maniskill.control_mode pd_joint_delta_pos \
#   --env.maniskill.mshab_task set_table \
#   --env.maniskill.mshab_obj fridge \
#   --logger.wandb_name dreamerv3-mshab-open-set-table-fridge

# python -m dreamerv3.main \
#   --configs maniskill_rgb mshab \
#   --task maniskill_CloseSubtaskTrain-v0 \
#   --logdir /mnt/data/$USERNAME/logdir/maniskill/$TIMESTAMP \
#   --env.maniskill.control_mode pd_joint_delta_pos \
#   --env.maniskill.mshab_task set_table \
#   --env.maniskill.mshab_obj kitchen_counter \
#   --logger.wandb_name dreamerv3-mshab-close-set-table-kitchen-counter

# python -m dreamerv3.main \
#   --configs maniskill_rgb mshab \
#   --task maniskill_CloseSubtaskTrain-v0 \
#   --logdir /mnt/data/$USERNAME/logdir/maniskill/$TIMESTAMP \
#   --env.maniskill.control_mode pd_joint_delta_pos \
#   --env.maniskill.mshab_task set_table \
#   --env.maniskill.mshab_obj fridge \
#   --logger.wandb_name dreamerv3-mshab-close-set-table-fridge

# python -m dreamerv3.main \
#   --configs maniskill_rgb mshab \
#   --task maniskill_NavigateSubtaskTrain-v0 \
#   --logdir /mnt/data/$USERNAME/logdir/maniskill/$TIMESTAMP \
#   --env.maniskill.control_mode pd_joint_delta_pos \
#   --env.maniskill.mshab_task tidy_house \
#   --logger.wandb_name dreamerv3-mshab-navigate-tidy-house

# python -m dreamerv3.main \
#   --configs maniskill_rgb mshab \
#   --task maniskill_NavigateSubtaskTrain-v0 \
#   --logdir /mnt/data/$USERNAME/logdir/maniskill/$TIMESTAMP \
#   --env.maniskill.control_mode pd_joint_delta_pos \
#   --env.maniskill.mshab_task prepare_groceries \
#   --logger.wandb_name dreamerv3-mshab-navigate-prepare-groceries

# python -m dreamerv3.main \
#   --configs maniskill_rgb mshab \
#   --task maniskill_NavigateSubtaskTrain-v0 \
#   --logdir /mnt/data/$USERNAME/logdir/maniskill/$TIMESTAMP \
#   --env.maniskill.control_mode pd_joint_delta_pos \
#   --env.maniskill.mshab_task set_table \
#   --logger.wandb_name dreamerv3-mshab-navigate-set-table

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
