#!/bin/bash
#SBATCH --job-name=dreamerv3-b1k
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

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavior

# Move to project directory
cd $HOME/projects/dreamer-hawkes

export WANDB_API_KEY="b1d6eed8871c7668a889ae74a621b5dbd2f3b070"
export OMNIGIBSON_ASSET_PATH=/mnt/data/tuannl/og_assets
export OMNIGIBSON_HEADLESS=1
export OMNI_KIT_ACCEPT_EULA=YES

# Print initial GPU state
nvidia-smi

# Monitor GPU in background
nvidia-smi -l 100 > $HOME/output/gpu_${SLURM_JOB_ID}.log &
GPU_MONITOR_PID=$!

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

python -m dreamerv3.main \
  --configs behavior1k \
  --task behavior1k_picking_up_trash \
  --logdir $HOME/logdir/behavior1k/$TIMESTAMP \
  --logger.wandb_name dreamerv3-b1k-picking-up-trash

# python -m dreamerv3.main \
#   --configs behavior1k \
#   --task behavior1k_washing_dishes \
#   --logdir $HOME/logdir/behavior1k/$TIMESTAMP \
#   --logger.wandb_name dreamerv3-b1k-washing-dishes

# python -m dreamerv3.main \
#   --configs behavior1k \
#   --task behavior1k_cleaning_floors \
#   --logdir $HOME/logdir/behavior1k/$TIMESTAMP \
#   --logger.wandb_name dreamerv3-b1k-cleaning-floors

# Stop GPU monitor
kill $GPU_MONITOR_PID

echo "Job finished"
