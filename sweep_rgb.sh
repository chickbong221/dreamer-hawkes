#!/usr/bin/env bash
set -euo pipefail

python -m dreamerv3.main \
  --configs maniskill_rgb size50m \
  --task maniskill_PushCube-v1 \
  --batch_size 8 \
  --logger.wandb_name dreamerv3-PushCube-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb size50m \
  --task maniskill_PickCube-v1 \
  --batch_size 8 \
  --logger.wandb_name dreamerv3-PickCube-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb size50m \
  --task maniskill_StackCube-v1 \
  --batch_size 8 \
  --logger.wandb_name dreamerv3-StackCube-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb size50m \
  --task maniskill_PegInsertionSide-v1 \
  --batch_size 8 \
  --logger.wandb_name dreamerv3-PegInsertionSide-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb size50m \
  --task maniskill_PushT-v1 \
  --batch_size 8 \
  --logger.wandb_name dreamerv3-PushT-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb size50m \
  --task maniskill_AnymalC-Reach-v1 \
  --batch_size 8 \
  --logger.wandb_name dreamerv3-AnymalC-Reach-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb size50m \
  --task maniskill_UnitreeG1TransportBox-v1 \
  --batch_size 8 \
  --logger.wandb_name dreamerv3-UnitreeG1TransportBox-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  -  --env.maniskill.control_mode pd_joint_
python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_PickSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task tidy_house \
  --env.maniskill.mshab_split train \
  --logger.wandb_name dreamerv3-mshab-pick-tidy-house-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_PickSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task prepare_groceries \
  --env.maniskill.mshab_split train \
  --logger.wandb_name dreamerv3-mshab-pick-prepare-groceries-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_PickSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task set_table \
  --env.maniskill.mshab_split train \
  --logger.wandb_name dreamerv3-mshab-pick-set-table-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_PlaceSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task tidy_house \
  --env.maniskill.mshab_split train \
  --logger.wandb_name dreamerv3-mshab-place-tidy-house-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_PlaceSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task prepare_groceries \
  --env.maniskill.mshab_split train \
  --logger.wandb_name dreamerv3-mshab-place-prepare-groceries-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_PlaceSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task set_table \
  --env.maniskill.mshab_split train \
  --logger.wandb_name dreamerv3-mshab-place-set-table-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_OpenSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task set_table \
  --env.maniskill.mshab_split train \
  --env.maniskill.mshab_obj kitchen_counter \
  --logger.wandb_name dreamerv3-mshab-open-set-table-kitchen-counter-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_OpenSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task set_table \
  --env.maniskill.mshab_split train \
  --env.maniskill.mshab_obj fridge \
  --logger.wandb_name dreamerv3-mshab-open-set-table-fridge-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_CloseSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task set_table \
  --env.maniskill.mshab_split train \
  --env.maniskill.mshab_obj kitchen_counter \
  --logger.wandb_name dreamerv3-mshab-close-set-table-kitchen-counter-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_CloseSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task set_table \
  --env.maniskill.mshab_split train \
  --env.maniskill.mshab_obj fridge \
  --logger.wandb_name dreamerv3-mshab-close-set-table-fridge-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_NavigateSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task tidy_house \
  --env.maniskill.mshab_split train \
  --logger.wandb_name dreamerv3-mshab-navigate-tidy-house-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_NavigateSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task prepare_groceries \
  --env.maniskill.mshab_split train \
  --logger.wandb_name dreamerv3-mshab-navigate-prepare-groceries-rgb-42

python -m dreamerv3.main \
  --configs maniskill_rgb size100m mshab \
  --task maniskill_NavigateSubtaskTrain-v0 \
  --env.maniskill.control_mode pd_joint_delta_pos \
  --env.maniskill.mshab_task set_table \
  --env.maniskill.mshab_split train \
  --logger.wandb_name dreamerv3-mshab-navigate-set-table-rgb-42
