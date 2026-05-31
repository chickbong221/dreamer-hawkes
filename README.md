# DreamerV3 + ManiSkill-HAB + BEHAVIOR-1K

## Setup

### ManiSkill / ManiSkill-HAB

**1. Clone this repo and its dependencies**

```bash
git clone https://github.com/chickbong221/dreamerv3-maniskill-hab.git
cd dreamerv3-maniskill-hab
git clone https://github.com/haosulab/ManiSkill.git
git clone https://github.com/arth-shukla/mshab.git
```

**2. Install dependencies**

```bash
bash install.sh
```

**3. Download NVIDIA userspace drivers**

```bash
mkdir -p $HOME/nvidia-userspace
cd $HOME/nvidia-userspace
wget https://us.download.nvidia.com/tesla/570.133.20/NVIDIA-Linux-x86_64-570.133.20.run
```

**4. Submit training job**

| Script | Description |
|--------|-------------|
| `sbatch run_ms.sh` | Vanilla ManiSkill tasks (PushCube, PickCube, …) |
| `sbatch run_ms_haw.sh` | Same tasks with HawkesRSSM |
| `sbatch run_mshab.sh` | ManiSkill-HAB subtasks (standard RSSM) |
| `sbatch run_mshab_haw.sh` | ManiSkill-HAB subtasks with HawkesRSSM |

---

### BEHAVIOR-1K (OmniGibson)

> **Requires a separate conda env.** Isaac Sim conflicts with the `dreamer` env — `install_behavior1k.sh` creates a dedicated `behavior` env automatically.

**1. Clone BEHAVIOR-1K**

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```

**2. Install dependencies**

```bash
bash install_behavior1k.sh
# To also download BEHAVIOR-1K assets (~50-200 GB) in one shot:
bash install_behavior1k.sh --dataset --accept-nvidia-eula --accept-dataset-tos
```

**3. Set asset and headless paths**

```bash
export OMNIGIBSON_ASSET_PATH=/mnt/data/$USER/og_assets
export OMNIGIBSON_HEADLESS=1   # required on headless servers
```

Add both to `~/.bashrc` to persist.

**4. Submit training job**

```bash
sbatch run_behavior1k.sh
```

Or run directly:

```bash
conda activate behavior
python -m dreamerv3.main \
  --configs behavior1k \
  --task behavior1k_picking_up_trash
```

Task names map to activity names in `BEHAVIOR-1K/bddl3/bddl/activity_definitions/` (e.g. `washing_dishes`, `cleaning_floors`).

To override the robot/scene/sensor config, edit or copy `embodied/envs/behavior1k_cfg/default.yaml` and pass `--env.behavior1k.config_path <path>`.
