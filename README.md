# DreamerV3 + ManiSkill-HAB

## Setup

**1. Install dependencies**

```bash
bash install.sh
```

**2. Download NVIDIA userspace drivers**

```bash
mkdir -p $HOME/nvidia-userspace
cd $HOME/nvidia-userspace
wget https://us.download.nvidia.com/tesla/570.133.20/NVIDIA-Linux-x86_64-570.133.20.run
```

**3. Create data folder and home alias**

```bash
mkdir -p /mnt/data/$USER
ln -sfn /mnt/data/$USER $HOME/mnt_data
```

**4. Download simulation assets**

```bash
export MS_ASSET_DIR=/mnt/data/$USER
mkdir -p $MS_ASSET_DIR/output

for dataset in ycb ReplicaCAD ReplicaCADRearrange; do
    python -m mani_skill.utils.download_asset "$dataset"
done
```

**5. Submit training job**

```bash
sbatch run_haw.sh
```
