# DreamerV3 + ManiSkill (GPU Parallel Rendering)

## Installation

Run the provided script from the repository root:

```bash
cd dreamerv3-maniskill-hab 
bash install.sh
```

**2. Download simulation assets** (ReplicaCAD scenes + rearrangement data, a few GB):

```bash
# Default install path: ~/.maniskill/data
# To change: export MS_ASSET_DIR=/your/path
for dataset in ycb ReplicaCAD ReplicaCADRearrange; do
    python -m mani_skill.utils.download_asset "$dataset"
done
```

