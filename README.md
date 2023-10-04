# LabelMaker

## Installation


## Setup Scene

### Download scene

```
export TRAINING_OR_VALIDATION=Training
export SCENE_ID=47333462
python 3rdparty/ARKitScenes/download_data.py raw --split $TRAINING_OR_VALIDATION --video_id $SCENE_ID --download_dir /tmp/ARKitScenes/ --raw_dataset_assets lowres_depth confidence lowres_wide.traj lowres_wide lowres_wide_intrinsics
```

### Convert scene to LabelMaker workspace

```
WORKSPACE_DIR=/home/weders/scratch/scratch/LabelMaker/arkitscenes/$SCENE_ID
python scripts/arkitscenes2labelmaker.py --scan_dir /tmp/ARKitScenes/raw/$TRAINING_OR_VALIDATION/$SCENE_ID --target_dir $WORKSPACE_DIR
```

### Run individual models

1. InternImage

```
python models/internimage.py --workspace $WORKSPACE_DIR
```

2. 