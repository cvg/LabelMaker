#!/bin/bash

export SCENE_ID=47333462
WORKSPACE_DIR=/home/weders/scratch/scratch/LabelMaker/arkitscenes/$SCENE_ID


# run internimage
python models/internimage.py --workspace $WORKSPACE_DIR

# run ovseg
python models/ovseg.py --workspace $WORKSPACE_DIR

# run grounded sam
python models/grounded_sam.py --workspace $WORKSPACE_DIR

# run cmx
python models/omnidata.py --workspace $WORKSPACE_DIR
python models/hha_depth.py --workspace $WORKSPACE_DIR
python models/cmx.py --workspace $WORKSPACE_DIR

# run mask3d
python models/mask3d_inst.py --workspace $WORKSPACE_DIR
