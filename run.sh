#!/bin/bash

export SCENE_ID=47333462
WORKSPACE_DIR=/home/weders/scratch/scratch/LabelMaker/arkitscenes/$SCENE_ID

# run internimage
python models/internimage.py --workspace $WORKSPACE_DIR

# run ovseg
python models/ovseg.py --workspace $WORKSPACE_DIR

# run grounded sam
python models/grounded_sam.py --workspace $WORKSPACE_DIR

# run omnidata for cmx and lifting (to obtain normals)
python models/omnidata_depth.py --workspace $WORKSPACE_DIR
python models/omnidata_normal.py --workspace $WORKSPACE_DIR

python models/hha_depth.py --workspace $WORKSPACE_DIR
python models/cmx.py --workspace $WORKSPACE_DIR

# run mask3d
python models/mask3d_inst.py --workspace $WORKSPACE_DIR

# run consensus
python labelmaker/consensus.py --workspace $WORKSPACE_DIR


# change environment to lifting

# run lifting
bash labelmaker/lifting_3d/lifting.sh $WORKSPACE_DIR
