#!/bin/bash
# this script is used for end to end evaluation of pose refinement

# CONFIG=$1
# SCENE=$SCENE

# # run pose optimization
# python pose_refinement/pose_optimization.py --config $CONFIG --scene $SCENE

# run evaluation
# prepare data for nerf
python scripts/scannet2transform.py \
        --scene_folder pose_refinement/output/debug_600_5_loftr_sequential/scene0575_00/scannet \
        --scaled_image 
python scripts/scannet2nerf.py \
        --scene_folder pose_refinement/output/debug_600_5_loftr_sequential/scene0575_00/scannet \
        --transform_train pose_refinement/output/debug_600_5_loftr_sequential/scene0575_00/scannet/transforms_train_scaled.json

# optimize and evaluate nerf
export ENV_WORKSTATION_NAME=env

ROOT_DIR=$(pwd)/pose_refinement/output/debug_600_5_loftr_sequential/scene0575_00/scannet
python scripts/train_nerf.py --root $ROOT_DIR