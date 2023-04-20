#!/bin/bash
# this script is used for end to end evaluation of pose refinement

# CONFIG=$1
# SCENE=$2


# # run pose optimization
# python pose_refinement/pose_optimization.py --config $CONFIG --scene $SCENE

wandb login --relogin 71de3ef4e4d319b8a54aae9f70862e8a126d880e

EXPRIMENT_DIR=pose_refinement/output/scannet_pose_refinement_downsample_5_overlap_loftr_no_icp/scene0575_00/scannet

# # # run evaluation on raw poses
# # # prepare data for nerf
python scripts/scannet2transform.py \
       $EXPRIMENT_DIR \
        --pose_mode pose_raw

python scripts/scannet2nerf.py \
        --scene_folder $EXPRIMENT_DIR \
        --transform_train $EXPRIMENT_DIR/transforms_train_scaled.json

# optimize and evaluate nerf
export ENV_WORKSTATION_NAME=env

ROOT_DIR=$(pwd)/$EXPRIMENT_DIR
python scripts/train_nerf.py \
        --root_dir $ROOT_DIR \
        --rendering_dir $ROOT_DIR/../nerf_raw/renderings

# # # run evaluation on pgo poses
# # # prepare data for nerf
python scripts/scannet2transform.py \
       $EXPRIMENT_DIR \
        --pose_mode pose_pgo

python scripts/scannet2nerf.py \
        --scene_folder $EXPRIMENT_DIR \
        --transform_train $EXPRIMENT_DIR/transforms_train_scaled.json

# optimize and evaluate nerf
export ENV_WORKSTATION_NAME=env

ROOT_DIR=$(pwd)/$EXPRIMENT_DIR
python scripts/train_nerf.py \
        --root_dir $ROOT_DIR \
        --rendering_dir $ROOT_DIR/../nerf_pgo/renderings

# # # run evaluation on ba poses
# # # prepare data for nerf
python scripts/scannet2transform.py \
        $EXPRIMENT_DIR \
        --pose_mode pose_ba

python scripts/scannet2nerf.py \
        --scene_folder $EXPRIMENT_DIR \
        --transform_train $EXPRIMENT_DIR/transforms_train_scaled.json

# optimize and evaluate nerf
export ENV_WORKSTATION_NAME=env

ROOT_DIR=$(pwd)/$EXPRIMENT_DIR
python scripts/train_nerf.py \
        --root_dir $ROOT_DIR \
        --rendering_dir $ROOT_DIR/../nerf_ba/renderings