#!/usr/bin/bash

echo $LABELMAKER_REPO
echo $TARGET_DIR
echo $TARGET_RENDER_REL_PATH
echo $TARGET_RENDER_LABEL_SPACE
echo $TARGET_RENDER_VIDEO_NAME

export SINGULARITYENV_TARGET_RENDER_LABEL_SPACE=$TARGET_RENDER_LABEL_SPACE
export SINGULARITYENV_TARGET_RENDER_VIDEO_NAME=$TARGET_RENDER_VIDEO_NAME

mkdir -p $TMPDIR/target
cp -r ${TARGET_DIR}/color $TMPDIR/target
cp -r ${TARGET_DIR}/${TARGET_RENDER_REL_PATH} $TMPDIR/target/temp_folder
mkdir -p $TMPDIR/.cache

module load gcc/11.4.0 cuda/12.1.1 eth_proxy
singularity exec --nv \
  --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints \
  --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker \
  --bind $LABELMAKER_REPO/models:/LabelMaker/models \
  --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts \
  --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline \
  --bind $TMPDIR/.cache:$HOME/.cache \
  --bind $TMPDIR/target:/target \
  /cluster/project/cvg/labelmaker/labelmaker_20231227.simg \
  bash ./LabelMaker/pipeline/subtask_scripts/render.sh

cp -r $TMPDIR/target/${TARGET_RENDER_VIDEO_NAME} $TARGET_DIR/video
