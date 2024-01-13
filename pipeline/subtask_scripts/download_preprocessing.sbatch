#!/usr/bin/bash

echo $LABELMAKER_REPO
echo $VIDEO_ID
echo $FOLD
echo $TARGET_DIR

# pass env variable to singluarity
export SINGULARITYENV_VIDEO_ID=$VIDEO_ID
export SINGULARITYENV_FOLD=$FOLD

module load gcc/11.4.0 cuda/12.1.1 eth_proxy

mkdir -p $TMPDIR/source mkdir -p $TMPDIR/target
mkdir -p $TMPDIR/.cache

singularity exec \
  --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints \
  --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker \
  --bind $LABELMAKER_REPO/models:/LabelMaker/models \
  --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts \
  --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline \
  --bind $TMPDIR/.cache:$HOME/.cache \
  --bind $TMPDIR/source:/source \
  --bind $TMPDIR/target:/target \
  /cluster/project/cvg/labelmaker/labelmaker_20231227.simg \
  bash /LabelMaker/pipeline/subtask_scripts/download_preprocessing.sh

# copy back
cp -r $TMPDIR/target/* $TARGET_DIR
