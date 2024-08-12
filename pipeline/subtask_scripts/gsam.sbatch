#!/usr/bin/bash

echo $LABELMAKER_REPO
echo $TARGET_DIR
echo $GSAM_OUTPUT_FOLDER
echo $FLIP

export SINGULARITYENV_GSAM_OUTPUT_FOLDER=$GSAM_OUTPUT_FOLDER
export SINGULARITYENV_FLIP=$FLIP

mkdir -p $TMPDIR/target
cp -r ${TARGET_DIR}/color $TMPDIR/target
cp -r ${TARGET_DIR}/correspondence.json $TMPDIR/target
mkdir -p $TMPDIR/.cache

module load stack/2024-06  gcc/12.2.0 cuda/12.1.1 eth_proxy

singularity exec --nv \
  --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints \
  --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker \
  --bind $LABELMAKER_REPO/models:/LabelMaker/models \
  --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts \
  --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline \
  --bind $TMPDIR/.cache:$HOME/.cache \
  --bind $TMPDIR/target:/target \
  /cluster/project/cvg/labelmaker/labelmaker_20231227.simg \
  bash /LabelMaker/pipeline/subtask_scripts/gsam.sh

cp -r $TMPDIR/target/intermediate/* $TARGET_DIR/intermediate
