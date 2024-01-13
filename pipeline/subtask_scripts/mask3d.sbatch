#!/usr/bin/bash

echo $LABELMAKER_REPO
echo $TARGET_DIR
echo $MASK3D_OUTPUT_FOLDER
echo $FLIP
echo $SEED

export SINGULARITYENV_MASK3D_OUTPUT_FOLDER=$MASK3D_OUTPUT_FOLDER
export SINGULARITYENV_FLIP=$FLIP
export SINGULARITYENV_SEED=$SEED

mkdir -p $TMPDIR/target
cp -r ${TARGET_DIR}/color $TMPDIR/target
cp -r ${TARGET_DIR}/depth $TMPDIR/target
cp -r ${TARGET_DIR}/intrinsic $TMPDIR/target
cp -r ${TARGET_DIR}/pose $TMPDIR/target
cp -r ${TARGET_DIR}/correspondence.json $TMPDIR/target
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
  bash /LabelMaker/pipeline/subtask_scripts/mask3d.sh

cp -r $TMPDIR/target/intermediate/* $TARGET_DIR/intermediate
