#!/usr/bin/bash

echo $LABELMAKER_REPO
echo $TARGET_DIR
echo $WANDB_ENTITY
echo $WANDB_API_KEY

rm -rf ${TARGET_DIR}/intermediate/sdfstudio_train

mkdir -p $TMPDIR/target
mkdir -p $TMPDIR/target/intermediate
cp -r ${TARGET_DIR}/intermediate/consensus $TMPDIR/target/intermediate
cp -r ${TARGET_DIR}/intermediate/depth_omnidata_1 $TMPDIR/target/intermediate
cp -r ${TARGET_DIR}/correspondence.json $TMPDIR/target
cp -r ${TARGET_DIR}/color $TMPDIR/target
cp -r ${TARGET_DIR}/depth $TMPDIR/target
cp -r ${TARGET_DIR}/intrinsic $TMPDIR/target
cp -r ${TARGET_DIR}/pose $TMPDIR/target
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
  --env WANDB_ENTITY=$WANDB_ENTITY \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  /cluster/project/cvg/labelmaker/labelmaker_20240812.simg \
  bash /LabelMaker/pipeline/subtask_scripts/sdfstudio_train.sh

cp -r $TMPDIR/target/intermediate/sdfstudio_preprocessing $TARGET_DIR/intermediate
cp -r $TMPDIR/target/intermediate/sdfstudio_train $TARGET_DIR/intermediate
