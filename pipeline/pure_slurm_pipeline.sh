#!/bin/bash

video_id=$1
fold=$2

target_dir=/cluster/project/cvg/labelmaker/ARKitScene_LabelMaker/$fold/$video_id
mkdir -p $target_dir

LABELMAKER_REPO=/cluster/home/guanji/LabelMaker

commonargs=" --parsable -n 1"

download_preprocessing_args="--cpus-per-task=1 --mem-per-cpu=2G"
download_preprocessing_time="20:00"

download_preprocessing_flag=$(sbatch $commonargs --time=$download_preprocessing_time --wrap="module load gcc/11.4.0 cuda/12.1.1 eth_proxy && mkdir -p \$TMPDIR/source && mkdir -p \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/env_v2:/LabelMaker/env_v2 --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/source:/source --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/arkitscene_download.py --video_id $video_id --download_dir /source && python ./scripts/arkitscenes2labelmaker.py --scan_dir /source/raw/$fold/$video_id --target_dir /target' && cp -r \TMPDIR/target/*  $target_dir")
