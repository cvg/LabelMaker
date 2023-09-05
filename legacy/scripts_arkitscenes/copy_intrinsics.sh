#!/bin/bash

DATA_PATH=/home/weders/scratch/scratch/scannetter/arkit/raw/Validation
SCENES=$(ls $DATA_PATH)

for sc in $SCENES; do
    echo $sc
    mkdir -p $DATA_PATH/$sc/intrinsic
    cp $DATA_PATH/$sc/intrinsics.txt $DATA_PATH/$sc/intrinsic/intrinsic_color.txt
    cp $DATA_PATH/$sc/intrinsics.txt $DATA_PATH/$sc/intrinsic/intrinsic_depth.txt


done;