#!/bin/bash

DATA_PATH=/home/weders/scratch/scratch/scannetter

# for sc in scene0164_02 scene0458_00 scene0474_01 scene0518_00 scene0000_00; do
for sc in scene0474_01 scene0000_00; do
    echo $sc
    sc_conv=$(echo "$sc" | sed 's/\([a-zA-Z]\)\([0-9]\)/\1_\2/g')
    echo $sc_conv
    python scripts/agile3d_postprocess_v2.py $DATA_PATH/$sc/ $DATA_PATH/interactive_dataset_fine/$sc_conv/
done;