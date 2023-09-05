#!/bin/bash

# This script is used to preprocess the ARKit dataset.
DATA_PATH=/home/weders/scratch/scratch/scannetter/arkit/raw/Validation
SCENES=$(ls $DATA_PATH)

# SCENES='42445991 42897688'

for SCENE in $SCENES; do
    echo "Processing scene $SCENE"
    python arkit2scannet_full.py $DATA_PATH/$SCENE
done