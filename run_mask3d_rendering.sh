#!/bin/bash

DATA_DIR=/home/weders/scratch/scratch/scannetter/arkit/raw/Validation


SCENES=$(ls $DATA_DIR)
# SCENES=42897688
for SCENE in $SCENES
do
    echo "Processing scene $SCENE"
    python scripts/mask3dmesh.py $DATA_DIR/$SCENE --arkitscenes
done