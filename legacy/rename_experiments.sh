#!/bin/bash

SCENES='scene0000_00 scene0164_02 scene0458_00 scene0474_01 scene0518_00'
DATA_PATH=/home/weders/scratch/scratch/scannetter



cp $DATA_PATH/scene0000_00/pred_sdfstudio_2023-07-30_112430_labels_3d.txt $DATA_PATH/scene0000_00/labelmaker_labels_3d.txt
cp $DATA_PATH/scene0164_02/pred_sdfstudio_2023-07-30_104700_labels_3d.txt $DATA_PATH/scene0164_02/labelmaker_labels_3d.txt
cp $DATA_PATH/scene0458_00/pred_sdfstudio_2023-07-30_104953_labels_3d.txt $DATA_PATH/scene0458_00/labelmaker_labels_3d.txt
cp $DATA_PATH/scene0474_01/pred_sdfstudio_2023-07-30_105014_labels_3d.txt $DATA_PATH/scene0474_01/labelmaker_labels_3d.txt
cp $DATA_PATH/scene0518_00/pred_sdfstudio_2023-07-30_104735_labels_3d.txt $DATA_PATH/scene0518_00/labelmaker_labels_3d.txt