#!/bin/bash

DATA_PATH=/home/weders/scratch/scratch/scannetter/arkit/

python $DATA_PATH/ARKitScenes/download_data.py raw --split Validation --video_id 42445991 --download_dir $DATA_PATH/. --raw_dataset_assets lowres_depth confidence
python $DATA_PATH/ARKitScenes/download_data.py raw --split Validation --video_id 42897688 --download_dir $DATA_PATH/. --raw_dataset_assets lowres_depth confidence