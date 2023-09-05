#!/bin/bash

rsync -avP -essh --include '*/' --include '*/depth/*' --include '*/color/*' --include '*/pred_mask3d_rendered_ours/*' --include '*/intrinsic/*' --exclude '*' /home/weders/scratch/scratch/scannetter/arkit/raw/Validation/ euler_hermann:/cluster/project/cvg/blumh/arkitscenes_new