#!/bin/bash

rsync -avP -essh --exclude '**/higres_depth' --exclude '**/vga_wide' --exclude '**/vga_wide_intrinsics' --exclude '**/lowres_depth' /home/weders/scratch/scratch/scannetter/arkit/raw/Validation/ euler_hermann:/cluster/project/cvg/blumh/arkitscenes_new