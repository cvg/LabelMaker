#!/bin/bash

rsync -azuP euler_hermann:/cluster/home/blumh/ScanNetter/outputs/ /home/weders/scratch/scratch/scannetter
rsync -azuP euler_hermann:/cluster/project/cvg/blumh/scannet/* /home/weders/scratch/scratch/scannetter
rsync -azuP euler_hermann:/cluster/project/cvg/blumh/replica/* /home/weders/scratch/scratch/scannetter
