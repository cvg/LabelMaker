#!/bin/bash

scene=$1
scenefolder=/home/weders/scratch/scratch/scannetter/arkit/raw/Validation/$scene
echo $scene

# python normals_omnidata.py $scenefolder
# python depthregression_omnidata.py --scene $scenefolder
# python depth2hha.py $scenefolder
# python segmentation_ovseg.py --classes wn_nodef $scenefolder # this has to run again
# python segmentation_ovseg.py --classes wn_nodef --flip $scenefolder # this has to run again
# python segmentation_internimage.py $scenefolder
# python segmentation_internimage.py --flip $scenefolder # this has to run again
# python segmentation_sam.py $scenefolder

# python segmentation_cmx.py $scenefolder
# python segmentation_cmx.py --flip $scenefolder

python segmentation_consensus.py --votes 2 $scenefolder

# to submit consensus only copy-paste this:
# sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scannetconsensus.bash scene0458_00"
