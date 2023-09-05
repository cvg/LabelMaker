#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=16000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weders@ethz.ch



set -e

scene=$1
echo $scene
scenedir=/cluster/project/cvg/blumh/scannet/$scene

mkdir $TMPDIR/$scene
for SUBDIR in color label-filt pred_internimage pred_internimage_flip pred_cmx pred_cmx_flip pred_ovseg_wn_nodef pred_ovseg_wn_nodef_flip pred_mask3d_rendered
do
	cp -r $scenedir/$SUBDIR $TMPDIR/$scene/
done

ls $TMPDIR/$scene

python scripts/segmentation_consensus.py --votes 3 --use_scannet $TMPDIR/$scene --scannet_weight 1 --no_ovseg --output_dir pred_consensus_no_ovseg
cp -r $TMPDIR/$scene/pred_consensus_no_ovseg $scenedir/

