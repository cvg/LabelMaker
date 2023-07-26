#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --tmp=16000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=blumh@ethz.ch



set -e

scene=$1

mkdir $TMPDIR/$scene
for SUBDIR in rgb pred_internimage pred_internimage_flip pred_cmx pred_cmx_flip pred_mask3d_rendered pred_ovseg_wn_nodef pred_ovseg_wn_nodef_flip
do
	cp -r /cluster/project/cvg/blumh/replica/$scene/$SUBDIR $TMPDIR/$scene/
done

python scripts/segmentation_consensus.py --replica True --votes 4 $TMPDIR/$scene

cp -r $TMPDIR/$scene/pred_wn_consensus /cluster/project/cvg/blumh/replica/$scene/

