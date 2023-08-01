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
scenedir=/cluster/project/cvg/blumh/arkitscenes/$scene

mkdir $TMPDIR/$scene
for SUBDIR in color pred_internimage pred_internimage_flip pred_cmx pred_cmx_flip pred_ovseg_wn_nodef pred_ovseg_wn_nodef_flip pred_mask3d_rendered
do
	cp -r $scenedir/$SUBDIR $TMPDIR/$scene/
done

ls $TMPDIR/$scene

python scripts/segmentation_consensus.py --votes 4 $TMPDIR/$scene
cp -r $TMPDIR/$scene/pred_consensus_noscannet $scenedir/

