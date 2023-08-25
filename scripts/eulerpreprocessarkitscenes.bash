#!/bin/bash

scene=$1
scenefolder=/cluster/project/cvg/blumh/arkitscenes_new/$scene
echo $scene
commonargsnogpu="--parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --mail-type=END,FAIL --mail-user=weders@ethz.ch"
commonargs="$commonargsnogpu --gpus=rtx_3090:1"

normals=$(sbatch $commonargs --wrap="bash -c './scripts/copy_arkitscenes.bash $scene && python scripts/normals_omnidata.py \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/omnidata_normal /cluster/project/cvg/blumh/arkitscenes_new/$scene/'")

depth=$(sbatch $commonargs --time=04:00:00 --wrap="bash -c './scripts/copy_arkitscenes.bash $scene && python scripts/depthregression_omnidata.py --scene \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/omnidata_depth /cluster/project/cvg/blumh/arkitscenes_new/$scene/ && python scripts/depth2hha.py \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/hha /cluster/project/cvg/blumh/arkitscenes_new/$scene/'")

# uncomment below
seg1=$(sbatch $commonargs --time=04:00:00  --wrap="bash -c './scripts/copy_arkitscenes.bash $scene && python scripts/segmentation_ovseg.py --classes wn_nodef \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_ovseg_wn_nodef /cluster/project/cvg/blumh/arkitscenes_new/$scene/'")
seg2=$(sbatch $commonargs --time=04:00:00 --wrap="bash -c './scripts/copy_arkitscenes.bash $scene && python scripts/segmentation_ovseg.py --classes wn_nodef --flip \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_ovseg_wn_nodef_flip /cluster/project/cvg/blumh/arkitscenes_new/$scene/'")

seg3=$(sbatch $commonargs --time=04:00:00 --wrap="bash -c './scripts/copy_arkitscenes.bash $scene && python scripts/segmentation_internimage.py \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_internimage /cluster/project/cvg/blumh/arkitscenes_new/$scene/'")
seg4=$(sbatch $commonargs --time=04:00:00 --wrap="bash -c './scripts/copy_arkitscenes.bash $scene && python scripts/segmentation_internimage.py --flip \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_internimage_flip /cluster/project/cvg/blumh/arkitscenes_new/$scene/'")

sam=$(sbatch $commonargs --time=22:00:00 --wrap="bash -c './scripts/copy_arkitscenes.bash $scene && python scripts/segmentation_sam.py \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_sam /cluster/project/cvg/blumh/arkitscenes_new/$scene/'")

#  -d afterany:$depth 
seg5=$(sbatch $commonargs -d afterany:$depth --time=04:00:00 --wrap="bash -c './scripts/copy_arkitscenes.bash $scene && cp -r $scenefolder/hha \$TMPDIR/$scene/ && python scripts/segmentation_cmx.py \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_cmx /cluster/project/cvg/blumh/arkitscenes_new/$scene/'")
seg6=$(sbatch $commonargs -d afterany:$depth --time=04:00:00 --wrap="bash -c './scripts/copy_arkitscenes.bash $scene && cp -r $scenefolder/hha \$TMPDIR/$scene/ && python scripts/segmentation_cmx.py --flip \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_cmx_flip $scenefolder/'")

sbatch $commonargsnogpu --time=04:00:00 -d afterany:$seg1,afterany:$seg2,afterany:$seg3,afterany:$seg4,afterany:$seg5,afterany:$seg6 --wrap="bash scripts/arkitscenesconsensus.bash $scene"


# to submit consensus only copy-paste this:
# sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/scannetconsensus.bash scene0458_00"
