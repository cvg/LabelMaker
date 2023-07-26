#!/bin/bash

scene=$1
scenefolder=/cluster/project/cvg/blumh/replica/$scene
echo $scene
commonargsnogpu="--parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000"
commonargs="$commonargsnogpu --gpus=rtx_3090:1"

normals=$(sbatch $commonargs --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/normals_omnidata.py --replica True \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/omnidata_normal $scenefolder/'")

depth=$(sbatch $commonargs --time=04:00:00 --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/depthregression_omnidata.py --replica True \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/omnidata_depth $scenefolder/ && python scripts/depth2hha.py --replica True \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/hha $scenefolder/'")

seg1=$(sbatch $commonargs --time=04:00:00  --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_ovseg.py --replica --classes wn_nodef \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_ovseg_wn_nodef $scenefolder/'")
seg2=$(sbatch $commonargs --time=04:00:00 --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_ovseg.py --replica --classes wn_nodef --flip \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_ovseg_wn_nodef_flip $scenefolder/'")

seg3=$(sbatch $commonargs --time=04:00:00 --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_internimage.py --replica \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_internimage $scenefolder/'")
seg4=$(sbatch $commonargs --time=04:00:00 --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_internimage.py  --replica --flip \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_internimage_flip $scenefolder/'")

sam=$(sbatch $commonargs --time=22:00:00 --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_sam.py --replica \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_sam $scenefolder/'")

seg5=$(sbatch $commonargs -d afterany:$depth --time=04:00:00 --wrap="bash -c './scripts/copy_replica.bash $scene && cp -r $scenefolder/hha \$TMPDIR/$scene/ && python scripts/segmentation_cmx.py --replica \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_cmx $scenefolder/'")
seg6=$(sbatch $commonargs -d afterany:$depth --time=04:00:00 --wrap="bash -c './scripts/copy_replica.bash $scene && cp -r $scenefolder/hha \$TMPDIR/$scene/ && python scripts/segmentation_cmx.py --replica --flip \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_cmx_flip $scenefolder/'")

sbatch $commonargsnogpu --time=04:00:00 -d afterany:$seg1,afterany:$seg2,afterany:$seg3,afterany:$seg4,afterany:$seg5,afterany:$seg6 --wrap="bash scripts/replicaconsensus.bash $scene"
# to submit consensus only copy-paste this:
# sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/scannetconsensus.bash scene0458_00"
