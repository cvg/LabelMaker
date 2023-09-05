#!/bin/bash

scene=room_0_2
scenefolder=/cluster/project/cvg/blumh/replica/$scene
echo $scene
commonargs="--parsable -n 1 --cpus-per-task=8 --gpus=1 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00"

normals=$(sbatch $commonargs --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/normals_omnidata.py --replica True \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/omnidata_normal /cluster/project/cvg/blumh/replica/$scene/'")

depth=$(sbatch $commonargs --time=04:00:00 --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/depthregression_omnidata.py --replica True \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/omnidata_depth /cluster/project/cvg/blumh/replica/$scene/ && python scripts/depth2hha.py --replica True \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/hha /cluster/project/cvg/blumh/replica/$scene/'")

seg1=$(sbatch $commonargs --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_ovseg.py --replica --classes wn_nodef \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_ovseg_wn_nodef /cluster/project/cvg/blumh/replica/$scene/'")
seg2=$(sbatch $commonargs --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_ovseg.py --replica --classes wn_nodef --flip \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_ovseg_wn_nodef_flip /cluster/project/cvg/blumh/replica/$scene/'")

seg3=$(sbatch $commonargs --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_internimage.py --replica \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_internimage /cluster/project/cvg/blumh/replica/$scene/'")
seg4=$(sbatch $commonargs --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_internimage.py --replica --flip \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_internimage_flip /cluster/project/cvg/blumh/replica/$scene/'")

sam=$(sbatch $commonargs --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_sam.py --replica \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_sam /cluster/project/cvg/blumh/replica/$scene/'")

seg5=$(sbatch $commonargs -d afterany:$depth --wrap="bash -c './scripts/copy_replica.bash $scene && cp -r $scenefolder/hha \$TMPDIR/$scene/ && python scripts/segmentation_cmx.py --replica \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_cmx /cluster/project/cvg/blumh/replica/$scene/'")
seg6=$(sbatch $commonargs -d afterany:$depth --wrap="bash -c './scripts/copy_replica.bash $scene && python scripts/segmentation_cmx.py --replica --flip \$TMPDIR/$scene && cp -r \$TMPDIR/$scene/pred_cmx_flip /cluster/project/cvg/blumh/replica/$scene/'")

sbatch -d afterany:$seg1,afterany:$seg2,afterany:$seg3,afterany:$seg4,afterany:$seg5,afterany:$seg6 $commonargs --wrap="bash scripts/replicaconsensus.bash $scene"
