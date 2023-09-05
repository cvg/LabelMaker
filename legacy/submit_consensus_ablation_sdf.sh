#!/bin/bash

source euler_env
sleep 2

commonargsnogpu="--parsable -n 1 --cpus-per-task=16 --time=36:00:00 --mem-per-cpu=12000 --tmp=16000 --mail-type=END,FAIL --mail-user=weders@ethz.ch --gres=gpumem:11264m"

# # no_cmx
# source venv/bin/activate
# sleep 2

# sbatch $commonargsnogpu --job-name=no_cmx_0000 --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0000_00"

# source venv3090/bin/activate
# sleep 2

# sbatch $commonargsnogpu --job-name=no_cmx_0164 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0164_02"
# sbatch $commonargsnogpu --job-name=no_cmx_0458 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0458_00"
# sbatch $commonargsnogpu --job-name=no_cmx_0474 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0474_01"
# sbatch $commonargsnogpu --job-name=no_cmx_0518 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0518_00"

# # no_intern
# source venv/bin/activate
# sleep 2

# sbatch $commonargsnogpu --job-name=no_intern_0000 --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0000_00"


# source venv3090/bin/activate
# sleep 2

# sbatch $commonargsnogpu --job-name=no_intern_0000 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0164_02"
# sbatch $commonargsnogpu --job-name=no_intern_0458 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0458_00"
# sbatch $commonargsnogpu --job-name=no_intern_0474 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0474_01"
# sbatch $commonargsnogpu --job-name=no_intern_0518 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0518_00"

# # no_mask3d
# source venv/bin/activate
# sleep 2

# sbatch $commonargsnogpu --job-name=no_mask3d_0000 --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0000_00"

# source venv3090/bin/activate
# sleep 2

# sbatch $commonargsnogpu --job-name=no_mask3d_0164 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0164_02"
# sbatch $commonargsnogpu --job-name=no_mask3d_0458 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0458_00"
# sbatch $commonargsnogpu --job-name=no_mask3d_0474 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0474_01"
# sbatch $commonargsnogpu --job-name=no_mask3d_0518 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0518_00"

# no_ovseg
source activate venv/bin/activate
sleep 2

sbatch $commonargsnogpu --job-name=no_ovseg_0000 --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0000_00"

source venv3090/bin/activate
sleep 2

sbatch $commonargsnogpu --job-name=no_ovseg_0458 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0458_00"
sbatch $commonargsnogpu --job-name=no_ovseg_0474 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0474_01"
sbatch $commonargsnogpu --job-name=no_ovseg_0164  --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0164_02"
sbatch $commonargsnogpu --job-name=no_ovseg_0518 --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0518_00"
