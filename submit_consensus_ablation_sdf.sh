#!/bin/bash

commonargsnogpu="--parsable -n 1 --time=48:00:00 --mem-per-cpu=8000 --tmp=16000 --mail-type=END,FAIL --mail-user=weders@ethz.ch"

# no_cmx
source venv/bin/activate
sbatch $commonargsnogpu --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0000_00"
source venv3090/bin/activate
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0164_02"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0458_00"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0474_01"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0518_00"

# no_intern
source venv/bin/activate
sbatch $commonargsnogpu --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0000_00"
source venv3090/bin/activate
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0164_02"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0458_00"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0474_01"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0518_00"

# no_mask3d
source venv/bin/activate
sbatch $commonargsnogpu --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0000_00"
source venv3090/bin/activate
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0164_02"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0458_00"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0474_01"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0518_00"

# no_ovseg
source activate venv/bin/activate
sbatch $commonargsnogpu --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0000_00"
source venv3090/bin/activate
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0164_02"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0458_00"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0474_01"
sbatch $commonargsnogpu --gpus=rtx_3090:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0518_00"
