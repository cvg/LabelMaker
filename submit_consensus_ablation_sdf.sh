#!/bin/bash

# no_cmx
source venv/bin/activate
sbatch --parsable --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0000_00"
source venv3090/bin/activate
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0164_02"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0458_00"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0474_01"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_cmx.bash scene0518_00"

# no_intern
source venv/bin/activate
sbatch --parsable --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0000_00"
source venv3090/bin/activate
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0164_02"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0458_00"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0474_01"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_intern.bash scene0518_00"

# no_mask3d
source venv/bin/activate
sbatch --parsable --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0000_00"
source venv3090/bin/activate
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0164_02"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0458_00"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0474_01"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_mask3d.bash scene0518_00"

# no_ovseg
source activate venv/bin/activate
sbatch --parsable --gpus=v100:1 --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0000_00"
source venv3090/bin/activate
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0164_02"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0458_00"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0474_01"
sbatch --parsable --wrap="bash scripts/consensus_ablation/sdf_class_weights_no_ovseg.bash scene0518_00"
