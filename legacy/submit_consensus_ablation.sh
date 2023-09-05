#!/bin/bash

# no_cmx
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_cmx.bash scene0000_00"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_cmx.bash scene0164_02"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_cmx.bash scene0458_00"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_cmx.bash scene0474_01"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_cmx.bash scene0518_00"

# no_intern
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_intern.bash scene0000_00"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_intern.bash scene0164_02"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_intern.bash scene0458_00"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_intern.bash scene0474_01"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_intern.bash scene0518_00"

# no_mask3d
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_mask3d.bash scene0000_00"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_mask3d.bash scene0164_02"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_mask3d.bash scene0458_00"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_mask3d.bash scene0474_01"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_mask3d.bash scene0518_00"

# no_ovseg
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_ovseg.bash scene0000_00"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_ovseg.bash scene0164_02"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_ovseg.bash scene0458_00"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_ovseg.bash scene0474_01"
sbatch --parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --time=04:00:00 --wrap="bash scripts/consensus_ablation/consensus_no_ovseg.bash scene0518_00"
