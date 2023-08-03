#!/bin/bash

commonargsnogpu="--parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --mail-type=END,FAIL --mail-user=weders@ethz.ch"
commonargs="$commonargsnogpu --gpus=rtx_3090:1"

sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash office_0_1"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash office_0_2"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash office_1_1"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash office_1_2"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash office_2_1"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash office_2_2"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash office_3_1"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash office_3_2"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash office_4_1"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash office_4_2"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash room_0_1"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash room_0_2"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash room_1_1"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash room_1_2"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash room_2_1"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/replicaconsensus.bash room_2_2"


