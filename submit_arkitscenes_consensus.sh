#!/bin/bash

scene=$1
scenefolder=/cluster/project/cvg/blumh/arkitscenes/$scene
echo $scene
commonargsnogpu="--parsable -n 1 --cpus-per-task=8 --mem-per-cpu=4000 --tmp=16000 --mail-type=END,FAIL --mail-user=weders@ethz.ch"
commonargs="$commonargsnogpu --gpus=rtx_3090:1"

sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/arkitscenesconsensus.bash 42445991"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/arkitscenesconsensus.bash 42446517"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/arkitscenesconsensus.bash 42446527"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/arkitscenesconsensus.bash 42897521"
sbatch $commonargsnogpu --time=04:00:00 --wrap="bash scripts/arkitscenesconsensus.bash 42897688"
