#!/usr/bin/bash
#SBATCH --job-name="labelmaker-delete"
#SBATCH --output=delete_log_%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G

# exit when any command fails
set -e

# decide which version of python cuda pytorch torchvision to use
# if [ -z "$1" ]; then
#   exit 1
# else
#   start_group=$1
# fi

# if [ -z "$2" ]; then
#   end_group=$start_group
# else
#   end_group=$2
# fi

start_group=78
end_group=101

module load stack/2024-06  gcc/12.2.0 python/3.10.13

root_dir=/cluster/project/cvg/labelmaker/ARKitScene_LabelMaker
# root_dir=/cluster/scratch/guanji/ARKitScene_LabelMaker

input="/cluster/home/guanji/LabelMaker/pipeline/arkitscenes_info.csv"
while IFS= read -r line; do
  csv_line=($(echo $line | tr "," "\n"))
  video_id=${csv_line[0]}
  visit_id=${csv_line[1]}
  fold=${csv_line[2]}
  num_images=${csv_line[3]}
  group=${csv_line[4]}
  group_int=${group:0:4}
  target_dir=$root_dir/$fold/$video_id
  log_dir=$root_dir/slurm_log/$video_id
  if [[ 10#$group_int -ge 10#$start_group ]] && [[ 10#$group_int -le 10#$end_group ]]; then
    if [ "$(python /cluster/home/guanji/LabelMaker/pipeline/check.py --root_dir $root_dir --fold $fold --video_id $video_id --no-verbose)" == "True" ]; then
      rm -rf $log_dir
    fi

  fi

  # fi
done <"$input"
