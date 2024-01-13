#!/usr/bin/bash
#SBATCH --job-name="labelmaker-check"
#SBATCH --output=%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G

module load gcc/11.4.0 python/3.11.6

# batch_idx='0060'
# batch_idx='0036'
batch_idx='0010'

root_dir=/cluster/project/cvg/labelmaker/ARKitScene_LabelMaker

input="/cluster/home/guanji/LabelMaker/pipeline/arkitscenes_info.csv"
while IFS= read -r line; do
  csv_line=($(echo $line | tr "," "\n"))
  video_id=${csv_line[0]}
  visit_id=${csv_line[1]}
  fold=${csv_line[2]}
  num_images=${csv_line[3]}
  group=${csv_line[4]}
  target_dir=$root_dir/$fold/$video_id
  if [[ $group == *"$batch_idx"* ]]; then
    echo $fold/$video_id
    python /cluster/home/guanji/LabelMaker/pipeline/check_progress.py --workspace $target_dir
  fi
done <"$input"
