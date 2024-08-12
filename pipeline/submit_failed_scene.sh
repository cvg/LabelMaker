#!/usr/bin/bash
#SBATCH --job-name="labelmaker-submit"
#SBATCH --output=submit_failed_batch_%j.out
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G

# exit when any command fails
set -e

start_group=62
end_group=101

failed_list="47895861 47431041 47429886 47430280 48458366 42899654 42445585 47331294 43649798 41069117 42445684 42445955 41254564 42444781 41254956 41159677 41126777 44358378 45260951 47669896 47334912 42897945 41126869 41254656 41159772 42898990 42447308 47331921 47669927 41254725 47430310 47332022 42444908 42445040 41125564 42899935 47334392 47333847 47333785 41159391 47333033 47429860 42445044 47895274 47331946 47895476 45663349 45261343 41159368 47115227 47333618 47895306 42898983 43828478 47333494 47333581 45663206 41048180 42899653 42899642 47333826 42897629"

echo $start_group $end_group

module load stack/2024-06  gcc/12.2.0 python/3.10.13

root_dir=/cluster/project/cvg/labelmaker/ARKitScene_LabelMaker

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
  if [[ 10#$group_int -ge 10#$start_group ]] && [[ 10#$group_int -le 10#$end_group ]]; then

    # if current video_id is in the failure list
    if [[ $failed_list =~ (^|[[:space:]])"$video_id"($|[[:space:]]) ]]; then
      echo video_id:$video_id fold:$fold group:$group
      python /cluster/home/guanji/LabelMaker/pipeline/submit.py \
        --root_dir $root_dir --fold $fold --video_id $video_id --num_images $num_images --sdfstudio_gpu_type 3090

      bash temp_slurm_submit.sh
    fi

  fi
done <"$input"

