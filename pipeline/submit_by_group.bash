batch_idx=$1

input="./pipeline/arkitscenes_info.csv"
while IFS= read -r line; do
  csv_line=($(echo $line | tr "," "\n"))
  video_id=${csv_line[0]}
  visit_id=${csv_line[1]}
  fold=${csv_line[2]}
  num_images=${csv_line[3]}
  group=${csv_line[4]}
  if [[ $group == *"$batch_idx"* ]]; then
    bash ./pipeline/pure_slurm_pipeline.sh $video_id
  fi
done <"$input"
