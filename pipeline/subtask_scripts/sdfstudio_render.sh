cd /LabelMaker
source ./pipeline/activate_sdfstudio.sh

output_dir=/target/intermediate/sdfstudio_train
results_dir=${output_dir}/$(ls $output_dir)
train_id=$(ls $results_dir)
config=$results_dir/$train_id/config.yml
render_dir=/target/neus_lifted
mkdir -p $render_dir

ns-render \
  --camera-path-filename /target/intermediate/sdfstudio_preprocessing/camera_path.json \
  --traj filename \
  --output-format images \
  --rendered-output-names semantics \
  --output-path $render_dir \
  --load-config $config
