cd /LabelMaker
source ./pipeline/activate_sdfstudio.sh

method=neus-facto
experiment_name=sdfstudio_train
output_dir=/target/intermediate/${experiment_name}
preprocess_data_dir=/target/intermediate/sdfstudio_preprocessing
results_dir=${output_dir}/$(ls $output_dir)
train_id=$(ls $results_dir)
config=$results_dir/$train_id/config.yml
ns-extract-mesh \
  --load-config $config \
  --create-visibility-mask True \
  --output-path $results_dir/$train_id/mesh_visible.ply \
  --resolution 512
