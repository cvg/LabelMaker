cd /LabelMaker
source ./pipeline/activate_labelmaker.sh

# omnidata normal for sdfstido training
python ./models/omnidata_normal.py --workspace /target

source ./pipeline/activate_sdfstudio.sh

# preprocessing
python ./labelmaker/lifting_3d/preprocessing.py \
  --sampling 1 \
  --train_width 320 \
  --train_height 240 \
  --workspace /target

# training
method=neus-facto
experiment_name=sdfstudio_train
output_dir=/target/intermediate/${experiment_name}
preprocess_data_dir=/target/intermediate/sdfstudio_preprocessing

export WANDB_MODE=online
wandb online

ns-train ${method} \
  --experiment-name $experiment_name \
  --pipeline.model.sdf-field.use-grid-feature True \
  --pipeline.model.sdf-field.hidden-dim 256 \
  --pipeline.model.sdf-field.num-layers 2 \
  --pipeline.model.sdf-field.num-layers-color 2 \
  --pipeline.model.sdf-field.semantic-num-layers 4 \
  --pipeline.model.sdf-field.semantic_layer_width 512 \
  --pipeline.model.sdf-field.use-appearance-embedding False \
  --pipeline.model.sdf-field.geometric-init True \
  --pipeline.model.sdf-field.inside-outside True \
  --pipeline.model.sdf-field.bias 0.8 \
  --pipeline.model.sdf-field.beta-init 0.3 \
  --pipeline.model.sensor-depth-l1-loss-mult 10.0 \
  --pipeline.model.sensor-depth-sdf-loss-mult 6000.0 \
  --pipeline.model.sensor-depth-freespace-loss-mult 10.0 \
  --pipeline.model.sensor-depth-truncation 0.015 \
  --pipeline.model.mono-normal-loss-mult 0.02 \
  --pipeline.model.mono-depth-loss-mult 0.00 \
  --pipeline.model.semantic-loss-mult 0.1 \
  --pipeline.model.semantic-patch-loss-mult 0.00 \
  --pipeline.model.semantic-patch-loss-min-step 1000 \
  --pipeline.model.semantic-ignore-label 0 \
  --trainer.steps-per-eval-image 1000 \
  --trainer.steps-per-eval-all-images 100000 \
  --trainer.steps-per-save 10000 \
  --trainer.max-num-iterations 20001 \
  --pipeline.datamanager.train-num-rays-per-batch 2048 \
  --pipeline.model.eikonal-loss-mult 0.1 \
  --pipeline.model.background-model none \
  --output-dir /target/intermediate \
  --vis wandb \
  sdfstudio-data \
  --data ${preprocess_data_dir} \
  --include-sensor-depth True \
  --include-semantics True \
  --include-mono-prior True
