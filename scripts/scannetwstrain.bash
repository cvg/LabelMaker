export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64'
set -e

scene=scene0575_00

python scripts/sdfstudio_scannet_preprocessing.py --sampling 20 \
    /media/blumh/data/scannet/$scene

# python scripts/rgbd_reconstruction.py --sdfstudio True /media/blumh/data/replica/office_0/Sequence_1/

ns-train neus-acc \
    --pipeline.model.sdf-field.use-grid-feature True \
    --pipeline.model.sdf-field.hidden-dim 256 \
    --pipeline.model.sdf-field.num-layers 2 \
    --pipeline.model.sdf-field.num-layers-color 2 \
    --pipeline.model.sdf-field.semantic-num-layers 2 \
    --pipeline.model.sdf-field.use-appearance-embedding False \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.inside-outside True  \
    --pipeline.model.sdf-field.bias 0.8 \
    --pipeline.model.sdf-field.beta-init 0.3 \
    --pipeline.model.sensor-depth-l1-loss-mult 0.1 \
    --pipeline.model.semantic-loss-mult 0.02 \
    --pipeline.model.semantic-patch-loss-mult 0.0 \
    --pipeline.model.semantic-patch-loss-min-step 1000 \
    --pipeline.model.semantic-ignore-label 0 \
    --trainer.steps-per-eval-image 5000 \
    --trainer.max-num-iterations 20001 \
    --pipeline.datamanager.train-num-rays-per-batch 2048 \
    --pipeline.model.eikonal-loss-mult 0.1 \
    --pipeline.model.background-model none \
    --vis wandb \
    sdfstudio-data \
    --data /media/blumh/data/scannet/$scene/sdfstudio \
    --include-sensor-depth True \
    --include-semantics True

train_id=$(ls -tl /home/blumh/CVG/scan_netter/ScanNetter/outputs/-media-blumh-data-scannet-$scene-sdfstudio/neus-acc/ | head -n 2 | tail -n 1 | awk '{print $9}')
echo $train_id

config=/home/blumh/CVG/scan_netter/ScanNetter/outputs/-media-blumh-data-scannet-$scene-sdfstudio/neus-acc/$train_id/config.yml
# the job below may OOM sometimes, so we wait such that all GPU memory is free
sleep 60

ns-render --camera-path-filename /media/blumh/data/scannet/$scene/sdfstudio/camera_path.json \
    --traj filename \
    --output-format images \
    --rendered-output-names semantics \
    --output-path /media/blumh/data/scannet/$scene/pred_sdfstudio_$train_id.png \
    --load-config $config

# ./eval_everything.bash
