#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:11264m
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=16000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weders@ethz.ch


source venv3090/bin/activate

set -e

scene=$1
experiment_name=${scene}_scannet_weight_5_class_weights_no_cmx
echo $scene
echo $scene

mkdir $TMPDIR/$scene
for SUBDIR in pred_consensus_no_cmx color depth label-filt intrinsic omnidata_depth omnidata_normal pose pred_sam pred_consensus pred_consensus_5_scannet
do
    echo Copying $SUBDIR ...
	cp -r /cluster/project/cvg/blumh/scannet/$scene/$SUBDIR $TMPDIR/$scene/
done

python scripts/sdfstudio_scannet_preprocessing.py --label_template pred_scannet_no_cmx --sampling 2 --size 416 --scannetpose \
    $TMPDIR/$scene

ns-train neus-facto \
    --experiment-name $experiment_name \
    --pipeline.model.sdf-field.use-grid-feature True \
    --pipeline.model.sdf-field.hidden-dim 256 \
    --pipeline.model.sdf-field.num-layers 2 \
    --pipeline.model.sdf-field.num-layers-color 2 \
    --pipeline.model.sdf-field.semantic-num-layers 4 \
    --pipeline.model.sdf-field.use-appearance-embedding False \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.inside-outside True  \
    --pipeline.model.sdf-field.bias 0.8 \
    --pipeline.model.sdf-field.beta-init 0.3 \
    --pipeline.model.sensor-depth-l1-loss-mult 0.3 \
    --pipeline.model.sensor-depth-sdf-loss-mult 0.3 \
    --pipeline.model.sensor-depth-freespace-loss-mult 0.3 \
    --pipeline.model.mono-normal-loss-mult 0.02 \
    --pipeline.model.mono-depth-loss-mult 0.000 \
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
    --output-dir $TMPDIR/output \
    --vis wandb \
    sdfstudio-data \
    --data $TMPDIR/$scene/sdfstudio \
    --include-sensor-depth True \
    --include-semantics True \
    --include-mono-prior True

outfolder=$TMPDIR/output/$(ls $TMPDIR/output)/neus-facto
train_id=$(ls $outfolder)
cp -r $outfolder/$train_id ./outputs/
echo "$outfolder/$train_id"

config=$outfolder/$train_id/config.yml
# the job below may OOM sometimes, so we wait such that all GPU memory is free
sleep 60

ns-extract-mesh --load-config $config --create-visibility-mask True --output-path $outfolder/$train_id/mesh_visible.ply --resolution 2048
cp -r $outfolder/$train_id ./outputs/
sleep 60

ns-render --camera-path-filename $TMPDIR/$scene/sdfstudio/camera_path.json \
    --traj filename \
    --output-format images \
    --rendered-output-names semantics \
    --output-path $TMPDIR/$scene/pred_sdfstudio_$train_id.png \
    --load-config $config
cp -r $TMPDIR/$scene/pred_sdfstudio_$train_id /cluster/project/cvg/blumh/scannet/$scene/

