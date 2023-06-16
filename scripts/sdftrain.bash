
python scripts/sdfstudio_replica_preprocessing.py --replica True --sampling 2 /media/blumh/data/replica/office_0/Sequence_1/

# python scripts/rgbd_reconstruction.py --sdfstudio True /media/blumh/data/replica/office_0/Sequence_1/

ns-train neus-acc \
    --pipeline.model.sdf-field.use-grid-feature True \
    --pipeline.model.sdf-field.hidden-dim 256 \
    --pipeline.model.sdf-field.num-layers 2 \
    --pipeline.model.sdf-field.num-layers-color 2 \
    --pipeline.model.sdf-field.use-appearance-embedding False \
    --pipeline.model.sdf-field.geometric-init True \
    --pipeline.model.sdf-field.inside-outside True  \
    --pipeline.model.sdf-field.bias 0.8 \
    --pipeline.model.sdf-field.beta-init 0.3 \
    --pipeline.model.sensor_depth_l1_loss_mult 0.1 \
    --trainer.steps-per-eval-image 5000 \
    --pipeline.datamanager.train-num-rays-per-batch 2048 \
    --pipeline.model.eikonal-loss-mult 0.1 \
    --pipeline.model.background-model none \
    --vis wandb \
    sdfstudio-data \
    --data /media/blumh/data/replica/office_0/Sequence_1/sdfstudio \
    --include-sensor-depth True

    # --pipeline.datamanager.train-num-times-to-repeat-images 0 \
    # --pipeline.datamanager.train-num-images-to-sample-from 1 \
