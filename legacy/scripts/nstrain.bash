ns-train nerfacto \
    --trainer.steps-per-eval-image 5000 \
    --trainer.max-num-iterations 60000 \
    --pipeline.datamanager.train-num-rays-per-batch 2048 \
    --pipeline.datamanager.camera-optimizer.mode off \
    --vis wandb \
    sdfstudio-data \
    --data /media/blumh/data/replica/office_0/Sequence_1/sdfstudio \
