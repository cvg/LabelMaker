
# python scripts/sdfstudio_replica_preprocessing.py --replica True --sampling 2 /media/blumh/data/replica/office_0/Sequence_1/
train_id=2023-06-13_193149
config=/home/blumh/CVG/scan_netter/ScanNetter/outputs/-media-blumh-data-replica-room_0-Sequence_1-sdfstudio/neus-acc/$train_id/config.yml

ns-render --camera-path-filename /media/blumh/data/replica/room_0/Sequence_1/sdfstudio/camera_path.json \
    --traj filename \
    --output-format images \
    --rendered-output-names semantics \
    --output-path /media/blumh/data/replica/room_0/Sequence_1/pred_sdfstudio_$train_id.png \
    --load-config $config

./eval_everything.bash
