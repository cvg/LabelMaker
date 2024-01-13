echo $MASK3D_OUTPUT_FOLDER
echo $FLIP
echo $SEED

cd /LabelMaker
source ./pipeline/activate_labelmaker.sh
python ./scripts/utils_3d.py --workspace /target --sdf_trunc 0.06 --voxel_length 0.02
python ./models/mask3d_inst.py --seed $SEED $FLIP --workspace /target --output intermediate/$MASK3D_OUTPUT_FOLDER
