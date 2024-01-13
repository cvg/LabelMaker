echo $GSAM_OUTPUT_FOLDER
echo $FLIP

cd /LabelMaker
source ./pipeline/activate_labelmaker.sh
python ./models/grounded_sam.py --workspace /target $FLIP --output intermediate/$GSAM_OUTPUT_FOLDER
