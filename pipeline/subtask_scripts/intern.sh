echo $INTERN_OUTPUT_FOLDER
echo $FLIP

cd /LabelMaker
source ./pipeline/activate_labelmaker.sh
python ./models/internimage.py --workspace /target $FLIP --output intermediate/$INTERN_OUTPUT_FOLDER
