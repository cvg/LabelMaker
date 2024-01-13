echo $OVSEG_OUTPUT_FOLDER
echo $FLIP

cd /LabelMaker
source ./pipeline/activate_labelmaker.sh
python ./models/ovseg.py --workspace /target $FLIP --output intermediate/$OVSEG_OUTPUT_FOLDER
