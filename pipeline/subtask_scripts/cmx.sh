echo $CMX_OUTPUT_FOLDER
echo $FLIP

cd /LabelMaker
source ./pipeline/activate_labelmaker.sh
python ./models/cmx.py $FLIP --workspace /target --output intermediate/$CMX_OUTPUT_FOLDER
