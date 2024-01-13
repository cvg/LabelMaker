cd /LabelMaker
source ./pipeline/activate_labelmaker.sh
python ./models/hha_depth.py --workspace /target --input temp_omni_depth --n_jobs 18
