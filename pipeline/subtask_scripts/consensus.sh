cd /LabelMaker
source ./pipeline/activate_labelmaker.sh
python ./labelmaker/consensus.py --n_jobs 16 --custom_vote_weight --workspace /target
