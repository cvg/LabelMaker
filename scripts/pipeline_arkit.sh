set -e

env_name=labelmaker
eval "$(conda shell.bash hook)"
conda activate $env_name

if [ -z "$1" ]; then
  echo "No ARKitScene directory specified!"
  exit 1
else
  original_dir=$1
fi

if [ -z "$2" ]; then
  echo "No target directory specified!"
  exit 1
else
  target_dir=$2
fi

# preprocessing
python scripts/arkitscenes2labelmaker.py \
  --scan_dir ${original_dir} \
  --target_dir ${target_dir}

# now run pipeline.sh
scripts/pipeline.sh ${target_dir}

