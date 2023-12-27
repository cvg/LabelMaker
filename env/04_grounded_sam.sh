set -e
echo ${ENV_FOLDER}

pip install --no-use-pep517 ${ENV_FOLDER}/../3rdparty/recognize-anything/
pip install ${ENV_FOLDER}/../3rdparty/Grounded-Segment-Anything/segment_anything
pip install ${ENV_FOLDER}/../3rdparty/Grounded-Segment-Anything/GroundingDINO
