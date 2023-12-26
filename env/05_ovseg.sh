echo ${ENV_FOLDER}

# install ovseg, ovseg customize clip, so reinstall from this after grounded sam
cd ${ENV_FOLDER}/../3rdparty/ov-seg/third_party/CLIP
python -m pip install -Ue .
python -m nltk.downloader -d ${NLTK_DATA} wordnet
