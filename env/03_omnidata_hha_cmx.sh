echo ${ENV_FOLDER}

# Step 1: create folder and install omnidata # might be deprecated as weight will be stored at other path
mkdir -p ${ENV_FOLDER}/../3rdparty/omnidata/omnidata_tools/torch/pretrained_models/

# Step 2: install HHA
cd ${ENV_FOLDER}/../3rdparty/Depth2HHA-python
pip install .

# Step 3: install cmx
cd ${ENV_FOLDER}/../3rdparty/mmsegmentation
pip install -v -e .

# Step 4: create an empty txt for cmx eval configuration
cd ${ENV_FOLDER}/../3rdparty/RGBX_Semantic_Segmentation
touch empty.txt

# Step 5: replace collectioin.iterable into collection.abc.iterable
sed -i 's/collections.Iterable/collections.abc.Iterable/g' ${ENV_FOLDER}/../3rdparty/RGBX_Semantic_Segmentation/utils/transforms.py
