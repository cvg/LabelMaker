echo ${ENV_FOLDER}

# # avoid an error when no cuda runtime available
sed -i 's/torch.cuda.is_available()/True/g' ${ENV_FOLDER}/../3rdparty/InternImage/segmentation/ops_dcnv3/setup.py
cd ${ENV_FOLDER}/../3rdparty/InternImage/segmentation/ops_dcnv3
sh ./make.sh
