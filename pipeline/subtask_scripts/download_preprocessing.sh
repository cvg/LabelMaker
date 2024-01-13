# check env variable
echo $VIDEO_ID
echo $FOLD


cd /LabelMaker
source ./pipeline/activate_labelmaker.sh

# download
python ./scripts/arkitscene_download.py --video_id $VIDEO_ID --download_dir /source
ls -lah /source/raw/$FOLD/$VIDEO_ID

# preprocessing
python ./scripts/arkitscenes2labelmaker.py --scan_dir /source/raw/$FOLD/$VIDEO_ID --target_dir /target
