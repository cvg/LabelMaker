echo $TARGET_RENDER_LABEL_SPACE
echo $TARGET_RENDER_VIDEO_NAME

cd /LabelMaker
source ./pipeline/activate_labelmaker.sh
python ./scripts/video_visualization.py \
  --workspace /target \
  --label_space ${TARGET_RENDER_LABEL_SPACE} \
  --label_folder temp_folder \
  --output_video_name ${TARGET_RENDER_VIDEO_NAME}
