#!/bin/bash

# bash pipeline/pure_slurm_pipeline.sh 41254441 Validation
# bash pipeline/pure_slurm_pipeline.sh 41159856 Training

video_id=$1
fold=$2

root_dir=/cluster/project/cvg/labelmaker/ARKitScene_LabelMaker
target_dir=$root_dir/$fold/$video_id
mkdir -p $target_dir
mkdir -p $target_dir/intermediate
log_dir=$root_dir/slurm_log
mkdir -p $log_dir

LABELMAKER_REPO=/cluster/home/guanji/LabelMaker

commonargs=" --parsable -n 1"

download_preprocessing_args="--cpus-per-task=1 --mem-per-cpu=32G" # large scene need memory for tsdf
download_preprocessing_time="20:00"

video_render_args="--cpus-per-task=8 --mem-per-cpu=4G"
video_render_time="1:00:00"

gsam_args="--cpus-per-task=1 --mem-per-cpu=32G --gpus=rtx_3090:1"
gsam_time="4:00:00"
gsam_vote=1
gsam_label_space=wordnet

download_preprocessing_flag=$(sbatch $commonargs $download_preprocessing_args --time=$download_preprocessing_time --output=$log_dir/${video_id}_download_preprocess.out --wrap="module load gcc/11.4.0 cuda/12.1.1 eth_proxy && mkdir -p \$TMPDIR/source && mkdir -p \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/source:/source --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/arkitscene_download.py --video_id $video_id --download_dir /source && ls -lah /source/raw/$fold/$video_id && python ./scripts/arkitscenes2labelmaker.py --scan_dir /source/raw/$fold/$video_id --target_dir /target' && cp -r \$TMPDIR/target/*  $target_dir")

# two gsam run, one flip, one not
gsam_1_name='gsam'
gsam_1_output_folder="intermediate/wordnet_groundedsam_vote:${gsam_vote}_1"
gsam_1_output_folder_real=$gsam_1_output_folder
gsam_1_video_name=${gsam_1_output_folder_real}_viz.mp4
gsam_1_flag=$(sbatch $commonargs $gsam_args -d afterany:$download_preprocessing_flag --time=$gsam_time --output=$log_dir/${video_id}_${gsam_1_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/grounded_sam.py --workspace /target --output $gsam_1_output_folder' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")

# render video
sbatch $commonargs $video_render_args -d afterany:$gsam_1_flag --time=$video_render_time --output=$log_dir/${video_id}_${gsam_1_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/${gsam_1_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${gsam_label_space} --label_folder temp_folder --output_video_name ${gsam_1_video_name}' && cp -r \$TMPDIR/target/${gsam_1_video_name}  $target_dir/intermediate"

# flip
gsam_2_name='gsam_flip'
gsam_2_output_folder="intermediate/wordnet_groundedsam_vote:${gsam_vote}_1"
gsam_2_output_folder_real=${gsam_2_output_folder}_flip
gsam_2_video_name=${gsam_2_output_folder_real}_viz.mp4
gsam_2_flag=$(sbatch $commonargs $gsam_args -d afterany:$download_preprocessing_flag --time=$gsam_time --output=$log_dir/${video_id}_${gsam_2_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/grounded_sam.py --workspace /target --output $gsam_2_output_folder' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")

# render video
sbatch $commonargs $video_render_args -d afterany:$gsam_2_flag --time=$video_render_time --output=$log_dir/${video_id}_${gsam_2_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/${gsam_2_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${gsam_label_space} --label_folder temp_folder --output_video_name ${gsam_2_video_name}' && cp -r \$TMPDIR/target/${gsam_2_video_name}  $target_dir/intermediate"
