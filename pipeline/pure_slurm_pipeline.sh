#!/bin/bash

# bash pipeline/pure_slurm_pipeline.sh 41254441 Validation
# bash pipeline/pure_slurm_pipeline.sh 41159856 Training

video_id=$1
fold=$2

root_dir=/cluster/project/cvg/labelmaker/ARKitScene_LabelMaker
target_dir=$root_dir/$fold/$video_id
mkdir -p $target_dir
mkdir -p $target_dir/intermediate
mkdir -p $target_dir/video

log_dir=$root_dir/slurm_log
mkdir -p $log_dir

LABELMAKER_REPO=/cluster/home/guanji/LabelMaker

commonargs=" --parsable -n 1"

download_preprocessing_args="--cpus-per-task=1 --mem-per-cpu=32G" # large scene need memory for tsdf
download_preprocessing_time="40:00"

video_render_args="--cpus-per-task=8 --mem-per-cpu=4G"
video_render_time="1:00:00"

gsam_args="--cpus-per-task=1 --mem-per-cpu=32G --gpus=rtx_3090:1"
gsam_time="4:00:00"
gsam_vote=1
gsam_label_space=wordnet

ovseg_args="--cpus-per-task=1 --mem-per-cpu=32G --gpus=rtx_3090:1"
ovseg_time="6:00:00" # ovseg is the slowest
ovseg_vote=1
ovseg_label_space=wordnet

intern_args="--cpus-per-task=1 --mem-per-cpu=32G --gpus=rtx_3090:1"
intern_time="3:00:00"
intern_vote=1
intern_label_space=ade20k

mask3d_args="--cpus-per-task=8 --mem-per-cpu=4G --gpus=rtx_3090:1"
mask3d_time="1:30:00"
mask3d_vote=1
mask3d_label_space=scannet200

omnidata_args="--cpus-per-task=8 --mem-per-cpu=4G --gpus=rtx_3090:1"
omnidata_time="2:00:00"

hha_args="--cpus-per-task=18 --mem-per-cpu=3G"
hha_time="1:30:00"

cmx_args="--cpus-per-task=1 --mem-per-cpu=32G --gpus=rtx_3090:1"
cmx_time="1:00:00"
cmx_vote=1
cmx_label_space=nyu40

consensus_args="--cpus-per-task=16 --mem-per-cpu=4G"
consensus_time="2:00:00"
consensus_label_space=wordnet
consensus_dependency=""

point_lifting_args="--cpus-per-task=2 --mem-per-cpu=18G"
point_lifting_time="1:00:00"

download_preprocessing_flag=$(sbatch $commonargs $download_preprocessing_args --time=$download_preprocessing_time --output=$log_dir/${video_id}_download_preprocess.out --wrap="module load gcc/11.4.0 cuda/12.1.1 eth_proxy && mkdir -p \$TMPDIR/source && mkdir -p \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/source:/source --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/arkitscene_download.py --video_id $video_id --download_dir /source && ls -lah /source/raw/$fold/$video_id && python ./scripts/arkitscenes2labelmaker.py --scan_dir /source/raw/$fold/$video_id --target_dir /target' && cp -r \$TMPDIR/target/*  $target_dir")

# two gsam run, one flip, one not
gsam_1_name='gsam'
gsam_1_output_folder="${gsam_label_space}_groundedsam_${gsam_vote}_1"
gsam_1_output_folder_real=$gsam_1_output_folder
gsam_1_video_name=${gsam_1_output_folder_real}_viz.mp4
gsam_1_flag=$(sbatch $commonargs $gsam_args -d afterany:$download_preprocessing_flag --time=$gsam_time --output=$log_dir/${video_id}_${gsam_1_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/grounded_sam.py --workspace /target --output intermediate/$gsam_1_output_folder' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")
consensus_dependency=${consensus_dependency},afterany:$gsam_1_flag
# render video
sbatch $commonargs $video_render_args -d afterany:$gsam_1_flag --time=$video_render_time --output=$log_dir/${video_id}_${gsam_1_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/${gsam_1_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${gsam_label_space} --label_folder temp_folder --output_video_name ${gsam_1_video_name}' && cp -r \$TMPDIR/target/${gsam_1_video_name}  $target_dir/video"
# flip
gsam_2_name='gsam_flip'
gsam_2_output_folder="${gsam_label_space}_groundedsam_${gsam_vote}_1"
gsam_2_output_folder_real=${gsam_2_output_folder}_flip
gsam_2_video_name=${gsam_2_output_folder_real}_viz.mp4
gsam_2_flag=$(sbatch $commonargs $gsam_args -d afterany:$download_preprocessing_flag --time=$gsam_time --output=$log_dir/${video_id}_${gsam_2_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/grounded_sam.py --flip --workspace /target --output intermediate/$gsam_2_output_folder' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")
consensus_dependency=${consensus_dependency},afterany:$gsam_2_flag
# render video
sbatch $commonargs $video_render_args -d afterany:$gsam_2_flag --time=$video_render_time --output=$log_dir/${video_id}_${gsam_2_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/${gsam_2_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${gsam_label_space} --label_folder temp_folder --output_video_name ${gsam_2_video_name}' && cp -r \$TMPDIR/target/${gsam_2_video_name}  $target_dir/video"

# ovseg
ovseg_1_name='ovseg'
ovseg_1_output_folder="${ovseg_label_space}_ovseg_${ovseg_vote}_1"
ovseg_1_output_folder_real=$ovseg_1_output_folder
ovseg_1_video_name=${ovseg_1_output_folder_real}_viz.mp4
ovseg_1_flag=$(sbatch $commonargs $ovseg_args -d afterany:$download_preprocessing_flag --time=$ovseg_time --output=$log_dir/${video_id}_${ovseg_1_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/ovseg.py --workspace /target --output intermediate/$ovseg_1_output_folder' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")
consensus_dependency=${consensus_dependency},afterany:$ovseg_1_flag
# render video
sbatch $commonargs $video_render_args -d afterany:$ovseg_1_flag --time=$video_render_time --output=$log_dir/${video_id}_${ovseg_1_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/${ovseg_1_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${ovseg_label_space} --label_folder temp_folder --output_video_name ${ovseg_1_video_name}' && cp -r \$TMPDIR/target/${ovseg_1_video_name}  $target_dir/video"
# ovseg flip
ovseg_2_name='ovseg_flip'
ovseg_2_output_folder="${ovseg_label_space}_ovseg_${ovseg_vote}_1"
ovseg_2_output_folder_real=${ovseg_2_output_folder}_flip
ovseg_2_video_name=${ovseg_2_output_folder_real}_viz.mp4
ovseg_2_flag=$(sbatch $commonargs $ovseg_args -d afterany:$download_preprocessing_flag --time=$ovseg_time --output=$log_dir/${video_id}_${ovseg_2_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/ovseg.py --flip --workspace /target --output intermediate/$ovseg_2_output_folder' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")
consensus_dependency=${consensus_dependency},afterany:$ovseg_2_flag
# render video
sbatch $commonargs $video_render_args -d afterany:$ovseg_2_flag --time=$video_render_time --output=$log_dir/${video_id}_${ovseg_2_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/${ovseg_2_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${ovseg_label_space} --label_folder temp_folder --output_video_name ${ovseg_2_video_name}' && cp -r \$TMPDIR/target/${ovseg_2_video_name}  $target_dir/video"

# internimage
intern_1_name='intern'
intern_1_output_folder="${intern_label_space}_internimage_${intern_vote}_1"
intern_1_output_folder_real=$intern_1_output_folder
intern_1_video_name=${intern_1_output_folder_real}_viz.mp4
intern_1_flag=$(sbatch $commonargs $intern_args -d afterany:$download_preprocessing_flag --time=$intern_time --output=$log_dir/${video_id}_${intern_1_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/internimage.py --workspace /target --output intermediate/$intern_1_output_folder' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")
consensus_dependency=${consensus_dependency},afterany:$intern_1_flag
# render video
sbatch $commonargs $video_render_args -d afterany:$intern_1_flag --time=$video_render_time --output=$log_dir/${video_id}_${intern_1_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/${intern_1_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${intern_label_space} --label_folder temp_folder --output_video_name ${intern_1_video_name}' && cp -r \$TMPDIR/target/${intern_1_video_name}  $target_dir/video"
# internimage flip
intern_2_name='intern_flip'
intern_2_output_folder="${intern_label_space}_internimage_${intern_vote}_1"
intern_2_output_folder_real=${intern_2_output_folder}_flip
intern_2_video_name=${intern_2_output_folder_real}_viz.mp4
intern_2_flag=$(sbatch $commonargs $intern_args -d afterany:$download_preprocessing_flag --time=$intern_time --output=$log_dir/${video_id}_${intern_2_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/internimage.py --flip --workspace /target --output intermediate/$intern_2_output_folder' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")
consensus_dependency=${consensus_dependency},afterany:$intern_2_flag
# render video
sbatch $commonargs $video_render_args -d afterany:$intern_2_flag --time=$video_render_time --output=$log_dir/${video_id}_${intern_2_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/${intern_2_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${intern_label_space} --label_folder temp_folder --output_video_name ${intern_2_video_name}' && cp -r \$TMPDIR/target/${intern_2_video_name}  $target_dir/video"

# mask3d
mask3d_1_name='mask3d_1'
mask3d_1_output_folder="${mask3d_label_space}_mask3d_${mask3d_vote}_1"
mask3d_1_output_folder_real=$mask3d_1_output_folder
mask3d_1_video_name=${mask3d_1_output_folder_real}_viz.mp4
mask3d_1_flag=$(sbatch $commonargs $mask3d_args -d afterany:$download_preprocessing_flag --time=$mask3d_time --output=$log_dir/${video_id}_${mask3d_1_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/depth \$TMPDIR/target && cp -r ${target_dir}/intrinsic \$TMPDIR/target && cp -r ${target_dir}/pose \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/utils_3d.py --workspace /target --sdf_trunc 0.06 --voxel_length 0.02 && python ./models/mask3d_inst.py --seed 42 --workspace /target --output intermediate/$mask3d_1_output_folder' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")
consensus_dependency=${consensus_dependency},afterany:$mask3d_1_flag
# render
sbatch $commonargs $video_render_args -d afterany:$mask3d_1_flag --time=$video_render_time --output=$log_dir/${video_id}_${mask3d_1_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/${mask3d_1_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${mask3d_label_space} --label_folder temp_folder --output_video_name ${mask3d_1_video_name}' && cp -r \$TMPDIR/target/${mask3d_1_video_name}  $target_dir/video"
# mask3d second pass
mask3d_2_name='mask3d_2'
mask3d_2_output_folder="${mask3d_label_space}_mask3d_${mask3d_vote}_2"
mask3d_2_output_folder_real=$mask3d_2_output_folder
mask3d_2_video_name=${mask3d_2_output_folder_real}_viz.mp4
mask3d_2_flag=$(sbatch $commonargs $mask3d_args -d afterany:$download_preprocessing_flag --time=$mask3d_time --output=$log_dir/${video_id}_${mask3d_2_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/depth \$TMPDIR/target && cp -r ${target_dir}/intrinsic \$TMPDIR/target && cp -r ${target_dir}/pose \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/utils_3d.py --workspace /target --sdf_trunc 0.06 --voxel_length 0.02 && python ./models/mask3d_inst.py --seed 43 --workspace /target --output intermediate/$mask3d_2_output_folder' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")
consensus_dependency=${consensus_dependency},afterany:$mask3d_2_flag
# render
sbatch $commonargs $video_render_args -d afterany:$mask3d_2_flag --time=$video_render_time --output=$log_dir/${video_id}_${mask3d_2_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/${mask3d_2_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${mask3d_label_space} --label_folder temp_folder --output_video_name ${mask3d_2_video_name}' && cp -r \$TMPDIR/target/${mask3d_2_video_name}  $target_dir/video"

# omnidepth + omninormal
omnidata_flag=$(sbatch $commonargs $omnidata_args -d afterany:$download_preprocessing_flag --time=$omnidata_time --output=$log_dir/${video_id}_omnidata.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/depth \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/omnidata_depth.py --workspace /target && python ./models/omnidata_normal.py --workspace /target' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")

# hha
hha_flag=$(sbatch $commonargs $hha_args -d afterany:$omnidata_flag --time=$hha_time --output=$log_dir/${video_id}_hha.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/intermediate/depth_omnidata_1 \$TMPDIR/target/temp_omni_depth && cp -r ${target_dir}/depth \$TMPDIR/target && cp -r ${target_dir}/intrinsic \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/hha_depth.py --workspace /target --input temp_omni_depth --n_jobs 18' && cp -r \$TMPDIR/target/intermediate/*  $target_dir/intermediate")

# cmx
cmx_1_name='cmx'
cmx_1_output_folder="${cmx_label_space}_cmx_${cmx_vote}_1"
cmx_1_output_folder_real=$cmx_1_output_folder
cmx_1_video_name=${cmx_1_output_folder_real}_viz.mp4
cmx_1_flag=$(sbatch $commonargs $cmx_args -d afterany:$hha_flag --time=$cmx_time --output=$log_dir/${video_id}_${cmx_1_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/target/intermediate && cp -r ${target_dir}/intermediate/hha \$TMPDIR/target/intermediate && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/cmx.py --workspace /target --output intermediate/$cmx_1_output_folder' && cp -r \$TMPDIR/target/intermediate/$cmx_1_output_folder_real  $target_dir/intermediate")
consensus_dependency=${consensus_dependency},afterany:$cmx_1_flag
# render
sbatch $commonargs $video_render_args -d afterany:$cmx_1_flag --time=$video_render_time --output=$log_dir/${video_id}_${cmx_1_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/${cmx_1_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${cmx_label_space} --label_folder temp_folder --output_video_name ${cmx_1_video_name}' && cp -r \$TMPDIR/target/${cmx_1_video_name}  $target_dir/video"
# flip
cmx_2_name='cmx_flip'
cmx_2_output_folder="${cmx_label_space}_cmx_${cmx_vote}_1"
cmx_2_output_folder_real=${cmx_2_output_folder}_flip
cmx_2_video_name=${cmx_2_output_folder_real}_viz.mp4
cmx_2_flag=$(sbatch $commonargs $cmx_args -d afterany:$hha_flag --time=$cmx_time --output=$log_dir/${video_id}_${cmx_2_name}.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/correspondence.json \$TMPDIR/target && mkdir -p \$TMPDIR/target/intermediate && cp -r ${target_dir}/intermediate/hha \$TMPDIR/target/intermediate && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./models/cmx.py --flip --workspace /target --output intermediate/$cmx_2_output_folder' && cp -r \$TMPDIR/target/intermediate/$cmx_2_output_folder_real  $target_dir/intermediate")
consensus_dependency=${consensus_dependency},afterany:$cmx_2_flag
# render
sbatch $commonargs $video_render_args -d afterany:$cmx_2_flag --time=$video_render_time --output=$log_dir/${video_id}_${cmx_2_name}_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/${cmx_2_output_folder_real} \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${cmx_label_space} --label_folder temp_folder --output_video_name ${cmx_2_video_name}' && cp -r \$TMPDIR/target/${cmx_2_video_name}  $target_dir/video"

# consensus
consensus_flag=$(sbatch $commonargs $consensus_args -d $consensus_dependency --time=$consensus_time --output=$log_dir/${video_id}_consensus.out --wrap="mkdir -p \$TMPDIR/target && mkdir -p \$TMPDIR/target/intermediate && cp -r ${target_dir}/intermediate/* \$TMPDIR/target/intermediate && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./labelmaker/consensus.py --n_jobs 16 --custom_vote_weight --workspace /target ' && cp -r \$TMPDIR/target/intermediate/consensus  $target_dir/intermediate")
# render
sbatch $commonargs $video_render_args -d afterany:$consensus_flag --time=$video_render_time --output=$log_dir/${video_id}_consensus_viz.out --wrap="mkdir -p \$TMPDIR/target && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/intermediate/consensus \$TMPDIR/target/temp_folder && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./scripts/video_visualization.py --workspace /target --label_space ${consensus_label_space} --label_folder temp_folder --output_video_name consensus_viz.mp4' && cp -r \$TMPDIR/target/consensus_viz.mp4  $target_dir/video"

# point lifting
sbatch $commonargs $point_lifting_args -d $consensus_flag --time=$point_lifting_time --output=$log_dir/${video_id}_point_lifting.out --wrap="mkdir -p \$TMPDIR/target && mkdir -p \$TMPDIR/target/intermediate && cp -r ${target_dir}/intermediate/consensus \$TMPDIR/target/intermediate && cp -r ${target_dir}/color \$TMPDIR/target && cp -r ${target_dir}/depth \$TMPDIR/target && cp -r ${target_dir}/intrinsic \$TMPDIR/target && cp -r ${target_dir}/pose \$TMPDIR/target && cp -r ${target_dir}/mesh.ply \$TMPDIR/target && mkdir -p \$TMPDIR/.cache && module load gcc/11.4.0 cuda/12.1.1 eth_proxy && singularity exec --nv --bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints --bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker --bind $LABELMAKER_REPO/models:/LabelMaker/models --bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts --bind $LABELMAKER_REPO/pipeline:/LabelMaker/pipeline --bind \$TMPDIR/.cache:\$HOME/.cache --bind \$TMPDIR/target:/target /cluster/project/cvg/labelmaker/labelmaker_20231227.simg bash -c 'cd /LabelMaker && source ./pipeline/activate_labelmaker.sh && python ./labelmaker/lifting_3d/lifting_points.py --workspace /target ' && cp -r \$TMPDIR/target/labels.txt $target_dir && cp -r \$TMPDIR/target/point_lifted_mesh.ply $target_dir"
