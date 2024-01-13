import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline.pipeline_utils import check_progress_given_tasks, config_to_tasks, pipeline_config


def submit(
    root_dir: str,
    video_id: str,
    fold: str,
    num_images: int,
    sdfstudio_gpu_type: str = "3090",  # 3090, v100, a100_40, a100_80
):
  bash_script = f"""#!/bin/bash
export VIDEO_ID={video_id}
export FOLD={fold}
export NUM_IMAGES={num_images}
export ROOT_DIR={root_dir}""" + """

export TARGET_DIR=$ROOT_DIR/$FOLD/$VIDEO_ID
mkdir -p $TARGET_DIR
mkdir -p $TARGET_DIR/intermediate
mkdir -p $TARGET_DIR/video

export LOG_DIR=$ROOT_DIR/slurm_log
mkdir -p $LOG_DIR
mkdir -p $LOG_DIR/$VIDEO_ID

export LABELMAKER_REPO=/cluster/home/guanji/LabelMaker

commonargs=" -A ls_polle --parsable -n 1"

download_preprocessing_args="--cpus-per-task=2 --mem-per-cpu=12G" # large scene need memory for tsdf
download_preprocessing_time="4:00:00"

video_render_args="--cpus-per-task=8 --mem-per-cpu=4G"
video_render_time="30:00"

gsam_args="--cpus-per-task=2 --mem-per-cpu=6G --gpus=rtx_3090:1"
gsam_time="6:00:00"


ovseg_args="--cpus-per-task=2 --mem-per-cpu=4G --gpus=rtx_3090:1"
ovseg_time="8:00:00" # ovseg is the slowest

intern_args="--cpus-per-task=2 --mem-per-cpu=5G --gpus=rtx_3090:1"
intern_time="8:00:00"

mask3d_args="--cpus-per-task=8 --mem-per-cpu=2G --gpus=rtx_3090:1"
mask3d_time="1:30:00"

omnidata_args="--cpus-per-task=8 --mem-per-cpu=1G --gpus=rtx_3090:1"
omnidata_time="2:00:00"

hha_args="--cpus-per-task=18 --mem-per-cpu=512M"
hha_time="2:00:00"

cmx_args="--cpus-per-task=2 --mem-per-cpu=4G --gpus=rtx_3090:1"
cmx_time="3:00:00"

consensus_args="--cpus-per-task=16 --mem-per-cpu=1G"
consensus_time="2:00:00"

point_lifting_args="--cpus-per-task=2 --mem-per-cpu=18G"
point_lifting_time="4:00:00"

export WANDB_API_KEY="6b447b1218e7f042525c176c16b0cd32d3e58956"
export WANDB_ENTITY="labelmaker-sdfstudio"

post_processing_args="--cpus-per-task=1 --mem-per-cpu=4G"
post_processing_time="30:00"

sdfstudio_render_args="--cpus-per-task=16 --mem-per-cpu=$((281 + ${NUM_IMAGES} / 2))M --gpus=rtx_3090:1"
sdfstudio_render_time="$((25 + ${NUM_IMAGES} / 5))"

sdfstudio_train_time="6:00:00"

sdfstudio_extract_time="4:00:00"

"""

  # sdfstudio render args depend on the scale of the scene
  if sdfstudio_gpu_type == "3090":
    bash_script += """sdfstudio_train_args="--cpus-per-task=16 --mem-per-cpu=$((414 + ${NUM_IMAGES}))M --gpus=rtx_3090:1"

sdfstudio_extract_args="--cpus-per-task=16 --mem-per-cpu=$((414 + ${NUM_IMAGES}))M --gpus=rtx_3090:1"
"""
  elif sdfstudio_gpu_type == "v100":
    bash_script += """sdfstudio_train_args="--cpus-per-task=16 --mem-per-cpu=$((414 + ${NUM_IMAGES}))M --gpus=1 --gres=gpumem:25g"

sdfstudio_extract_args="--cpus-per-task=16 --mem-per-cpu=$((414 + ${NUM_IMAGES}))M --gpus=1 --gres=gpumem:25g"
"""
  elif sdfstudio_gpu_type == "a100_40g":
    bash_script += """sdfstudio_train_args="--cpus-per-task=16 --mem-per-cpu=$((414 + ${NUM_IMAGES}))M --gpus=1 --gres=gpumem:33g"

sdfstudio_extract_args="--cpus-per-task=16 --mem-per-cpu=$((414 + ${NUM_IMAGES}))M --gpus=1 --gres=gpumem:33g"
"""
  elif sdfstudio_gpu_type == "a100_80g":
    bash_script += """sdfstudio_train_args="--cpus-per-task=16 --mem-per-cpu=$((414 + ${NUM_IMAGES}))M --gpus=1 --gres=gpumem:41g"

sdfstudio_extract_args="--cpus-per-task=16 --mem-per-cpu=$((414 + ${NUM_IMAGES}))M --gpus=1 --gres=gpumem:41g"
"""

  bash_script += """
echo $sdfstudio_train_args $sdfstudio_extract_args $sdfstudio_render_args $sdfstudio_render_time
"""

  workspace = Path(root_dir) / fold / video_id

  tasks = check_progress_given_tasks(
      workspace=workspace,
      tasks=config_to_tasks(pipeline_config=pipeline_config),
  )

  flag_finished = {}
  for task in tasks:
    if "flag" in task.keys():
      flag_finished[task['flag']] = task['finished']

  monitor_flag = []

  for task in tasks:
    if task['finished']:
      continue

    if "flag" in task.keys():
      monitor_flag.append(task["flag"])

    deps = [d for d in task['dependency'] if not flag_finished[d]]
    if len(deps) == 0:
      deps_arg = ""
    else:
      deps_arg = "-d " + ','.join([f"afterok:${d}" for d in deps])

    if task['type'] == "download_preprocessing":
      bash_script += f"""
# download and preprocessing
{task['flag']}=$(sbatch $commonargs {deps_arg} -J {video_id}_{task['name']} $download_preprocessing_args --time=$download_preprocessing_time --output=$LOG_DIR/$VIDEO_ID/download_preprocess_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/download_preprocessing.sbatch)
"""
    elif task['type'] == "render":
      bash_script += f"""# render video
export TARGET_RENDER_REL_PATH={task['rel_path']}
export TARGET_RENDER_LABEL_SPACE={task['label_space']}
export TARGET_RENDER_VIDEO_NAME={task['video_render_name']}""" + f"""
sbatch $commonargs -J {video_id}_{task['name']} $video_render_args {deps_arg} --time=$video_render_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/render.sbatch
"""

    elif task['type'] == "gsam":
      bash_script += f"""
# grounded sam
export GSAM_OUTPUT_FOLDER={task['output_folder_args']}
export FLIP={'--flip' if task['flip'] else ''}
{task['flag']}=$(sbatch $commonargs $gsam_args -J {video_id}_{task['name']} {deps_arg} --time=$gsam_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/gsam.sbatch)
"""

    elif task['type'] == "internimage":
      bash_script += f"""
# internimage
export INTERN_OUTPUT_FOLDER={task['output_folder_args']}
export FLIP={'--flip' if task['flip'] else ''}
{task['flag']}=$(sbatch $commonargs $intern_args -J {video_id}_{task['name']} {deps_arg} --time=$intern_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/intern.sbatch)
"""

    elif task['type'] == "mask3d":
      bash_script += f"""
# mask3d
export MASK3D_OUTPUT_FOLDER={task['output_folder_args']}
export FLIP={'--flip' if task['flip'] else ''}
export SEED={task['seed']}
{task['flag']}=$(sbatch $commonargs $mask3d_args -J {video_id}_{task['name']} {deps_arg} --time=$mask3d_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/mask3d.sbatch)
"""

    elif task['type'] == "cmx":
      bash_script += f"""
# cmx
export CMX_OUTPUT_FOLDER={task['output_folder_args']}
export FLIP={'--flip' if task['flip'] else ''}
{task['flag']}=$(sbatch $commonargs $cmx_args -J {video_id}_{task['name']} {deps_arg} --time=$cmx_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/cmx.sbatch)
"""

    elif task['type'] == "ovseg":
      bash_script += f"""
# ovseg
export OVSEG_OUTPUT_FOLDER={task['output_folder_args']}
export FLIP={'--flip' if task['flip'] else ''}
{task['flag']}=$(sbatch $commonargs $ovseg_args -J {video_id}_{task['name']} {deps_arg} --time=$ovseg_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/ovseg.sbatch)
"""

    elif task['type'] == "hha":
      bash_script += f"""
# hha
{task['flag']}=$(sbatch $commonargs $hha_args -J {video_id}_{task['name']} {deps_arg} --time=$hha_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/hha.sbatch)
"""

    elif task['type'] == "omnidepth":
      bash_script += f"""
# omnidepth
{task['flag']}=$(sbatch $commonargs $omnidata_args -J {video_id}_{task['name']} {deps_arg} --time=$omnidata_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/omnidepth.sbatch)
"""

    elif task['type'] == "consensus":
      bash_script += f"""
# consensus
{task['flag']}=$(sbatch $commonargs $consensus_args -J {video_id}_{task['name']} {deps_arg} --time=$consensus_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/consensus.sbatch)
"""

    elif task['type'] == "point_lifting":
      bash_script += f"""
# point lifting
{task['flag']}=$(sbatch $commonargs $point_lifting_args -J {video_id}_{task['name']} {deps_arg} --time=$point_lifting_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/point_lifting.sbatch)
"""

    elif task['type'] == "sdfstudio_train":
      bash_script += f"""
# sdfstudio train
{task['flag']}=$(sbatch $commonargs $sdfstudio_train_args -J {video_id}_{task['name']} {deps_arg} --time=$sdfstudio_train_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/sdfstudio_train.sbatch)
"""

    elif task['type'] == "sdfstudio_extract":
      bash_script += f"""
# sdfstudio extract
{task['flag']}=$(sbatch $commonargs $sdfstudio_extract_args -J {video_id}_{task['name']} {deps_arg} --time=$sdfstudio_extract_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/sdfstudio_extract.sbatch)
"""

    elif task['type'] == "sdfstudio_render":
      bash_script += f"""
# sdfstudio render
{task['flag']}=$(sbatch $commonargs $sdfstudio_render_args -J {video_id}_{task['name']} {deps_arg} --time=$sdfstudio_render_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/sdfstudio_render.sbatch)
"""

    elif task['type'] == "sdfstudio_post":
      bash_script += f"""
# post processing, after rendering, delete unnesesary files
sbatch $commonargs $post_processing_args -J {video_id}_{task['name']} {deps_arg} --time=$post_processing_time --output=$LOG_DIR/$VIDEO_ID/{task['name']}_%j.out $LABELMAKER_REPO/pipeline/subtask_scripts/sdfstudio_post.sbatch
"""

  # stats about each task
  if len(monitor_flag) != 0:
    monitor_deps_arg = "-d " + ','.join([f"afterok:${d}" for d in monitor_flag])
    bash_script += f"""
export MONITOR_JOBIDS=({' '.join([f"${item}" for item in monitor_flag])})
"""
    bash_script += f"""
sbatch $commonargs $post_processing_args {monitor_deps_arg} -J {video_id}_stats --output=$LOG_DIR/$VIDEO_ID/final.out --time=$post_processing_time  $LABELMAKER_REPO/pipeline/subtask_scripts/stats.sbatch
"""

  with open(os.path.join(os.path.dirname(__file__), '../temp_slurm_submit.sh'),
            'w') as f:
    f.write(bash_script)


def arg_parser():
  parser = argparse.ArgumentParser(description='Check pipeline.')
  parser.add_argument(
      '--root_dir',
      type=str,
      required=True,
  )
  parser.add_argument(
      '--fold',
      type=str,
      required=True,
  )
  parser.add_argument(
      '--video_id',
      type=str,
      required=True,
  )
  parser.add_argument(
      '--num_images',
      type=int,
      required=True,
  )
  parser.add_argument(
      '--sdfstudio_gpu_type',
      type=str,
      required=True,
  )
  return parser.parse_args()


if __name__ == '__main__':
  args = arg_parser()
  submit(
      root_dir=args.root_dir,
      fold=args.fold,
      video_id=args.video_id,
      num_images=args.num_images,
      sdfstudio_gpu_type=args.sdfstudio_gpu_type,
  )

  # submit(
  #     root_dir="/cluster/project/cvg/labelmaker/ARKitScene_LabelMaker",
  #     video_id="42897688",
  #     fold="Validation",
  #     num_images=5726,
  #     sdfstudio_gpu_type="3090",
  # )
