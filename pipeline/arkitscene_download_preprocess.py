import logging
import os
import shutil
import sys
from os.path import abspath, dirname, join
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from prefect import flow, task
from prefect_dask.task_runners import DaskTaskRunner

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from labelmaker.utils import check_scene_in_labelmaker_format

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from arkitscene_download import download_necessary_data, get_fold, get_scene_download_dir
from arkitscenes2labelmaker import process_arkit


def get_taskrunner_preprocess():
  """
    get the task runner, paramters like image resolution and size passed by environment variables.
  """
  mode = os.environ.get('LABELMAKER_SERVER_TYPE', 'local')
  assert mode in ['local', 'slurm']

  taskrunner = None
  if mode == 'local':
    taskrunner = DaskTaskRunner(
        cluster_class="distributed.LocalCluster",
        cluster_kwargs={
            "n_workers": 1,
            "threads_per_worker": 1,
            "memory_limit": "2GiB",
        },
    )
  elif mode == 'slurm':
    taskrunner = DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "n_workers":
                1,
            "memory":
                "8GiB",
            "job_cpu":
                1,
            "cores":
                1,
            "interface":
                "access",  # possibles are access, lo, eth0, eth1, eth3
            "walltime":
                "1:00:00",
            "job_extra_directives": [
                "--mem-per-cpu=8G",
                "--output=/cluster/home/guanji/LabelMaker/preprocess_%j.out",
            ],
            "job_directives_skip": ["--mem"],
            "job_script_prologue": [
                "module load eth_proxy",
                'export PATH="/cluster/project/cvg/labelmaker/miniconda3/bin:${PATH}"',
                'env_name=labelmaker', 'eval "$(conda shell.bash hook)"',
                'conda activate $env_name',
                'conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"',
                'conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"',
                'export PYTHONPATH=/cluster/home/guanji/LabelMaker:${PYTHONPATH}'
            ],
        },
    )
  else:
    raise ValueError

  return taskrunner


@task(
    name="ARKitScene_download",
    task_run_name="ARKitScene_download-{video_id}",
    retries=5,
    retry_delay_seconds=1,
)
def wrap_download(
    video_id: str,
    download_dir: str,
):
  # remove the traget folder
  fold = get_fold(video_id)
  scene_dir = get_scene_download_dir(
      video_id=video_id,
      fold=fold,
      download_dir=download_dir,
  )

  if os.path.exists(scene_dir):
    shutil.rmtree(scene_dir, ignore_errors=True)

  download_necessary_data(
      video_id=video_id,
      download_dir=download_dir,
  )


@task(
    name="ARKitScene_preprocess",
    task_run_name="ARKitScene_preprocess-{video_id}",
    retries=5,
    retry_delay_seconds=1,
)
def wrap_preprocess(
    download_finish_flag,
    video_id: str,
    download_dir: str,
    target_dir: str,
    sdf_trunc: float,
    voxel_length: float,
    depth_trunc: float,
):
  fold = get_fold(video_id)
  scene_dir = get_scene_download_dir(
      video_id=video_id,
      fold=fold,
      download_dir=download_dir,
  )

  process_arkit(
      scan_dir=scene_dir,
      target_dir=target_dir,
      sdf_trunc=sdf_trunc,
      voxel_length=voxel_length,
      depth_trunc=depth_trunc,
  )


@task(
    name="ARKitScene_check",
    task_run_name="ARKitScene_check-{video_id}",
)
def wrap_check(
    preprocess_finish_flag,
    video_id: str,
    target_dir: str,
):
  check_scene_in_labelmaker_format(scene_dir=target_dir)


@flow(
    task_runner=get_taskrunner_preprocess(),
    retries=10,
    retry_delay_seconds=1.0,
)
def arkitscene_download_and_preprocess(
    download_dir: str,
    video_id: str,
    target_dir: str,
    sdf_trunc: float,
    voxel_length: float,
    depth_trunc: float,
):

  download_finish_flag = wrap_download.submit(
      video_id=video_id,
      download_dir=download_dir,
  )
  process_finish_flag = wrap_preprocess(
      download_finish_flag=download_finish_flag,
      video_id=video_id,
      download_dir=download_dir,
      target_dir=target_dir,
      sdf_trunc=sdf_trunc,
      voxel_length=voxel_length,
      depth_trunc=depth_trunc,
  )
  wrap_check(
      preprocess_finish_flag=process_finish_flag,
      video_id=video_id,
      target_dir=target_dir,
  )


if __name__ == "__main__":
  # this is only test code
  mode = os.environ.get("LABELMAKER_SERVER_TYPE", 'local')
  if mode == 'slurm':
    arkitscene_download_and_preprocess(
        download_dir=os.environ.get('TMPDIR'),
        video_id='41098093',
        target_dir='',
        sdf_trunc=0.04,
        voxel_length=0.008,
        depth_trunc=3.0,
    )

  elif mode == 'local':
    arkitscene_download_and_preprocess(
        download_dir='/scratch/quanta/Datasets/ARKitScenes',
        video_id='41098093',
        target_dir='/scratch/quanta/Experiments/LabelMaker/41098093',
        sdf_trunc=0.04,
        voxel_length=0.008,
        depth_trunc=3.0,
    )

  else:
    raise ValueError
