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

from labelmaker.label_data import get_wordnet
from labelmaker.utils import get_keys, get_unprocessed_keys, is_uint16_img, remove_files_by_keys

sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from grounded_sam import load_grounded_sam, process_image, setup_seeds

logging.basicConfig(level="INFO")
log = logging.getLogger('Grounded SAM Segmentation')


def get_gsam_dask_task_runner(mode: str = 'local'):
  """Dask is a module to lauch SLURM jobs, it can also do local computation.
  """
  assert mode in ['local', 'slurm', 'slurm_singularity']
  if mode == 'local':
    task_runner = DaskTaskRunner(
        cluster_class="distributed.LocalCluster",
        cluster_kwargs={
            "n_workers": 1,
            "threads_per_worker": 1,
            "memory_limit": "36GiB",
        },
    )
  elif mode == "slurm_singularity":
    task_runner = DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "n_workers":
                1,
            # "memory":
            #     "44GiB",
            # "job_cpu":
            #     3,
            # "cores":
            #     2,
            "memory":
                "36GiB",
            "job_cpu":
                1,
            "cores":
                1,
            "interface":
                "access",  # possibles are access, lo, eth0, eth1, eth3
            "walltime":
                "00:10:00",
            "job_extra_directives": [
                "--gpus=rtx_3090:1",
                "--mem-per-cpu=36G",
                "--output=/cluster/home/guanji/LabelMaker/job%j.out",
            ],
            "job_directives_skip": ["--mem"],
            "job_script_prologue": [
                "module load eth_proxy",
                'export PATH="/cluster/project/cvg/labelmaker/miniconda3/bin:${PATH}"',
                'env_name=labelmaker', 'eval "$(conda shell.bash hook)"',
                'conda activate $env_name',
                'conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"',
                'conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"',
                'export AM_I_DOCKER=1',
                'export CUDA_HOST_COMPILER="$conda_home/bin/gcc"',
                'export CUDA_PATH="$conda_home"', 'export CUDA_HOME=$CUDA_PATH',
                'export NLTK_DATA="${ENV_FOLDER}/../3rdparty/nltk_data"',
                'export PYTHONPATH=/cluster/home/guanji/LabelMaker/models:${PYTHONPATH}'
                # "module load gcc/11.4.0 cuda/12.1.1 eth_proxy",
                # "LABELMAKER_REPO=/cluster/home/guanji/LabelMaker",
                # "scene=room_0",
                # "sequence=1",
                # "source_dir=/cluster/scratch/guanji/Replica_Dataset_Semantic_Nerf/${scene}/Sequence_${sequence}",
                # "target_dir=$SCRATCH/replica_${scene}_${sequence}",
                # "mkdir -p $target_dir",
                # "mkdir -p $TMPDIR/.cache",
            ],
            # "python":

            # ' '.join([
            #     "singularity exec --nv",
            #     "--bind /cluster/project/cvg/labelmaker/checkpoints:/LabelMaker/checkpoints",
            #     "--bind $LABELMAKER_REPO/env_v2:/LabelMaker/env_v2",
            #     "--bind $LABELMAKER_REPO/labelmaker:/LabelMaker/labelmaker",
            #     "--bind $LABELMAKER_REPO/testing:/LabelMaker/testing",
            #     "--bind $LABELMAKER_REPO/models:/LabelMaker/models",
            #     "--bind $LABELMAKER_REPO/scripts:/LabelMaker/scripts"
            #     "--bind $LABELMAKER_REPO/.gitmodules:/LabelMaker/.gitmodules",
            #     "--bind $TMPDIR/.cache:$HOME/.cache",
            #     "--bind $source_dir:/source",
            #     "--bind $target_dir:/target",
            #     "/cluster/project/cvg/labelmaker/labelmaker_20231227.simg",
            #     "/miniconda3/envs/labelmaker/bin/python",
            # ])
        },
    )
    # print(task_runner._cluster.job_script())
  else:
    raise NotImplementedError

  return task_runner


@task(name="Grounded SAM preparation", retries=5, retry_delay_seconds=1.0)
def wrap_load(device):
  # load model
  ram_ckpt = abspath(
      join(__file__, '../..', 'checkpoints', 'ram_swin_large_14m.pth'))
  groundingdino_ckpt = abspath(
      join(__file__, '../..', 'checkpoints', 'groundingdino_swint_ogc.pth'))
  sam_hq_ckpt = abspath(
      join(__file__, '../..', 'checkpoints', 'sam_hq_vit_h.pth'))
  (
      ram,
      ram_transform,
      grounding_dino,
      grounding_dino_transform,
      sam_predictor,
      sam_transform,
  ) = load_grounded_sam(
      ram_ckpt=ram_ckpt,
      groundingdino_ckpt=groundingdino_ckpt,
      sam_hq_ckpt=sam_hq_ckpt,
      device=device,
  )

  # load compact label to wordnet id mapping
  wordnet = get_wordnet()
  wordnet.sort(key=lambda x: x.get('id'))
  compact_to_wordnet = [int(item['id']) for item in wordnet]
  compact_to_wordnet = np.array(compact_to_wordnet)

  return (
      ram,
      ram_transform,
      grounding_dino,
      grounding_dino_transform,
      sam_predictor,
      sam_transform,
      compact_to_wordnet,
  )


@task(name="Atomic GSAM Inference", retries=5, retry_delay_seconds=1.0)
def wrap_process_single_image(
    loads,
    img_path,
    save_path,
    device,
    box_threshold,
    text_threshold,
    iou_threshold,
    sam_defect_threshold,
    flip=False,
    skip=False,
):
  if skip:
    return

  (
      ram,
      ram_transform,
      grounding_dino,
      grounding_dino_transform,
      sam_predictor,
      sam_transform,
      compact_to_wordnet,
  ) = loads
  label = process_image(
      ram=ram,
      ram_transform=ram_transform,
      grounding_dino=grounding_dino,
      grounding_dino_transform=grounding_dino_transform,
      sam_predictor=sam_predictor,
      sam_transform=sam_transform,
      img_path=img_path,
      device=device,
      box_threshold=box_threshold,
      text_threshold=text_threshold,
      iou_threshold=iou_threshold,
      sam_defect_threshold=sam_defect_threshold,
      flip=flip,
  )
  cv2.imwrite(str(save_path), compact_to_wordnet[label].astype(np.uint16))


@flow(
    task_runner=get_gsam_dask_task_runner(
        mode=os.environ.get('LABELMAKER_CLUSTER_TYPE', 'slurm_singularity')),
    retries=10,
    retry_delay_seconds=1.0,
)
def run_grounded_sam_pipeline(
    scene_dir,
    output_folder,
    device='cuda:0',
    box_threshold=0.25,
    text_threshold=0.2,
    iou_threshold=0.5,
    sam_defect_threshold=30,
    flip=False,
    clean_run=False,
):
  # convert str to Path object
  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  # check if scene_dir exists
  assert scene_dir.exists() and scene_dir.is_dir()

  input_color_dir = scene_dir / 'color'
  assert input_color_dir.exists() and input_color_dir.is_dir()

  output_dir = scene_dir / output_folder
  output_dir = Path(str(output_dir) + '_flip') if flip else output_dir

  if clean_run:
    shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=True)

  log.info(f'[Grounded SAM] inference in {str(input_color_dir)}')
  keys = get_keys(scene_dir=scene_dir)

  # get the list of keys that is not processed
  unproc_keys = get_unprocessed_keys(
      keys=keys,
      target_dir=output_dir,
      target_file_template='{k:06d}.png',
      validity_fn=is_uint16_img,
  )
  # if unprocessing is due to file read failure, we have to delete those files
  remove_files_by_keys(
      unproc_keys,
      target_dir=output_dir,
      target_file_template='{k:06d}.png',
  )

  # loading models, using submit creates a dependency of loading model first, then inference.
  log.info('[Grounded SAM] loading model')
  loads = wrap_load.submit(device=device)
  log.info('[Grounded SAM] model loaded!')

  # launching tasks
  input_files = [input_color_dir / '{k:06d}.jpg'.format(k=key) for key in keys]
  output_files = [output_dir / '{k:06d}.png'.format(k=key) for key in keys]

  for k, input_file, output_file in zip(keys, input_files, output_files):
    wrap_process_single_image.submit(
        loads=loads,
        img_path=input_file,
        save_path=output_file,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        iou_threshold=iou_threshold,
        sam_defect_threshold=sam_defect_threshold,
        flip=flip,
        skip=k not in unproc_keys,
    )


if __name__ == "__main__":
  #   10G of memory use when processing an image of 640x480 giving it 16g ram is enough
  # module load gcc/8.2.0 cuda/11.3.1 python/3.9.9 ffmpeg/3.2.4 openblas/0.3.20 sqlite/3.35.5
  #  /cluster/project/cvg/labelmaker/labelmaker_venv/bin/python pipeline/grounded_sam_pipeline.py
  run_grounded_sam_pipeline(
      scene_dir='/cluster/project/cvg/labelmaker/replica/room_0_1',
      output_folder='gsam_prefect_test',
      clean_run=False,
      flip=True,
  )
