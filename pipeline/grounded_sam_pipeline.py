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
from tqdm import tqdm

from labelmaker.label_data import get_wordnet
from labelmaker.utils import get_keys, is_uint16_img, unprocessed_keys

sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from grounded_sam import load_grounded_sam, process_image, setup_seeds

logging.basicConfig(level="INFO")
log = logging.getLogger('Grounded SAM Segmentation')


def get_gsam_dask_task_runner(mode: str = 'local'):
  """Dask is a module to lauch SLURM jobs, it can also do local computation.
  """
  assert mode in ['local', 'slurm']
  if mode == 'local':
    return DaskTaskRunner(
        cluster_class="distributed.LocalCluster",
        cluster_kwargs={
            "n_workers": 1,
            "threads_per_worker": 1,
            "memory_limit": "16GiB",
        },
    )
  else:
    raise NotImplementedError


@task
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


@task
def wrap_process_single_image(
    # ram,
    # ram_transform,
    # grounding_dino,
    # grounding_dino_transform,
    # sam_predictor,
    # sam_transform,
    # compact_to_wordnet,
    loads,
    img_path,
    save_path,
    device,
    box_threshold,
    text_threshold,
    iou_threshold,
    sam_defect_threshold,
    flip=False,
):
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


@flow(task_runner=get_gsam_dask_task_runner(
    mode=os.environ.get('LABELMAKER_CLUSTER_TYPE', 'local')))
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

  # get the list of keys that is not processed
  keys = get_keys(scene_dir=scene_dir)
  unproc_keys = unprocessed_keys(
      keys=keys,
      target_dir=output_dir,
      target_file_template='{k:06d}.png',
      validity_fn=is_uint16_img,
  )

  if len(unproc_keys) == 0:
    return  # this pipeline is finished

  input_files = [
      input_color_dir / '{k:06d}.jpg'.format(k=key) for key in unproc_keys
  ]
  output_files = [
      output_dir / '{k:06d}.png'.format(k=key) for key in unproc_keys
  ]
  log.info(f'[Grounded SAM] inference in {str(input_color_dir)}')

  # loading models, using submit creates a dependency of loading model first, then inference.
  log.info('[Grounded SAM] loading model')
  #   (
  #       ram,
  #       ram_transform,
  #       grounding_dino,
  #       grounding_dino_transform,
  #       sam_predictor,
  #       sam_transform,
  #       compact_to_wordnet,
  #   ) = wrap_load(device=device)
  loads = wrap_load.submit(device=device)
  log.info('[Grounded SAM] model loaded!')

  for input_file, output_file in zip(input_files, output_files):
    wrap_process_single_image.submit(
        # ram=ram,
        # ram_transform=ram_transform,
        # grounding_dino=grounding_dino,
        # grounding_dino_transform=grounding_dino_transform,
        # sam_predictor=sam_predictor,
        # sam_transform=sam_transform,
        # compact_to_wordnet=compact_to_wordnet,
        loads=loads,
        img_path=input_file,
        save_path=output_file,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        iou_threshold=iou_threshold,
        sam_defect_threshold=sam_defect_threshold,
        flip=flip,
    )


if __name__ == "__main__":
  #   10G of memory use when processing an image of 640x480 giving it 16g ram is enough
  run_grounded_sam_pipeline(
      scene_dir=
      '/scratch/quanta/Experiments/LabelMaker/replica_room_0_squence_1',
      output_folder='gsam_prefect_test',
      clean_run=False,
      flip=True,
  )

# python models/grounded_sam.py \
#     --workspace /scratch/quanta/Experiments/LabelMaker/replica_room_0_squence_1 \
#     --output gsam_prefect_test \
#     --flip
