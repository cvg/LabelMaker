import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Union

import cv2
import gin
import mmcv
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core import get_classes, get_palette
from tqdm import tqdm

from labelmaker.utils import rotate_image, rotate_image_back

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'InternImage',
                 'segmentation'))

import mmcv_custom  # this is actually needed for correct model registry
import mmseg_custom

logging.basicConfig(level="INFO")
log = logging.getLogger('InternImage Segmentation')


def setup_seeds(seed):

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  cudnn.benchmark = False
  cudnn.deterministic = True


def load_internimage(device: Union[str, torch.device],):

  config_file = os.path.join(
      os.path.dirname(__file__), '..', '3rdparty', 'InternImage',
      'segmentation', 'configs', 'ade20k',
      'mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py')
  checkpoint_file = os.path.join(
      os.path.dirname(__file__), '..', 'checkpoints',
      'mask2former_internimage_h_896_80k_cocostuff2ade20k.pth')

  # build the model from a config file and a checkpoint file
  model = init_segmentor(config_file, checkpoint=None, device=device)
  checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')

  if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
  else:
    print('"CLASSES" not found in meta, use dataset.CLASSES instead')
    model.CLASSES = get_classes('ade20k')

  if 'PALETTE' in checkpoint.get('meta', {}):
    model.PALETTE = checkpoint['meta']['PALETTE']
  else:
    print('"PALETTE" not found in meta, use dataset.PALETTE instead')
    model.PALETTE = get_palette('ade20k')

  return model


@gin.configurable
def run(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    device: Union[str, torch.device] = 'cuda:0',
    flip: bool = False,
):
  # convert str to Path object
  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  assert scene_dir.exists() and scene_dir.is_dir()

  input_color_dir = scene_dir / 'color'
  assert input_color_dir.exists() and input_color_dir.is_dir()

  input_corres_pth = scene_dir / 'correspondence.json'
  assert input_corres_pth.exists() and input_corres_pth.is_file()

  corres_df = pd.read_json(
      str(input_corres_pth),
      dtype={
          'frame_id': 'String',
          'z_direction': "int",
      },
  ).set_index('frame_id')

  output_dir = scene_dir / output_folder
  output_dir = Path(str(output_dir) + '_flip') if flip else output_dir

  # check if output directory exists
  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)

  log.info('[internimage] loading model')
  model = load_internimage(device=device)
  log.info(f'[internimage] running inference in {str(input_color_dir)}')
  print(f'[internimage] running inference in {str(input_color_dir)}',
        flush=True)

  keys = [p.stem for p in input_color_dir.glob('*.jpg')]

  input_files = input_color_dir.glob('*')
  input_files = sorted(input_files, key=lambda x: int(x.stem.split('_')[-1]))

  # for file in tqdm(input_files):
  for k in tqdm(keys):
    img_path = input_color_dir / f'{k}.jpg'
    z_direction = corres_df['z_direction'].loc[[k]].item()

    img = mmcv.imread(img_path)
    img = rotate_image(img, z_direction=z_direction)

    if flip:
      img = img[:, ::-1]

    result = inference_segmentor(model, img)[0]
    if flip:
      result = result[:, ::-1]
    result = rotate_image_back(result, z_direction=z_direction)

    cv2.imwrite(str(output_dir / f'{k}.png'), result.astype(np.uint8))


# all models should have this command line interface
def arg_parser():
  parser = argparse.ArgumentParser(description='InternImage Segmentation')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help='Path to workspace directory. There should be a "color" folder.',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/ade20k_internimage_1',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version',
  )
  parser.add_argument('--seed', type=int, default=42, help='random seed')
  parser.add_argument(
      '--flip',
      action="store_true",
      help='Flip the input image, this is part of test time augmentation.',
  )
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == '__main__':
  args = arg_parser()

  if args.config is not None:
    gin.parse_config_file(args.config)

  setup_seeds(seed=args.seed)
  run(scene_dir=args.workspace, output_folder=args.output, flip=args.flip)
