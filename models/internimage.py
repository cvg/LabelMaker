import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Union

import cv2
import gin
import mmcv
import torch
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core import get_classes, get_palette
from tqdm import tqdm

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'InternImage',
                 'segmentation'))

import mmcv_custom  # this is actually needed for correct model registry
import mmseg_custom

logging.basicConfig(level="INFO")
log = logging.getLogger('InternImage Segmentation')


def load_internimage(device: Union[str, torch.device],):
  config_file = os.path.join(os.path.dirname(__file__), '..', '3rdparty',
                             'InternImage', 'segmentation', 'configs', 'ade20k',
                             'upernet_internimage_h_896_160k_ade20k.py')
  checkpoint_file = os.path.join(os.path.dirname(__file__), '..', 'checkpoints',
                                 'upernet_internimage_h_896_160k_ade20k.pth')

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
    device: Union[str, torch.device] = 'cpu',
    flip: bool = False,
):
  # convert str to Path object
  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  assert scene_dir.exists() and scene_dir.is_dir()

  input_color_dir = scene_dir / 'color'
  assert input_color_dir.exists() and input_color_dir.is_dir()

  output_dir = scene_dir / output_folder
  output_dir = output_dir + '_flip' if flip else output_dir

  # check if output directory exists
  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)

  log.info('[internimage] loading model')
  model = load_internimage(device=device)
  log.info(f'[internimage] running inference in {str(input_color_dir)}')
  print(f'[internimage] running inference in {str(input_color_dir)}',
        flush=True)

  input_files = input_color_dir.glob('*')
  input_files = sorted(input_files, key=lambda x: int(x.stem.split('_')[-1]))

  for file in tqdm(input_files):
    img = mmcv.imread(file)
    result = inference_segmentor(model, img)
    cv2.imwrite(str(output_dir / f'{file.stem}.png'), result[0])


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
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == '__main__':
  args = arg_parser()
  
  if args.config is not None:
    gin.parse_config_file(args.config)
    
  run(scene_dir=args.workspace, output_folder=args.output)
