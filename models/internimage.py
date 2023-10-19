# import internimage repo
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import gin
import mmcv
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core import get_classes, get_palette

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'InternImage',
                 'segmentation'))

import mmcv_custom  # this is actually needed for correct model registry
import mmseg_custom

logging.basicConfig(level="INFO")
log = logging.getLogger('InternImage Segmentation')


def load_internimage():
  config_file = os.path.join(os.path.dirname(__file__), '..', '3rdparty',
                             'InternImage', 'segmentation', 'configs', 'ade20k',
                             'upernet_internimage_h_896_160k_ade20k.py')
  checkpoint_file = os.path.join(os.path.dirname(__file__), '..', '3rdparty',
                                 'upernet_internimage_h_896_160k_ade20k.pth')

  # build the model from a config file and a checkpoint file
  model = init_segmentor(config_file, checkpoint=None, device='cuda:0')
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
def run(input_dir, output_dir, flip=False):

  output_dir = output_dir + '_flip' if flip else output_dir

  # check if output directory exists
  shutil.rmtree(output_dir, ignore_errors=True)
  output_dir.mkdir(exist_ok=False)

  log.info('[internimage] loading model')
  model = load_internimage()
  log.info(f'[internimage] running inference in {str(input_dir)}')
  print(f'[internimage] running inference in {str(input_dir)}', flush=True)

  input_files = input_dir.glob('*')
  input_files = sorted(input_files, key=lambda x: int(x.stem.split('_')[-1]))

  for file in tqdm(input_files):
    img = mmcv.imread(file)
    result = inference_segmentor(model, img)
    cv2.imwrite(str(output_dir / f'{file.stem}.png'), result[0])


# all models should have this command line interface
def arg_parser():
  parser = argparse.ArgumentParser(description='InternImage Segmentation')
  parser.add_argument('--workspace',
                      type=str,
                      required=True,
                      help='Path to workspace directory')
  parser.add_argument('--input',
                      type=str,
                      default='color',
                      help='Name of input directory in the workspace directory')
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/ade20k_internimage_1',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version'
  )
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


def main(args):

  # check if workspace exists
  workspace = Path(args.workspace)
  assert workspace.exists() and workspace.is_dir()

  # check if input directory exists
  input_dir = workspace / args.input
  assert input_dir.exists() and input_dir.is_dir()

  # check if output directory exists
  output_dir = workspace / args.output
  if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)

  run(input_dir, output_dir)


if __name__ == '__main__':
  args = arg_parser()
  main(args)
