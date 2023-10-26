import argparse
import logging
import os
import shutil
import sys
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Union

import cv2
import gin
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core import get_classes, get_palette
from tqdm import tqdm

sys.path.append(
    os.path.join(os.path.dirname(__file__), '../3rdparty',
                 'RGBX_Semantic_Segmentation'))

from config import config
from dataloader.dataloader import ValPre
from dataloader.RGBXDataset import RGBXDataset
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import compute_score, hist_info
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img

from models.builder import EncoderDecoder as segmodel

logging.basicConfig(level="INFO")
log = logging.getLogger('CMX Segmentation')


def load_cmx(device: Union[str, torch.device] = 'cuda:0'):
  device = str(device)
  assert device[:4] == 'cuda'
  try:
    device_id = str(int(device[:][-1]))
  except:
    assert False, "device should be cuda device and in format of 'cuda:xx'."

  log.info('loading model')
  checkpoint_file = abspath(
      join(dirname(__file__), '../checkpoints/NYUDV2_CMX+Segformer-B2.pth'))
  network = segmodel(cfg=config,
                     criterion=None,
                     norm_layer=torch.nn.BatchNorm2d)
  eval_source = abspath(
      join(dirname(__file__),
           '../3rdparty/RGBX_Semantic_Segmentation/empty.txt'))
  data_setting = {
      'rgb_root': config.rgb_root_folder,
      'rgb_format': config.rgb_format,
      'gt_root': config.gt_root_folder,
      'gt_format': config.gt_format,
      'transform_gt': config.gt_transform,
      'x_root': config.x_root_folder,
      'x_format': config.x_format,
      'x_single_channel': config.x_is_single_channel,
      'class_names': config.class_names,
      'train_source': config.train_source,
      'eval_source': eval_source,
      'class_names': config.class_names
  }
  val_pre = ValPre()
  dataset = RGBXDataset(data_setting, 'val', val_pre)
  evaluator = Evaluator(
      dataset=dataset,
      class_num=40,
      norm_mean=config.norm_mean,
      norm_std=config.norm_std,
      network=network,
      multi_scales=config.eval_scale_array,
      is_flip=config.eval_flip,
      devices=parse_devices(device_id),
  )
  evaluator.compute_metric = lambda x: str()
  evaluator.run('mmseg', checkpoint_file, '/dev/null', '/tmp/fakelog')
  return evaluator


@gin.configurable
def run(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    device: Union[str, torch.device] = 'cuda:0',
    confidence_threshold=0.995,
    flip=False,
):

  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  assert scene_dir.exists() and scene_dir.is_dir()

  input_hha_dir = scene_dir / 'intermediate/hha'
  assert input_hha_dir.exists() and input_hha_dir.is_dir()

  input_color_dir = scene_dir / 'color'
  assert input_color_dir.exists() and input_color_dir.is_dir()

  assert len(list(input_hha_dir.iterdir())) == len(
      list(input_color_dir.iterdir()))

  output_dir = scene_dir / output_folder
  output_dir = output_dir + '_flip' if flip else output_dir
  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)

  evaluator = load_cmx(device=device)
  log.info('[cmx] running inference')

  keys = [p.stem for p in input_color_dir.glob('*.jpg')]
  for k in tqdm(keys):
    img = cv2.imread(str(input_color_dir / f'{k}.jpg'))[..., ::-1]
    hha = cv2.imread(str(input_hha_dir / f'{k}.png'))

    if flip:
      img = img[:, ::-1]
      hha = hha[:, ::-1]
    pred = evaluator.sliding_eval_rgbX(
        img,
        hha,
        config.eval_crop_size,
        config.eval_stride_rate,
        device=device,
    )

    pred = pred + 1
    if flip:
      pred = pred[:, ::-1]
    cv2.imwrite(str(output_dir / f'{k}.png'), pred)


def arg_parser():
  parser = argparse.ArgumentParser(description='CMX Segmentation')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help=
      'Path to workspace directory. There should be "color" and "intermediate/hha" folder inside.',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/nyu40_cmx_1',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version'
  )
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == "__main__":
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  run(scene_dir=args.workspace, output_folder=args.output)
