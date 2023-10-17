import os
import sys
from os.path import abspath, dirname, join

import gin

sys.path.append(
    os.path.join(os.path.dirname(__file__), '../mmseg',
                 'RGBX_Semantic_Segmentation'))
import argparse
import logging
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from config import config
from dataloader.dataloader import ValPre
from dataloader.RGBXDataset import RGBXDataset
from engine.evaluator import Evaluator
from engine.logger import get_logger
from tqdm import tqdm
from utils.metric import compute_score, hist_info
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core import get_classes, get_palette
from models.builder import EncoderDecoder as segmodel

logging.basicConfig(level="INFO")
log = logging.getLogger('CMX Segmentation')


def load_cmx():
  log.info('loading model')
  checkpoint_file = abspath(
      join(dirname(__file__), '../3rdparty/NYUDV2_CMX+Segformer-B2.pth'))
  network = segmodel(cfg=config,
                     criterion=None,
                     norm_layer=torch.nn.BatchNorm2d)
  eval_source = abspath(join(dirname(__file__), '../mmseg/empty.txt'))
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
  evaluator = Evaluator(dataset, 40, config.norm_mean, config.norm_std, network,
                        config.eval_scale_array, config.eval_flip,
                        parse_devices('0'))
  evaluator.compute_metric = lambda x: str()
  evaluator.run('mmseg', checkpoint_file, '/dev/null', '/tmp/fakelog')
  return evaluator


@gin.configurable
def run(args, confidence_threshold=0.995, flip=False):

  evaluator = load_cmx()

  workspace_dir = Path(args.workspace)

  result_dir = Path(args.workspace) / args.output
  if flip:
    result_dir = result_dir + '_flip'

  shutil.rmtree(result_dir, ignore_errors=True)
  result_dir.mkdir(exist_ok=False)
  log.info('[cmx] running inference')

  assert (workspace_dir / 'intermediate' / 'hha').exists()
  assert len(list((workspace_dir / 'intermediate' / 'hha').iterdir())) == len(
      list((workspace_dir / args.input).iterdir()))
  keys = [p.stem for p in (Path(args.workspace) / args.input).glob('*.jpg')]
  for k in tqdm(keys):
    img = cv2.imread(str(workspace_dir / args.input / f'{k}.jpg'))[..., ::-1]
    hha = cv2.imread(str(workspace_dir / 'intermediate/hha' / f'{k}.png'))

    if flip:
      img = img[:, ::-1]
      hha = hha[:, ::-1]
    pred = evaluator.sliding_eval_rgbX(img, hha, config.eval_crop_size,
                                       config.eval_stride_rate, 'cuda')

    pred = pred + 1
    if flip:
      pred = pred[:, ::-1]
    cv2.imwrite(str(result_dir / f'{k}.png'), pred)


def main(args):
  run(args)


def arg_parser():
  parser = argparse.ArgumentParser(description='CMX Segmentation')
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
      default='intermediate/nyu40_cmx_1',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version'
  )
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == "__main__":
  args = arg_parser()
  main(args)
