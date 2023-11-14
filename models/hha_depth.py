import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Union

import cv2
import gin
import numpy as np
import torch
from hha.getHHA import getHHA
from joblib import Parallel, delayed
from tqdm import tqdm

logging.basicConfig(level="INFO")
log = logging.getLogger('Depth to HHA conversion')


@gin.configurable
def run(
    scene_dir: Union[str, Path],
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    n_jobs=1,
):

  scene_dir = Path(scene_dir)
  input_folder = Path(input_folder)
  output_folder = Path(output_folder)

  assert scene_dir.exists() and scene_dir.is_dir()

  input_depth_dir = scene_dir / 'depth'
  assert input_depth_dir.exists() and input_depth_dir.is_dir()

  input_intrinsic_dir = scene_dir / 'intrinsic'
  assert input_intrinsic_dir.exists() and input_intrinsic_dir.is_dir()

  omnidata_depth_dir = scene_dir / input_folder
  assert omnidata_depth_dir.exists() and omnidata_depth_dir.is_dir()

  assert len(list(input_depth_dir.iterdir())) == len(
      list(omnidata_depth_dir.iterdir()))

  output_dir = scene_dir / output_folder
  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)

  log.info(f'running depth to hha conversion for scene {scene_dir}')

  def depth_to_hha(k):
    intrinsics = np.loadtxt(str(input_intrinsic_dir / f'{k}.txt'))[:3, :3]
    orig_depth = cv2.imread(str(input_depth_dir / f'{k}.png'),
                            cv2.COLOR_BGR2GRAY) / 1000
    omni_depth = cv2.imread(str(omnidata_depth_dir / f'{k}.png'),
                            cv2.COLOR_BGR2GRAY) / 1000
    hha = getHHA(intrinsics, omni_depth, orig_depth)
    cv2.imwrite(str(output_dir / f'{k}.png'), hha)

  keys = [p.stem for p in (scene_dir / 'depth').glob('*.png')]
  if n_jobs > 1:
    Parallel(n_jobs=n_jobs)(delayed(depth_to_hha)(k) for k in tqdm(keys))
  else:
    for k in tqdm(keys):
      depth_to_hha(k)


def arg_parser():
  parser = argparse.ArgumentParser(description='HHA')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help=
      'Path to workspace directory. There should be a "depth" and "instrinsic" folder',
  )
  parser.add_argument(
      '--input',
      type=str,
      default='intermediate/depth_omnidata_1',
      help='Name of input directory in the workspace directory',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/hha',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version',
  )
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == "__main__":
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  run(
      scene_dir=args.workspace,
      input_folder=args.input,
      output_folder=args.output,
  )
