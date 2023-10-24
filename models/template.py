import argparse
import os
import shutil
from pathlib import Path
from typing import Union

import gin
import torch


@gin.configurable
def run(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    device: Union[str, torch.device] = 'cpu',
):
  # check if workspace exists
  scene_dir = Path(scene_dir)
  assert scene_dir.exists() and scene_dir.is_dir()
  output_folder = Path(output_folder)

  # convert str to Path object
  input_color_dir = scene_dir / 'color'
  assert input_color_dir.exists() and input_color_dir.is_dir()

  input_depth_dir = scene_dir / 'depth'
  assert input_depth_dir.exists() and input_depth_dir.is_dir()

  input_pose_dir = scene_dir / 'pose'
  assert input_pose_dir.exists() and input_pose_dir.is_dir()

  input_intrinsic_dir = scene_dir / 'intrinsic'
  assert input_intrinsic_dir.exists() and input_intrinsic_dir.is_dir()

  # check if output directory exists
  output_dir = scene_dir / output_folder
  output_dir = output_dir + '_flip' if flip else output_dir
  shutil.rmtree(output_dir, ignore_errors=True)

  # makedirs instead of mkdir
  os.makedirs(str(output_dir), exist_ok=False)


def arg_parser():
  parser = argparse.ArgumentParser(description='Blablabla')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help='Path to workspace directory',
  )

  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/blablablag',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version',
  )
  parser.add_argument('--config', help='Name of config file')

  return parser.parse_args()


if __name__ == '__main__':
  args = arg_parser()
  gin.parse_config_file(args.config)
  run(input_dir=input_dir, output_dir=output_dir)
