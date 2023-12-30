import argparse
import os
import sys

import gin
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../3rdparty'))
from ARKitScenes.download_data import download_data


def get_fold(video_id: str):
  data = pd.read_csv(
      os.path.join(
          os.path.dirname(__file__),
          '../3rdparty',
          'ARKitScenes',
          'raw',
          'raw_train_val_splits.csv',
      ),
      index_col='video_id',
      dtype={'video_id': 'string'},
  )
  return data.loc[video_id]['fold']


def download_necessary_data(
    video_id: str,
    download_dir: str,
):
  download_data(
      dataset='raw',
      video_ids=[video_id],
      dataset_splits=[get_fold(video_id)],
      download_dir=download_dir,
      keep_zip=False,
      raw_dataset_assets=[
          'confidence',
          'lowres_depth',
          'lowres_wide.traj',
          'vga_wide',
          'vga_wide_intrinsics',
      ],
      should_download_laser_scanner_point_cloud=False,
  )


def get_scene_download_dir(
    video_id: str,
    fold: str,
    download_dir: str,
):
  return os.path.join(
      download_dir,
      'raw',
      fold,
      video_id,
  )


@gin.configurable
def run(
    video_id: str,
    download_dir: str,
):
  download_necessary_data(
      video_id=video_id,
      download_dir=download_dir,
  )


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--video_id", type=str)
  parser.add_argument("--download_dir", type=str)
  parser.add_argument('--config', help='Name of config file')

  return parser.parse_args()


if __name__ == "__main__":
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  run(
      video_id=args.video_id,
      download_dir=args.download_dir,
  )
