import os
import sys

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
