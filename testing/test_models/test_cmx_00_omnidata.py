import os
import sys
from os.path import abspath, dirname, exists, join
from pathlib import Path

import cv2
import numpy as np

sys.path.append(abspath(join(dirname(__file__), '../../models')))


def test_omnidata():
  from omnidata import run as run_omnidata

  scene_dir = Path(abspath(join(dirname(__file__), '../test_scan')))
  output_folder = 'intermediate/depth_omnidata_1'

  run_omnidata(
      scene_dir=scene_dir,
      output_folder=output_folder,
      device='cuda:0',
  )
