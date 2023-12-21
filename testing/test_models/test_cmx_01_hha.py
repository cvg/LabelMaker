import os
import sys
from os.path import abspath, dirname, exists, join
from pathlib import Path

import cv2
import numpy as np

sys.path.append(abspath(join(dirname(__file__), '../../models')))


def test_hha():
  from hha_depth import run as run_hha

  scene_dir = Path(abspath(join(dirname(__file__), '../test_scan')))
  input_folder = 'intermediate/depth_omnidata_1'
  output_folder = 'intermediate/hha'

  run_hha(
      scene_dir=scene_dir,
      input_folder=input_folder,
      output_folder=output_folder,
      n_jobs=2,
  )
