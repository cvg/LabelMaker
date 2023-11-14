import os
import sys
from os.path import abspath, dirname, exists, join
from pathlib import Path

import cv2
import numpy as np


def test_consensus():
  from labelmaker.consensus import run as run_consensus

  scene_dir = Path(abspath(join(dirname(__file__), '../test_scan')))
  output_folder = 'intermediate/consensus'
  name = '000000.png'

  run_consensus(
      scene_dir=scene_dir,
      output_folder=output_folder,
  )

  img_arr = cv2.imread(str(scene_dir / output_folder / name),
                       cv2.IMREAD_UNCHANGED)
  assert np.unique(img_arr).shape[0] > 1
