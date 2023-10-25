import os
import sys
from os.path import abspath, dirname, exists, join
from pathlib import Path

import cv2
import numpy as np

sys.path.append(abspath(join(dirname(__file__), '../..')))


def test_grounded_sam():
  from models.grounded_sam import run as run_grounded_sam

  scene_dir = Path(abspath(join(dirname(__file__), '../test_scan')))
  output_folder = 'intermediate/wordnet_groundedsam_1'
  name = '000000.png'

  run_grounded_sam(
      scene_dir=scene_dir,
      output_folder=output_folder,
      device='cuda:0',
  )

  img_arr = cv2.imread(str(scene_dir / output_folder / name),
                       cv2.IMREAD_UNCHANGED)
  assert np.unique(img_arr).shape[0] > 1
