import sys, os
from hha.getHHA import getHHA

import logging
from joblib import Parallel, delayed
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level="INFO")
log = logging.getLogger('Depth to HHA conversion')

parser = argparse.ArgumentParser()
parser.add_argument('scene')
flags = parser.parse_args()


scene_dir = Path(flags.scene)
assert scene_dir.exists() and scene_dir.is_dir()
keys = sorted(int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
depth_intrinsics = np.loadtxt(
    str(scene_dir / 'intrinsic/intrinsic_depth.txt')
)[:3, :3]


def depth_to_hha(k):
    orig_depth = cv2.imread(str(scene_dir / 'depth' / f'{k}.png'), cv2.COLOR_BGR2GRAY) / 1000
    omni_depth = cv2.imread(str(scene_dir / 'omnidata_depth' / f'{k}.png'), cv2.COLOR_BGR2GRAY) / 1000
    hha = getHHA(depth_intrinsics, omni_depth, orig_depth)
    cv2.imwrite(str(scene_dir / 'hha' / f'{k}.png'), hha)

(scene_dir / 'hha').mkdir(exist_ok=True)
Parallel(n_jobs=8)(delayed(depth_to_hha)(k) for k in tqdm(keys))
