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


def run_depth_to_hha(scene_dir,
                     keys,
                     depth_intrinsics,
                     depth_template='depth/{k}.png',
                     depth_completion_template='omnidata_depth/{k}.png',
                     n_jobs=8):
    scene_dir = Path(scene_dir)

    def depth_to_hha(k):
        orig_depth = cv2.imread(str(scene_dir / depth_template.format(k=k)),
                                cv2.COLOR_BGR2GRAY) / 1000
        omni_depth = cv2.imread(str(scene_dir / depth_completion_template.format(k=k)),
                                cv2.COLOR_BGR2GRAY) / 1000
        hha = getHHA(depth_intrinsics, omni_depth, orig_depth)
        cv2.imwrite(str(scene_dir / 'hha' / f'{k}.png'), hha)

    (scene_dir / 'hha').mkdir(exist_ok=True)
    Parallel(n_jobs=n_jobs)(delayed(depth_to_hha)(k) for k in tqdm(keys))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--replica', default=False)
    parser.add_argument('--j', default=8)
    flags = parser.parse_args()
    scene_dir = Path(flags.scene)
    assert scene_dir.exists() and scene_dir.is_dir()
    if flags.replica:
        keys = sorted(
            int(x.name.split('.')[0].split('_')[1])
            for x in (scene_dir / 'rgb').iterdir())
        img_template = 'rgb/rgb_{k}.png'
        # focal length is just guess-copied from scannet
        depth_intrinsics = np.array([[580, 0, 320, 0], [0, 580, 240, 0],
                                     [0, 0, 1, 0], [0, 0, 0, 1]])
        depth_template = 'depth/depth_{k}.png'
        # depth is already complete
        depth_completion_template = depth_template
    else:
        keys = sorted(
            int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
        img_template = 'color/{k}.png'
        depth_intrinsics = np.loadtxt(
            str(scene_dir / 'intrinsic/intrinsic_depth.txt'))[:3, :3]
        depth_template = 'depth/{k}.png'
        depth_completion_template = 'omnidata_depth/{k}.png'
    run_depth_to_hha(scene_dir,
                     keys,
                     depth_intrinsics,
                     depth_template=depth_template,
                     depth_completion_template=depth_completion_template,
                     n_jobs=flags.j)
