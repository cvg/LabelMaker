import sys, os
from hha.getHHA import getHHA
import gin

import logging
from joblib import Parallel, delayed
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level="INFO")
log = logging.getLogger('Depth to HHA conversion')



def run(args,
        n_jobs=8):
    
  scene_dir = Path(args.workspace)
  log.info(f'running depth to hha conversion for scene {scene_dir}')
  
  depth_dir = scene_dir / 'depth'
  omnidata_depth_dir = scene_dir / args.input
  
  assert scene_dir.exists() and scene_dir.is_dir()
  assert depth_dir.exists() and depth_dir.is_dir()
  assert omnidata_depth_dir.exists() and omnidata_depth_dir.is_dir()
  assert len(list(depth_dir.iterdir())) == len(list(omnidata_depth_dir.iterdir()))
  
  def depth_to_hha(k):
    intrinsics = np.loadtxt(str(scene_dir / 'intrinsic' / f'{k}.txt'))[:3, :3]
    orig_depth = cv2.imread(str(depth_dir / f'{k}.png'),
                            cv2.COLOR_BGR2GRAY) / 1000
    omni_depth = cv2.imread(str(omnidata_depth_dir / f'{k}.png'),
                            cv2.COLOR_BGR2GRAY) / 1000
    hha = getHHA(intrinsics, omni_depth, orig_depth)
    cv2.imwrite(str(scene_dir / args.output / f'{k}.png'), hha)

  (scene_dir / args.output).mkdir(exist_ok=True)
  keys = [p.stem for p in (scene_dir / 'depth').glob('*.png')]
  Parallel(n_jobs=n_jobs)(delayed(depth_to_hha)(k) for k in tqdm(keys))


def main(args):
    run(args)

def arg_parser():
    parser = argparse.ArgumentParser(description='CMX Segmentation')
    parser.add_argument('--workspace',
                        type=str,
                        required=True,
                        help='Path to workspace directory')
    parser.add_argument(
        '--input',
        type=str,
        default='intermediate/depth_omnidata_1',
        help='Name of input directory in the workspace directory')
    parser.add_argument(
        '--output',
        type=str,
        default='intermediate/hha',
        help=
        'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version'
    )
    parser.add_argument('--config', help='Name of config file')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('scene')
#     parser.add_argument('--replica', default=False)
#     parser.add_argument('--j', default=8)
#     flags = parser.parse_args()
#     scene_dir = Path(flags.scene)
#     assert scene_dir.exists() and scene_dir.is_dir()
#     if flags.replica:
#         keys = sorted(
#             int(x.name.split('.')[0].split('_')[1])
#             for x in (scene_dir / 'rgb').iterdir())
#         img_template = 'rgb/rgb_{k}.png'
#         # focal length is just guess-copied from scannet
#         depth_intrinsics = np.array([[320, 0, 320, 0], [0, 320, 240, 0],
#                                      [0, 0, 1, 0], [0, 0, 0, 1]])
#         depth_template = 'depth/depth_{k}.png'
#         # depth is already complete
#         depth_completion_template = depth_template
#     else:
#         keys = sorted(
#             int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
#         img_template = 'color/{k}.png'
#         depth_intrinsics = np.loadtxt(
#             str(scene_dir / 'intrinsic/intrinsic_depth.txt'))[:3, :3]
#         depth_template = 'depth/{k}.png'
#         depth_completion_template = 'omnidata_depth/{k}.png'
#     run_depth_to_hha(scene_dir,
#                      keys,
#                      depth_intrinsics,
#                      depth_template=depth_template,
#                      depth_completion_template=depth_completion_template,
#                      n_jobs=flags.j)
