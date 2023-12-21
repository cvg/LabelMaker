import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Union

import cv2
import gin
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import PIL
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from joblib import Parallel, delayed
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'omnidata',
                     'omnidata_tools', 'torch')))

from data.transforms import get_transform
from modules.midas.dpt_depth import DPTDepthModel
from modules.unet import UNet

logging.basicConfig(level="INFO")
log = logging.getLogger('Omnidata Depth')


def setup_seeds(seed):

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  cudnn.benchmark = False
  cudnn.deterministic = True


def load_omnidepth(device: Union[str, torch.device] = 'cuda:0',):
  log.info('loading model')
  pretrained_weights_path = Path(os.path.abspath(os.path.dirname(
      __file__))) / '..' / 'checkpoints' / 'omnidata_dpt_depth_v2.ckpt'
  model = DPTDepthModel(backbone='vitb_rn50_384')
  checkpoint = torch.load(pretrained_weights_path, map_location=device)
  if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
      state_dict[k[6:]] = v
  else:
    state_dict = checkpoint
  model.load_state_dict(state_dict)
  model.to(device)
  return model


def omnidepth_completion(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    patch_size=32,
):
  # convert str to Path object
  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  assert scene_dir.exists() and scene_dir.is_dir()

  input_depth_dir = scene_dir / 'depth'
  assert input_depth_dir.exists() and input_depth_dir.is_dir()

  output_dir = scene_dir / output_folder
  assert (output_dir).exists()

  log.info('[omnidepth] running completion')

  def depth_completion(k):
    orig_depth = cv2.imread(str(input_depth_dir / f'{k}.png'),
                            cv2.IMREAD_UNCHANGED)
    omnidepth = cv2.imread(str(output_dir / f'{k}.png'), cv2.IMREAD_UNCHANGED)

    # now complete the original depth with omnidepth predictions, fitted to scale
    # within a patch around each missing pixel
    fused_depth = orig_depth.copy()
    coords_u, coords_v = np.where(fused_depth == 0)
    for i in range(len(coords_u)):
      u = coords_u[i]
      v = coords_v[i]
      window_u = max(0, u - patch_size), min(fused_depth.shape[0],
                                             u + patch_size)
      window_v = max(0, v - patch_size), min(fused_depth.shape[1],
                                             v + patch_size)
      target = orig_depth[window_u[0]:window_u[1], window_v[0]:window_v[1]]
      source = omnidepth[window_u[0]:window_u[1], window_v[0]:window_v[1]]
      source = source[target != 0]
      target = target[target != 0]
      a, b = np.linalg.lstsq(np.stack([source, np.ones_like(source)], axis=-1),
                             target,
                             rcond=None)[0]
      # for some areas this will completely break the geometry, we need to revert to omnidepth
      if a < 0.5 or a > 2:
        fused_depth[u, v] = omnidepth[u, v]
      else:
        fused_depth[u, v] = a * omnidepth[u, v] + b
    fused_depth[fused_depth == 0] = omnidepth[fused_depth == 0]
    cv2.imwrite(str(output_dir / f'{k}.png'), fused_depth)

  keys = [p.stem for p in input_depth_dir.glob('*.png')]
  Parallel(n_jobs=8)(delayed(depth_completion)(k) for k in tqdm(keys))


@gin.configurable
def run(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    device: Union[str, torch.device] = 'cuda:0',
    depth_size=(480, 640),
    completion=True,
):
  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  assert scene_dir.exists() and scene_dir.is_dir()

  input_color_dir = scene_dir / 'color'
  assert input_color_dir.exists() and input_color_dir.is_dir()

  input_depth_dir = scene_dir / 'depth'
  assert input_depth_dir.exists() and input_depth_dir.is_dir()

  output_dir = scene_dir / output_folder

  log.info('[omnidepth] loading model')
  model = load_omnidepth(device=device)
  trans_totensor = transforms.Compose([
      transforms.Resize((384, 384), interpolation=PIL.Image.BILINEAR),
      transforms.ToTensor(),
      transforms.Normalize(mean=0.5, std=0.5)
  ])

  log.info('[omnidepth] running inference')

  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)

  keys = [p.stem for p in input_color_dir.glob('*.jpg')]

  for k in tqdm(keys):

    img = Image.open(str(input_color_dir / f'{k}.jpg'))
    with torch.no_grad():
      img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)
      if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat_interleave(3, 1)
      output = model(img_tensor).clamp(min=0, max=1)
      output = F.interpolate(output.unsqueeze(0), depth_size,
                             mode='bicubic').squeeze(0)
      output = output.clamp(0, 1)
      omnidepth = output.detach().cpu().squeeze().numpy()

    # find a linear scaling a * depth + b to fit to original depth
    orig_depth = cv2.imread(str(input_depth_dir / f'{k}.png'),
                            cv2.IMREAD_UNCHANGED)
    targets = orig_depth[orig_depth != 0]
    source = omnidepth[orig_depth != 0]
    a, b = np.linalg.lstsq(np.stack([source, np.ones_like(source)], axis=-1),
                           targets,
                           rcond=None)[0]
    omnidepth = (a * omnidepth + b).astype(orig_depth.dtype)
    cv2.imwrite(str(output_dir / f'{k}.png'), omnidepth)
  if completion:
    omnidepth_completion(scene_dir=scene_dir, output_folder=output_folder)


def arg_parser():
  parser = argparse.ArgumentParser(description='Omnidata Depth Estimation')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help=
      'Path to workspace directory. There should be "color" and "depth" folder inside.',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/depth_omnidata_1',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version',
  )
  parser.add_argument('--seed', type=int, default=42, help='random seed')
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == "__main__":
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  setup_seeds(seed=args.seed)
  run(scene_dir=args.workspace, output_folder=args.output)
