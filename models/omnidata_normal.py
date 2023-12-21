import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Union

import gin
import numpy as np
import PIL
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "3rdparty", "omnidata",
                     "omnidata_tools", "torch")))

from data.transforms import get_transform
from modules.midas.dpt_depth import DPTDepthModel
from modules.unet import UNet

logging.basicConfig(level="INFO")
log = logging.getLogger("Omnidata Normal")


def setup_seeds(seed):

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  cudnn.benchmark = False
  cudnn.deterministic = True


def load_omninormal(device: Union[str, torch.device] = 'cuda:0',):
  log.info('loading model')
  pretrained_weights_path = Path(os.path.abspath(os.path.dirname(
      __file__))) / '..' / 'checkpoints' / 'omnidata_dpt_normal_v2.ckpt'
  model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
  map_location = (lambda storage, loc: storage.cuda(device=device))
  checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

  if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
      state_dict[k[6:]] = v
  else:
    state_dict = checkpoint

  model.load_state_dict(state_dict)
  model.to(device)
  return model


@gin.configurable
def run(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    device: Union[str, torch.device] = 'cuda:0',
    size=(480, 640),
):
  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  assert scene_dir.exists() and scene_dir.is_dir()

  input_color_dir = scene_dir / 'color'
  assert input_color_dir.exists() and input_color_dir.is_dir()

  output_dir = scene_dir / output_folder

  log.info('[omninormal] loading model')
  model = load_omninormal(device=device)
  trans_totensor = transforms.Compose([
      transforms.Resize((384, 384), interpolation=PIL.Image.BILINEAR),
      transforms.CenterCrop(384),
      get_transform('rgb', image_size=None)
  ])

  log.info('[omninormal] running inference')

  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)

  keys = [p.stem for p in input_color_dir.glob('*.jpg')]

  for k in tqdm(keys):
    img = Image.open(str(input_color_dir / f'{k}.jpg'))

    with torch.no_grad():
      img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

    if img_tensor.shape[1] == 1:
      img_tensor = img_tensor.repeat_interleave(3, 1)

    output = model(img_tensor).clamp(min=0, max=1)  # (1, 3, 384, 384)
    output = F.interpolate(
        output,
        size,
        mode='nearest',
    ).squeeze(0)  # (3, H, W)

    omninormal = output.detach().cpu().squeeze().numpy()  # (3, H, W)
    omninormal = omninormal.transpose(1, 2, 0)  # (H, W, 3)

    np.save(str(output_dir / f'{k}.npy'), omninormal)


def arg_parser():
  parser = argparse.ArgumentParser(description='Omnidata Normal Estimation')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help='Path to workspace directory. There should be "color" folder inside.',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/normal_omnidata_1',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version',
  )
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == "__main__":
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  run(scene_dir=args.workspace, output_folder=args.output)
