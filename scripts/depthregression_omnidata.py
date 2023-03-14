import sys, os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'omnidata',
                     'omnidata_tools', 'torch')))
from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform

import matplotlib.pyplot as plt
import logging
import mmcv
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import PIL
from PIL import Image

logging.basicConfig(level="INFO")
log = logging.getLogger('Omnidata Depth')

parser = argparse.ArgumentParser()
parser.add_argument('scene')
flags = parser.parse_args()

map_location = (lambda storage, loc: storage.cuda()
                ) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

log.info('loading model')
omnidata_path = Path(os.path.abspath(os.path.dirname(
    __file__))) / '..' / '3rdparty' / 'omnidata' / 'omnidata_tools' / 'torch'
pretrained_weights_path = omnidata_path / 'pretrained_models' / 'omnidata_dpt_depth_v2.ckpt'
model = DPTDepthModel(backbone='vitb_rn50_384')
checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
if 'state_dict' in checkpoint:
    state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        state_dict[k[6:]] = v
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.to(device)
trans_totensor = transforms.Compose([
    transforms.Resize((384, 384), interpolation=PIL.Image.BILINEAR),
    #transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

trans_rgb = transforms.Compose([
    transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
    transforms.CenterCrop(512)
])


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value *
                               len(sorted_img)):int((1 - trunc_value) *
                                                    len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def run_inference(img):
    with torch.no_grad():
        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3, 1)
        output = model(img_tensor).clamp(min=0, max=1)
        output = F.interpolate(output.unsqueeze(0), (480, 640),
                               mode='bicubic').squeeze(0)
        output = output.clamp(0, 1)
        #output = 1 - output
        #output = standardize_depth_map(output)
        return output.detach().cpu().squeeze()


scene_dir = Path(flags.scene)
assert scene_dir.exists() and scene_dir.is_dir()
(scene_dir / 'omnidata_depth').mkdir(exist_ok=True)
keys = sorted(
    int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())

log.info('running inference')
results = []
for k in tqdm(keys):
    img = Image.open(str(scene_dir / 'color' / f'{k}.jpg'))
    depth = run_inference(img).numpy()
    # find a linear scaling a * depth + b to fit to original depth
    orig_depth = cv2.imread(str(scene_dir / 'depth' / f'{k}.png'),
                            cv2.IMREAD_UNCHANGED)
    targets = orig_depth[orig_depth != 0]
    source = depth[orig_depth != 0]
    sources = np.stack([source, np.ones_like(source)])
    a, b = np.linalg.lstsq(np.stack([source, np.ones_like(source)], axis=-1),
                           targets,
                           rcond=None)[0]
    depth = (a * depth + b).astype(orig_depth.dtype)
    cv2.imwrite(str(scene_dir / 'omnidata_depth' / f'{k}.png'), depth)
