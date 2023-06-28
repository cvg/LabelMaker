import sys, os

sys.path.insert(
    0,
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
import gin
import shutil
from joblib import Parallel, delayed

logging.basicConfig(level="INFO")
log = logging.getLogger('Omnidata Normals')


def load_omninormal():
    map_location = (lambda storage, loc: storage.cuda()
                    ) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    log.info('loading model')
    omnidata_path = Path(
        os.path.abspath(os.path.dirname(__file__))
    ) / '..' / '3rdparty' / 'omnidata' / 'omnidata_tools' / 'torch'
    pretrained_weights_path = omnidata_path / 'pretrained_models' / 'omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
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


@gin.configurable()
def run_inference(img, size=(480, 640)):
    with torch.no_grad():
        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3, 1)
        output = model(img_tensor).clamp(min=0, max=1)
        output = F.interpolate(output.unsqueeze(0), depth_size,
                               mode='bicubic').squeeze(0)
        output = output.clamp(0, 1)
        #output = 1 - output
        #output = standardize_depth_map(output)
        return output.detach().cpu().squeeze()


def omninormal_inference(scene_dir,
                         keys,
                         img_template='color/{k}.jpg',
                         size=(480, 640)):
    log.info('[omninormal] loading model')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_omninormal()
    trans_totensor = transforms.Compose([
        transforms.Resize((384, 384), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    trans_rgb = transforms.Compose([
        transforms.Resize(512, interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop(512)
    ])

    log.info('[omninormal] running inference')
    shutil.rmtree(scene_dir / 'omnidata_normal', ignore_errors=True)
    (scene_dir / 'omnidata_normal').mkdir(exist_ok=False)
    for k in tqdm(keys):
        img = Image.open(str(scene_dir / img_template.format(k=k)))
        with torch.no_grad():
            img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)
            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3, 1)
            output = model(img_tensor).clamp(min=0, max=1)

            output = F.interpolate(output, size,
                                   mode='nearest').squeeze(0)
            omninormal = output.detach().cpu().squeeze().numpy()
            omninormal = omninormal.transpose(1, 2, 0)
        np.save(str(scene_dir / 'omnidata_normal' / f'{k}.npy'), omninormal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--config')
    parser.add_argument('--replica', default=False)
    flags = parser.parse_args()

    if flags.config is not None:
        gin.parse_config_file(flags.config)

    scene_dir = Path(flags.scene)
    assert scene_dir.exists() and scene_dir.is_dir()
    if flags.replica:
        keys = sorted(
            int(x.name.split('.')[0].split('_')[1])
            for x in (scene_dir / 'rgb').iterdir())
        img_template = 'rgb/rgb_{k}.png'
    else:
        keys = sorted(
            int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
        img_template = 'color/{k}.jpg'

    omninormal_inference(scene_dir,
                         keys,
                         img_template=img_template)
