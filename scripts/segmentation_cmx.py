import sys, os

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'mmseg',
                 'RGBX_Semantic_Segmentation'))
from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre

import matplotlib.pyplot as plt

import logging
from mmseg.core import get_classes, get_palette
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level="INFO")
log = logging.getLogger('CMX Segmentation')

parser = argparse.ArgumentParser()
parser.add_argument('scene')
flags = parser.parse_args()


log.info('loading model')
checkpoint_file = './mmseg/NYUDV2_CMX+Segformer-B2.pth'
network = segmodel(cfg=config, criterion=None, norm_layer=torch.nn.BatchNorm2d)
data_setting = {
    'rgb_root': config.rgb_root_folder,
    'rgb_format': config.rgb_format,
    'gt_root': config.gt_root_folder,
    'gt_format': config.gt_format,
    'transform_gt': config.gt_transform,
    'x_root': config.x_root_folder,
    'x_format': config.x_format,
    'x_single_channel': config.x_is_single_channel,
    'class_names': config.class_names,
    'train_source': config.train_source,
    'eval_source': './mmseg/empty.txt',
    'class_names': config.class_names
}
val_pre = ValPre()
dataset = RGBXDataset(data_setting, 'val', val_pre)
evaluator = Evaluator(dataset, 40, config.norm_mean, config.norm_std, network,
                      config.eval_scale_array, config.eval_flip,
                      parse_devices('0'))
evaluator.compute_metric = lambda x: str()
evaluator.run('./mmseg', checkpoint_file, '/dev/null', '/tmp/fakelog')


scene_dir = Path(flags.scene)
assert scene_dir.exists() and scene_dir.is_dir()
(scene_dir / 'pred_cmx').mkdir(exist_ok=True)
keys = sorted(
    int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())

log.info('running inference')
for k in tqdm(keys):
    img = cv2.imread(str(scene_dir / 'color' / f'{k}.jpg'))[..., ::-1]
    img = cv2.resize(img, (640, 480))
    hha = cv2.imread(str(scene_dir / 'hha' / f'{k}.png'))
    result = evaluator.sliding_eval_rgbX(img, hha,
                                     config.eval_crop_size,
                                     config.eval_stride_rate, 'cuda')
    cv2.imwrite(str(scene_dir / 'pred_cmx' / f'{k}.png'), result)
