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

from hha.getHHA import getHHA

from mmseg.core import get_classes, get_palette
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import torch
import cv2
import numpy as np


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

# test a single image and show the results
img = '/media/blumh/data/scannet/scene0000_00/color/0.jpg'  # or img = mmcv.imread(img), which will only load it once
img = cv2.imread(img)
img = cv2.resize(img, (640, 480))
print(img.shape)

depth = '/media/blumh/data/scannet/scene0000_00/depth/0.png'  # or img = mmcv.imread(img), which will only load it once
depth = cv2.imread(depth, cv2.COLOR_BGR2GRAY) / 1000
#depth[depth == 0] = depth.max()
depth_intrinsics = np.loadtxt(
    '/media/blumh/data/scannet/scene0000_00/intrinsic/intrinsic_depth.txt'
)[:3, :3]
hha = getHHA(depth_intrinsics, depth, depth)
plt.imshow(hha)
result = evaluator.sliding_eval_rgbX(img, hha,
                                     config.eval_crop_size,
                                     config.eval_stride_rate, 'cuda')
print(result.shape)
print(result)
