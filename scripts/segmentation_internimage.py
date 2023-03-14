import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mmseg', 'InternImage', 'segmentation'))
import mmcv_custom
import mmseg_custom

from mmcv.runner import load_checkpoint
from mmseg.core import get_classes, get_palette
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2

logging.basicConfig(level="INFO")
log = logging.getLogger('InternImage Segmentation')

parser = argparse.ArgumentParser()
parser.add_argument('scene')
flags = parser.parse_args()

def load_internimage():
   config_file = './mmseg/upernet_internimage_xl_640_160k_ade20k.py'
   checkpoint_file = './mmseg/upernet_internimage_xl_640_160k_ade20k.pth'
   # build the model from a config file and a checkpoint file
   model = init_segmentor(config_file, checkpoint=None, device='cuda:0')
   checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
   if 'CLASSES' in checkpoint.get('meta', {}):
      model.CLASSES = checkpoint['meta']['CLASSES']
   else:
      print('"CLASSES" not found in meta, use dataset.CLASSES instead')
      model.CLASSES = get_classes('ade20k')
   if 'PALETTE' in checkpoint.get('meta', {}):
      model.PALETTE = checkpoint['meta']['PALETTE']
   else:
      print('"PALETTE" not found in meta, use dataset.PALETTE instead')
      model.PALETTE = get_palette('ade20k')
   return model

log.info('loading model')
model = load_internimage()

scene_dir = Path(flags.scene)
assert scene_dir.exists() and scene_dir.is_dir()
(scene_dir / 'pred_internimage').mkdir(exist_ok=True)
keys = sorted(
    int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())

log.info('running inference')
for k in tqdm(keys):
    img = str(scene_dir / 'color' / f'{k}.jpg')
    result = inference_segmentor(model, img)
    cv2.imwrite(str(scene_dir / 'pred_internimage' / f'{k}.png'), result[0])
