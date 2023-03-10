import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mmseg', 'InternImage', 'segmentation'))
import mmcv_custom
import mmseg_custom

from mmcv.runner import load_checkpoint
from mmseg.core import get_classes, get_palette
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

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

# test a single image and show the results
img = '/media/blumh/data/scannet/scene0000_00/color/0.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# visualize the results in a new window
model.show_result(img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
model.show_result(img, result, out_file='result.jpg', opacity=0.5)
