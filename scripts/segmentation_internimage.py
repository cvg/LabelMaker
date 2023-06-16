import sys, os

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'InternImage',
                 'segmentation'))
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
import shutil

logging.basicConfig(level="INFO")
log = logging.getLogger('InternImage Segmentation')


def load_internimage():
    # config_file = os.path.join(os.path.dirname(__file__), '..', '3rdparty',
    #                            'upernet_internimage_xl_640_160k_ade20k.py')
    # checkpoint_file = os.path.join(
    #     os.path.dirname(__file__), '..', '3rdparty',
    #     'upernet_internimage_xl_640_160k_ade20k.pth')
    config_file = os.path.join(os.path.dirname(__file__), '..', '3rdparty',
                               'InternImage', 'segmentation', 'configs',
                               'ade20k',
                               'upernet_internimage_h_896_160k_ade20k.py')
    checkpoint_file = os.path.join(
        os.path.dirname(__file__), '..', '3rdparty',
        'upernet_internimage_h_896_160k_ade20k.pth')
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


def internimage_inference(scene_dir,
                          keys,
                          img_template='color/{k}.png',
                          flip=False):
    log.info('[internimage] loading model')
    model = load_internimage()
    log.info('[internimage] running inference')
    if flip:
        result_directory = scene_dir / 'pred_internimage_flip'
    else:
        result_directory = scene_dir / 'pred_internimage'
    shutil.rmtree(result_directory, ignore_errors=True)
    result_directory.mkdir(exist_ok=False)
    for k in tqdm(keys):
        img = str(scene_dir / img_template.format(k=k))
        img = mmcv.imread(img)
        if flip:
            img = img[:, ::-1]
        result = inference_segmentor(model, img)
        if flip:
            result[0] = result[0][:, ::-1]
        cv2.imwrite(str(result_directory / f'{k}.png'), result[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--replica', default=False)
    parser.add_argument('--flip', default=False)
    flags = parser.parse_args()
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
        img_template = 'color/{k}.png'
    internimage_inference(scene_dir,
                          keys,
                          img_template=img_template,
                          flip=flags.flip)
