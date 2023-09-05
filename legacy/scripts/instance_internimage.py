import sys, os

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'InternImage',
                 'detection'))
import mmcv_custom
import mmdet_custom

from mmcv.runner import load_checkpoint
import mmcv
from mmdet.apis import inference_detector, init_detector
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import shutil
import torch
import numpy as np

logging.basicConfig(level="INFO")
log = logging.getLogger('InternImage Segmentation')


def load_internimage():
    # config_file = os.path.join(os.path.dirname(__file__), '..', '3rdparty',
    #                            'upernet_internimage_xl_640_160k_ade20k.py')
    # checkpoint_file = os.path.join(
    #     os.path.dirname(__file__), '..', '3rdparty',
    #     'upernet_internimage_xl_640_160k_ade20k.pth')
    config_file = os.path.join(os.path.dirname(__file__), '..', '3rdparty',
                               'InternImage', 'detection', 'configs', 'coco',
                               'cascade_internimage_xl_fpn_3x_coco.py')
    checkpoint_file = os.path.join(os.path.dirname(__file__), '..', '3rdparty',
                                   'cascade_internimage_xl_fpn_3x_coco.pth')
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint=checkpoint_file, device='cuda:0')
    #checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    return model


def internimage_inference(scene_dir,
                          keys,
                          img_template='color/{k}.png',
                          flip=False):
    log.info('[internimage] loading model')
    model = load_internimage()
    log.info(f'[internimage] running inference for {str(scene_dir)}')
    if flip:
        result_directory = scene_dir / 'inst_internimage_flip'
    else:
        result_directory = scene_dir / 'inst_internimage'
    shutil.rmtree(result_directory, ignore_errors=True)
    result_directory.mkdir(exist_ok=False)
    for k in tqdm(keys):
        img = str(scene_dir / img_template.format(k=k))
        img = mmcv.imread(img)
        if flip:
            img = img[:, ::-1]
        bbox_result, segm_result = inference_detector(model, img)
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
        output = np.zeros(segms.shape[1:], dtype=np.uint8)
        for i in range(segms.shape[0]):
            output[segms[i]] = i + 1
        if flip:
            output= output[:, ::-1]
        cv2.imwrite(str(result_directory / f'{k}.png'), output)


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
