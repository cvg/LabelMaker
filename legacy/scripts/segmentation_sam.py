import sys, os
import matplotlib.pyplot as plt
import logging
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

logging.basicConfig(level="INFO")
log = logging.getLogger('Segment Anything')


def sam_inference(scene_dir,
                  keys,
                  img_template='color/{k}.png',
                  flip=False):
    sam = sam_model_registry['vit_h'](checkpoint='3rdparty/sam_vit_h_4b8939.pth')
    sam.to('cuda:0')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        min_mask_region_area=100,
    )
    if flip:
        results_dir = scene_dir / 'pred_sam_flip'
    else:
        results_dir = scene_dir / 'pred_sam'
    shutil.rmtree(results_dir, ignore_errors=True)
    results_dir.mkdir(exist_ok=False)
    log.info('[SAM] running inference')
    for k in tqdm(keys):
        img = cv2.imread(str(scene_dir / img_template.format(k=k)))[..., ::-1]
        if flip:
            img = img[:, ::-1]
        masks = mask_generator.generate(img)
        pred = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for i, mask in enumerate(masks):
            pred[mask['segmentation']] = i
        if flip:
            pred = pred[:, ::-1]
        cv2.imwrite(str(results_dir / f'{k}.png'), pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--replica', default=False, action='store_true')
    parser.add_argument('--flip', default=False, action='store_true')
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
        img_template = 'color/{k}.jpg'
    sam_inference(scene_dir, keys, img_template=img_template, flip=flags.flip)
