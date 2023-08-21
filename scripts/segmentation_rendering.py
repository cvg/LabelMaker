import sys, os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import shutil
import re
from glob import glob
import skimage.measure
from joblib import Parallel, delayed
import subprocess

from segmentation_tools.label_data import get_nyu40, get_wordnet, get_replica, get_scannet_all

logging.basicConfig(level="INFO")
log = logging.getLogger('Creating Segmentation Illustrations')



def render_segmentation(scene_dir,
                        keys,
                        input_template,
                        output_template,
                        colorspace,
                        image_template='color/{k}.jpg',
                        alpha=0.8,
                        fps=30.0,
                            n_jobs=8):

    log.info(f'Rendering segmentation for {scene_dir} with input {input_template}.')

    scene_dir = Path(scene_dir)
    assert scene_dir.exists() and scene_dir.is_dir()


    if colorspace == 'nyu40':
        classes = get_nyu40()
        colors = np.stack([x['color'] for x in sorted(classes, key=lambda x: x['id'])], axis=0)
    elif colorspace == 'wordnet':
        colors = np.stack([
            x['color'] for x in sorted(get_wordnet(label_key='wn199'),
                                       key=lambda x: x['id'])
        ],
                          axis=0)
    elif colorspace == 'replica':
        classes = get_replica()
        colors = np.stack([x['color'] for x in sorted(classes, key=lambda x: x['id'])] , axis=0)
    elif colorspace == 'scannet':
        scannet_id_to_color = {x['id'] : x['color'] for x in get_scannet_all()}
        colors = np.stack([scannet_id_to_color[i] if i in scannet_id_to_color else [0, 0, 0] for i in range(2000)])
    else:
        raise NotImplementedError

    output_dir = scene_dir / output_template.split('/')[0]
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=False)

    def render(k):
        pred = cv2.imread(str(scene_dir / input_template.format(k=k)), cv2.IMREAD_UNCHANGED)
        if 'sdfstudio' in input_template:
            img = cv2.imread(str(scene_dir / image_template.format(k=k*2)))[..., ::-1]
        else:
            img = cv2.imread(str(scene_dir / image_template.format(k=k)))[..., ::-1]
        vis = colors[pred]
        vis = (1 - alpha) * img + alpha * vis
        cv2.imwrite(str(scene_dir / output_template.format(k=k)), vis[..., ::-1])

    Parallel(n_jobs=n_jobs)(delayed(render)(k) for k in tqdm(keys))

    input_rate = fps
    if 'sdfstudio' in input_template:
        # has half the frames
        input_rate = fps // 2

    subprocess.call([
      'ffmpeg', '-r', f'{input_rate:.0f}', '-i', str(scene_dir / output_template.replace('{k:05d}', '%05d')), '-c:v',
      'libx264', '-crf', '27', '-r', f'{fps:.0f}',# '-vf', f'fps={fps}',
      str(scene_dir / f'{output_template.split("/")[0]}' / f'{output_template.split("/")[0]}.mp4')
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('colorspace', type=str)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--name', type=str, default='')
    flags = parser.parse_args()

    scene_dir = Path(flags.input_dir).parent
    keys = sorted(
        int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())

    files = glob(str(scene_dir / flags.input_dir / '*.png'),
                    recursive=True)
    keys = sorted(
        int(re.search('(\d+).png', x).group(1))
        for x in files)
    input_dir_name = flags.input_dir.split('/')[-1]
    if 'sdfstudio' in flags.input_dir:
        input_template = input_dir_name + '/{k:05d}.png'
    else:
        input_template = input_dir_name+ '/{k}.png'
    if len(flags.name) > 0:
        output_template = 'rendered_' + flags.name + '/{k:05d}.jpg'
    else:
        output_template = 'rendered_' + input_dir_name+ '/{k:05d}.jpg'

    render_segmentation(scene_dir, keys, input_template, output_template, flags.colorspace)
