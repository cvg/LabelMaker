import sys, os

import matplotlib.pyplot as plt

import logging
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import segmentation_cmx as cmx
import segmentation_internimage as internimage
import segmentation_ovseg as ovseg
import segmentation_eval
import depth2hha

logging.basicConfig(level="INFO")
log = logging.getLogger('Segmentation')


def process_scene(scene_dir, replica=False):
    scene_dir = Path(scene_dir)
    assert scene_dir.exists() and scene_dir.is_dir()
    if replica:
        keys = sorted(
            int(x.name.split('.')[0].split('_')[1])
            for x in (scene_dir / 'rgb').iterdir())
        img_template = 'rgb/rgb_{k}.png'
        label_template = 'semantic_class/semantic_class_{k}.png'
        # focal length is just guess-copied from scannet
        depth_intrinsics = np.array([[580, 0, 320, 0], [0, 580, 240, 0],
                                     [0, 0, 1, 0], [0, 0, 0, 1]])
        depth_template = 'depth/depth_{k}.png'
        # depth is already complete
        depth_completion_template = depth_template
    else:
        keys = sorted(
            int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
        img_template = 'color/{k}.png'
        label_template = 'label-filt/{k}.png'
        depth_intrinsics = np.loadtxt(
            str(scene_dir / 'intrinsic/intrinsic_depth.txt'))[:3, :3]
        depth_template = 'depth/{k}.png'
        depth_completion_template = 'omnidata_depth/{k}.png'

    depth2hha.run_depth_to_hha(
        scene_dir,
        keys,
        depth_intrinsics,
        depth_template=depth_template,
        depth_completion_template=depth_completion_template,
        n_jobs=8)
    cmx.cmx_inference(scene_dir, keys, img_template=img_template)
    internimage.internimage_inference(scene_dir,
                                      keys,
                                      img_template=img_template)
    ovseg.ovseg_inference(scene_dir,
                          keys,
                          classes='wordnet',
                          img_template=img_template)
    ovseg.ovseg_inference(scene_dir,
                          keys,
                          classes='wn_nodef',
                          img_template=img_template)
    ovseg.ovseg_inference(scene_dir,
                          keys,
                          classes='wn_nosyn',
                          img_template=img_template)
    ovseg.ovseg_inference(scene_dir,
                          keys,
                          classes='wn_nosyn_nodef',
                          img_template=img_template)
    segmentation_eval.evaluate_scene(
        scene_dir,
        'ade20k',
        'replicaid' if replica else 'id',
        pred_template='pred_internimage/{k}.png',
        label_template=label_template)
    segmentation_eval.evaluate_scene(
        scene_dir,
        'nyu40id',
        'replicaid',
        pred_template='pred_cmx/{k}.png',
        label_template=label_template)
    segmentation_eval.evaluate_scene(
        scene_dir,
        'wn199',
        'replicaid',
        pred_template='pred_ovseg_wordnet/{k}.png',
        label_template=label_template)
    segmentation_eval.evaluate_scene(
        scene_dir,
        'wn199',
        'replicaid',
        pred_template='pred_ovseg_wn_nodef/{k}.png',
        label_template=label_template)
    segmentation_eval.evaluate_scene(
        scene_dir,
        'wn199',
        'replicaid',
        pred_template='pred_ovseg_wn_nosyn/{k}.png',
        label_template=label_template)
    segmentation_eval.evaluate_scene(
        scene_dir,
        'wn199',
        'replicaid',
        pred_template='pred_ovseg_wn_nosyn_nodef/{k}.png',
        label_template=label_template)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--replica', default=False)
    flags = parser.parse_args()
    process_scene(scene_dir=flags.scene, replica=flags.replica)
