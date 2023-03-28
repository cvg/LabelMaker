import sys, os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

from segmentation_tools.label_mappings import MatcherScannetADE150, MatcherScannetNYU40, MatcherADE150NYU40, get_ade150_to_scannet

logging.basicConfig(level="INFO")
log = logging.getLogger('Segmentation Consensus')

parser = argparse.ArgumentParser()
parser.add_argument('scene')
flags = parser.parse_args()

scene_dir = Path(flags.scene)
assert scene_dir.exists() and scene_dir.is_dir()
(scene_dir / 'pred_consensus').mkdir(exist_ok=True)
keys = sorted(
    int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())

ade150_to_scannet = get_ade150_to_scannet()
matcher_scannet_ade150 = MatcherScannetADE150()
matcher_scannet_nyu40 = MatcherScannetNYU40()
matcher_ade150_nyu40 = MatcherADE150NYU40()

for k in tqdm(keys):
    intern_ade150 = cv2.imread(str(scene_dir / 'pred_internimage' / f'{k}.png'), cv2.IMREAD_UNCHANGED)
    cmx_nyu40 = cv2.imread(str(scene_dir / 'pred_cmx' / f'{k}.png'), cv2.IMREAD_UNCHANGED)
    cmx_nyu40 = cv2.resize(cmx_nyu40, (1296, 968), cv2.INTER_NEAREST)
    label = cv2.imread(str(scene_dir / 'label-filt' / f'{k}.png'), cv2.IMREAD_UNCHANGED)

    # InternImage ADE150 -> Scannet
    label_intern_match = matcher_scannet_ade150.match(label, intern_ade150)
    # CMX NYU40 -> Scannet
    label_cmx_match = matcher_scannet_nyu40.match(label, cmx_nyu40 + 1)
    # InternImage ADE150 -> CMX NYU40
    intern_cmx_match = matcher_ade150_nyu40.match(intern_ade150, cmx_nyu40 + 1)

    new_label = -1 * np.ones_like(label)
    new_label[label_intern_match == 1] = label[label_intern_match == 1]
    new_label[label_cmx_match == 1] = label[label_cmx_match == 1]

    matching_predictors = np.logical_and(new_label == -1, intern_cmx_match == 1)
    new_label[matching_predictors] = ade150_to_scannet[intern_ade150[matching_predictors]]

    cv2.imwrite(str(scene_dir / 'pred_consensus' / f'{k}.png'), new_label)
