import sys, os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from joblib import Parallel, delayed

from segmentation_tools.label_mappings import LabelMatcher
import segmentation_eval


def test_label_vs_label():
    scene_dir = Path('/media/blumh/data/replica/office_0/Sequence_1/')
    keys = sorted(
        int(x.name.split('.')[0].split('_')[1])
        for x in (scene_dir / 'rgb').iterdir())[:1]
    label_template = 'semantic_class/semantic_class_{k}.png'
    label_space = 'replicaid'
    confmat = segmentation_eval._get_confmat(scene_dir, keys, label_space, label_space,
                           label_template, label_template)
    assert confmat.shape == (93, 93)
    assert np.all(confmat[np.eye(93) == 0] == 0)
    assert np.diag(confmat).sum() > 0

def test_label_in_bigger_space():
    scene_dir = Path('/media/blumh/data/replica/office_0/Sequence_1/')
    keys = sorted(
        int(x.name.split('.')[0].split('_')[1])
        for x in (scene_dir / 'rgb').iterdir())[:1]
    label_template = 'semantic_class/semantic_class_{k}.png'
    label_space = 'replicaid'
    matcher = LabelMatcher('nyu40id', label_space)
    confmat = np.zeros((len(matcher.right_ids), len(matcher.right_ids)),
                       dtype=np.int64)
    for k in tqdm(keys):
        label = cv2.imread(str(scene_dir / label_template.format(k=k)),
                           cv2.IMREAD_UNCHANGED)
        # map the prediction from replica to nyu40id
        nyu_label = matcher.mapping[label]
        if np.sum(nyu_label == -2) > 0:
            for source_id, target_ids in matcher.mapping_multiples.items():
                nyu_label[label == source_id] = target_ids[0]
        confmat += matcher.confusion_matrix(nyu_label, label)
    assert confmat.shape == (93, 93)
    assert np.all(confmat[np.eye(93) == 0] == 0)
    assert np.diag(confmat).sum() > 0


def test_label_in_smaller_space():
    scene_dir = Path('/media/blumh/data/replica/office_0/Sequence_1/')
    keys = sorted(
        int(x.name.split('.')[0].split('_')[1])
        for x in (scene_dir / 'rgb').iterdir())[:1]
    label_template = 'semantic_class/semantic_class_{k}.png'
    matcher = LabelMatcher('replicaid', 'nyu40id')
    confmat = np.zeros((len(matcher.right_ids), len(matcher.right_ids)),
                       dtype=np.int64)
    for k in tqdm(keys):
        label = cv2.imread(str(scene_dir / label_template.format(k=k)),
                          cv2.IMREAD_UNCHANGED)
        # map the prediction from replica to nyu40id
        nyu_label = matcher.mapping[label]
        if np.sum(nyu_label == -2) > 0:
            for source_id, target_ids in matcher.mapping_multiples.items():
                nyu_label[label == source_id] = target_ids[0]
        confmat += matcher.confusion_matrix(label, nyu_label)
    assert confmat.shape == (40, 40)
    assert np.all(confmat[np.eye(40) == 0] == 0)
    assert np.diag(confmat).sum() > 0
