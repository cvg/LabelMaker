import sys, os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix

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
    assert confmat.shape == (94, 94)
    assert np.all(confmat[np.eye(94) == 0] == 0)
    assert np.diag(confmat).sum() > 0
    matcher = LabelMatcher(label_space, label_space)
    matched = matcher.match_confmat(confmat)
    assert np.all(matched == confmat)

def test_label_in_bigger_space():
    scene_dir = Path('/media/blumh/data/replica/office_0/Sequence_1/')
    keys = sorted(
        int(x.name.split('.')[0].split('_')[1])
        for x in (scene_dir / 'rgb').iterdir())[:1]
    label_template = 'semantic_class/semantic_class_{k}.png'
    label_space = 'replicaid'
    matcher = LabelMatcher('nyu40id', label_space)
    confmat = np.zeros((len(matcher.left_ids) + 1, len(matcher.right_ids) + 1),
                       dtype=np.int64)
    left_id_to_confmat_idx = np.zeros(max(matcher.left_ids) + 1,
                                      dtype=np.int64)
    for i, left_id in enumerate(matcher.left_ids):
        left_id_to_confmat_idx[left_id] = i + 1
    right_id_to_confmat_idx = np.zeros(max(matcher.right_ids) + 1,
                                       dtype=np.int64)
    for i, right_id in enumerate(matcher.right_ids):
        right_id_to_confmat_idx[right_id] = i + 1
    for k in tqdm(keys):
        label = cv2.imread(str(scene_dir / label_template.format(k=k)),
                           cv2.IMREAD_UNCHANGED)
        # map the prediction from replica to nyu40id
        nyu_label = matcher.mapping[label]
        if np.sum(nyu_label == -2) > 0:
            for source_id, target_ids in matcher.mapping_multiples.items():
                nyu_label[label == source_id] = target_ids[0]
        sample_weights = np.ones_like(label.flatten(), dtype=np.int64)
        left = left_id_to_confmat_idx[nyu_label.flatten()]
        right = right_id_to_confmat_idx[label.flatten()]
        confmat += coo_matrix((sample_weights, (left, right)),
                              shape=confmat.shape,
                              dtype=np.int64).toarray()
    assert confmat.shape == (41, 94)
    confmat = matcher.match_confmat(confmat)
    assert confmat.shape == (94, 94)
    print(confmat>0)
    assert np.all(confmat[np.eye(94) == 0] == 0)
    assert np.diag(confmat).sum() > 0


def test_label_in_smaller_space():
    scene_dir = Path('/media/blumh/data/replica/office_0/Sequence_1/')
    keys = sorted(
        int(x.name.split('.')[0].split('_')[1])
        for x in (scene_dir / 'rgb').iterdir())[:1]
    label_template = 'semantic_class/semantic_class_{k}.png'
    matcher = LabelMatcher('replicaid', 'nyu40id')
    confmat = np.zeros((len(matcher.left_ids) + 1, len(matcher.right_ids) + 1),
                       dtype=np.int64)
    left_id_to_confmat_idx = np.zeros(max(matcher.left_ids) + 1,
                                      dtype=np.int64)
    for i, left_id in enumerate(matcher.left_ids):
        left_id_to_confmat_idx[left_id] = i + 1
    right_id_to_confmat_idx = np.zeros(max(matcher.right_ids) + 1,
                                       dtype=np.int64)
    for i, right_id in enumerate(matcher.right_ids):
        right_id_to_confmat_idx[right_id] = i + 1
    non_matched_labels = 0
    for k in tqdm(keys):
        label = cv2.imread(str(scene_dir / label_template.format(k=k)),
                          cv2.IMREAD_UNCHANGED)
        # map the prediction from replica to nyu40id
        nyu_label = matcher.mapping[label]
        if np.sum(nyu_label == -2) > 0:
            for source_id, target_ids in matcher.mapping_multiples.items():
                nyu_label[label == source_id] = target_ids[0]
        non_matched_labels += np.sum(nyu_label < 0)
        sample_weights = np.ones_like(label.flatten(), dtype=np.int64)
        left = left_id_to_confmat_idx[label.flatten()]
        right = right_id_to_confmat_idx[nyu_label.flatten()]
        confmat += coo_matrix((sample_weights, (left, right)),
                              shape=confmat.shape,
                              dtype=np.int64).toarray()
    assert confmat.shape == (94, 41)
    confmat = matcher.match_confmat(confmat)
    assert confmat.shape == (41, 41)
    # there are some labels that cannot be matched between NYU and replica
    assert np.sum(confmat[:, 0]) == 0
    assert np.sum(confmat[0, :]) == non_matched_labels
    assert np.all(confmat[1:, 1:][np.eye(40) == 0] == 0)
    assert np.diag(confmat).sum() > 0
