import sys, os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from glob import glob
import re
from scipy.sparse import coo_matrix
import open3d as o3d
from joblib import Parallel, delayed

from segmentation_tools.label_mappings import LabelMatcher

logging.basicConfig(level="INFO")
log = logging.getLogger('Segmentation Evaluation')


def _dist_get_matcher_confmat(scene_dir, keys, pred_space, label_space,
                              pred_template, label_template):
    matcher = LabelMatcher(pred_space, label_space)
    confmat = np.zeros((len(matcher.right_ids), len(matcher.right_ids)),
                       dtype=np.int64)
    for k in tqdm(keys):
        pred = cv2.imread(str(scene_dir / pred_template.format(k=k)),
                          cv2.IMREAD_UNCHANGED)
        label = cv2.imread(str(scene_dir / label_template.format(k=k)),
                           cv2.IMREAD_UNCHANGED)
        if pred.shape[0] != label.shape[0] or pred.shape[1] != label.shape[1]:
            pred = cv2.resize(pred, (label.shape[1], label.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        confmat += matcher.confusion_matrix(pred, label)
    return confmat


def _dist_get_unmatched_confmat(scene_dir, point_indices, pred_space,
                                label_space, pred_pointcloud, pred_classes,
                                label_pointcloud, label_classes):
    matcher = LabelMatcher(pred_space, label_space)
    confmat = np.zeros((len(matcher.left_ids) + 1, len(matcher.right_ids) + 1),
                       dtype=np.int64)
    # we do not know whether all predictions or labels actually only contain the ids listed,
    # or if there are gaps in the data
    # Therefore, we keep 0 in each dimension as a "not in list" category
    left_id_to_confmat_idx = np.zeros(max(matcher.left_ids) + 1,
                                      dtype=np.int64)
    for i, left_id in enumerate(matcher.left_ids):
        left_id_to_confmat_idx[left_id] = i + 1
    right_id_to_confmat_idx = np.zeros(max(matcher.right_ids) + 1,
                                       dtype=np.int64)
    for i, right_id in enumerate(matcher.right_ids):
        right_id_to_confmat_idx[right_id] = i + 1

    pc_pred = o3d.io.read_point_cloud(str(scene_dir / pred_pointcloud))
    knn_pred = o3d.geometry.KDTreeFlann(pc_pred)
    pc_label = o3d.io.read_point_cloud(str(scene_dir / label_pointcloud))
    if pred_classes.endswith('.npy'):
        pred = np.load(str(scene_dir / pred_classes))
    elif pred_classes.endswith('.txt'):
        pred = np.loadtxt(str(scene_dir / pred_classes), dtype=np.int64)
    else:
        raise ValueError(f'Unknown prediction file format {pred_classes}')
    if label_classes.endswith('.npy'):
        label = np.load(str(scene_dir / label_classes))
    elif label_classes.endswith('.txt'):
        label = np.loadtxt(str(scene_dir / label_classes), dtype=np.int64)
    else:
        raise ValueError(f'Unknown label file format {label_classes}')
    assert pred.shape[0] == np.asarray(pc_pred.points).shape[0]
    assert label.shape[0] == np.asarray(pc_label.points).shape[0]
    for i in tqdm(point_indices):
        if label[i] == 0:
            continue
        if label[i] == -1:
            continue
        # find nearest neighbor in prediction
        [_, idx, _] = knn_pred.search_knn_vector_3d(pc_label.points[i], 1)
        nearest_prediction = pred[idx]

        left = left_id_to_confmat_idx[nearest_prediction]
        right = right_id_to_confmat_idx[label[i]]
        confmat[left, right] += 1
    return confmat


"""
def _get_confmat(scene_dir,
                 keys,
                 pred_space,
                 label_space,
                 pred_template,
                 label_template,
                 n_jobs=8):
    confmat_path = scene_dir / pred_template.split(
        '/')[0] / f'confmat_{label_space}.txt'
    if confmat_path.exists():
        log.info(f'using cached {confmat_path}')
        return np.loadtxt(str(confmat_path)).astype(np.int64)
    # split keys into chunks for parallel execution
    keys = np.array_split(keys, n_jobs)
    confmats = Parallel(n_jobs=n_jobs)(
        delayed(_dist_get_confmat)(scene_dir, keys[i], pred_space, label_space,
                                   pred_template, label_template)
        for i in range(n_jobs))
    confmat = np.sum(confmats, axis=0)
    np.savetxt(str(confmat_path), confmat)
    return confmat.astype(np.int64)
"""


def metrics_from_confmat(confmat):
    assert confmat.shape[0] == confmat.shape[1]
    assert confmat[:, 0].sum() == 0
    float_confmat = confmat.astype(float)
    metrics = {
        'iou':
        np.diag(float_confmat) /
        (float_confmat.sum(axis=1) + float_confmat.sum(axis=0) -
         np.diag(float_confmat)),
        'acc':
        np.diag(float_confmat) / float_confmat.sum(0),
    }
    metrics['acc'] = np.nan_to_num(metrics['acc'])[1:]  # fill with 0
    metrics['iou'] = np.nan_to_num(metrics['iou'])[1:]  # fill with 0
    metrics['mIoU'] = metrics['iou'].mean()
    metrics['mAcc'] = metrics['acc'].mean()
    metrics['tAcc'] = np.diag(float_confmat).sum() / float_confmat.sum()
    return metrics


def _get_confmat(scene_dir,
                 pred_space,
                 label_space,
                 pred_dir,
                 pred_pointcloud,
                 pred_classes,
                 label_pointcloud,
                 label_classes,
                 n_jobs=4):
    confmat_path = scene_dir / pred_dir / f'3dconfmat_{pred_space}_{label_space}.txt'
    if confmat_path.exists():
        confmat = np.loadtxt(str(confmat_path)).astype(np.int64)
    else:
        # split points into chunks for parallel execution
        pc_label = o3d.io.read_point_cloud(str(scene_dir / label_pointcloud))
        point_indices = np.arange(np.asarray(pc_label.points).shape[0])
        point_indices = np.array_split(point_indices, n_jobs)
        confmats = Parallel(n_jobs=n_jobs)(
            delayed(_dist_get_unmatched_confmat)(
                scene_dir, point_indices[i], pred_space, label_space,
                pred_pointcloud, pred_classes, label_pointcloud, label_classes)
            for i in range(n_jobs))
        confmat = np.sum(confmats, axis=0)
        np.savetxt(str(confmat_path), confmat)
    matcher = LabelMatcher(pred_space, label_space)
    return matcher.match_confmat(confmat)


def evaluate_scene(scene_dir,
                   pred_space,
                   label_space,
                   pred_dir,
                   pred_pointcloud,
                   pred_classes,
                   label_pointcloud,
                   label_classes,
                   n_jobs=4):
    scene_dir = Path(scene_dir)
    log.info(f"getting confmat for {pred_pointcloud} in {scene_dir}")
    confmat = _get_confmat(scene_dir,
                           pred_space,
                           label_space,
                           pred_dir,
                           pred_pointcloud,
                           pred_classes,
                           label_pointcloud,
                           label_classes,
                           n_jobs=n_jobs)
    metrics = metrics_from_confmat(confmat)
    return metrics, confmat


def evaluate_scenes(scene_dirs,
                    pred_space,
                    label_space,
                    pred_dir,
                    pred_pointcloud,
                    pred_classes,
                    label_pointcloud,
                    label_classes,
                    n_jobs=8):
    confmat = None
    for scene_dir in scene_dirs:
        _, c = evaluate_scene(scene_dir,
                              pred_space,
                              label_space,
                              pred_dir=pred_dir,
                              pred_pointcloud=pred_pointcloud,
                              pred_classes=pred_classes,
                              label_pointcloud=label_pointcloud,
                              label_classes=label_classes,
                              n_jobs=n_jobs)
        if confmat is None:
            confmat = c
        else:
            confmat += c
    metrics = metrics_from_confmat(confmat)
    return metrics, confmat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--replica', default=False)
    parser.add_argument('--j', default=8)
    flags = parser.parse_args()
    scene_dir = Path(flags.scene)
    assert scene_dir.exists() and scene_dir.is_dir()
    if flags.replica:
        label_template = 'semantic_class/semantic_class_{k}.png'
        label_space = 'replicaid'
    else:
        label_template = 'label_agile3d/{k}.png'
        label_space = 'wn199'

    # check which predictors are present
    for subdir in scene_dir.iterdir():
        if subdir.is_dir():
            if subdir.name == 'pred_internimage':
                pred_space = 'ade20k'
                pred_template = 'pred_internimage/{k}.png'
            elif subdir.name == 'pred_cmx':
                pred_space = 'nyu40id'
                pred_template = 'pred_cmx/{k}.png'
            elif subdir.name == 'pred_consensus':
                if flags.replica:
                    pred_space = 'replicaid'
                else:
                    pred_space = 'wn199'
                pred_template = 'pred_consensus/{k}.png'
            elif subdir.name == 'pred_wn_consensus':
                pred_space = 'wn199'
                pred_template = 'pred_wn_consensus/{k}.png'
            elif subdir.name == 'pred_ovseg_replica':
                pred_space = 'replicaid'
                pred_template = 'pred_ovseg_replica/{k}.png'
            elif subdir.name.startswith('pred_ovseg_w'):
                pred_space = 'wn199'
                pred_template = subdir.name + '/{k}.png'
            elif subdir.name == 'label-filt':
                pred_space = 'id'
                pred_template = 'label-filt/{k}.png'
            elif subdir.name == 'nerf':
                pred_space = 'replicaid'
                pred_template = 'nerf/pred_nerf_{k}.png'
            elif subdir.name == 'pred_mask3d_rendered':
                pred_space = 'id'
                pred_template = 'pred_mask3d_rendered/{k}.png'
            elif subdir.name.startswith('pred_sdfstudio'):
                if flags.replica:
                    pred_space = 'replicaid'
                else:
                    pred_space = 'wn199'
                pred_template = subdir.name + '/{k:05d}.png'
            else:
                continue
        metrics, confmat = evaluate_scene(scene_dir,
                                          pred_space,
                                          label_space,
                                          pred_template=pred_template,
                                          label_template=label_template,
                                          n_jobs=int(flags.j))
