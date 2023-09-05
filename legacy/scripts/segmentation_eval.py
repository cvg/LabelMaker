import sys, os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from glob import glob
import re
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix

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


def _dist_get_unmatched_confmat(scene_dir, keys, pred_space, label_space,
                                pred_template, label_template, subsampling):
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
    for k in tqdm(keys):
        pred = cv2.imread(str(scene_dir / pred_template.format(k=(k // subsampling))),
                          cv2.IMREAD_UNCHANGED)
        label = cv2.imread(str(scene_dir / label_template.format(k=k)),
                           cv2.IMREAD_UNCHANGED)
        if pred.shape[0] != label.shape[0] or pred.shape[1] != label.shape[1]:
            pred = cv2.resize(pred, (label.shape[1], label.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        sample_weights = np.ones_like(label.flatten(), dtype=np.int64)
        left = left_id_to_confmat_idx[pred.flatten()]
        right = right_id_to_confmat_idx[label.flatten()]
        confmat += coo_matrix((sample_weights, (left, right)),
                              shape=confmat.shape,
                              dtype=np.int64).toarray()
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

    nan_mask_c = confmat[1:, :].sum(axis=1) == 0 # no prediction for this class
    nan_mask_r = confmat[:, 1:].sum(axis=0) == 0 # no groundtruth for this class

    nan_mask = np.logical_and(nan_mask_c, nan_mask_r) 
    nan_mask = nan_mask_r

    acc = np.nan_to_num(metrics['acc'][1:], 0)  # fill with 0
    iou = np.nan_to_num(metrics['iou'][1:], 0)  # fill with 0

    metrics['mIoU'] = (iou * (1 - nan_mask)).sum() / (1 - nan_mask).sum()
    metrics['mAcc'] = (acc * (1 - nan_mask)).sum() / (1 - nan_mask).sum()

    # metrics['mIoU'] = iou.mean()
    # metrics['mAcc'] = acc.mean()

    metrics['tAcc'] = np.diag(float_confmat).sum() / float_confmat.sum()

    acc[nan_mask == 1] = 'nan'
    iou[nan_mask == 1] = 'nan'

    metrics['acc'] = acc.copy()
    metrics['iou'] = iou.copy()



    return metrics


def _get_confmat(scene_dir,
                 keys,
                 pred_space,
                 label_space,
                 pred_template,
                 label_template,
                 subsampling=1,
                 overwrite_confmat=False,
                 n_jobs=8):
    confmat_path = scene_dir / pred_template.split(
        '/')[0] / f'confmat_{pred_space}_{label_space}.txt'
    if confmat_path.exists() and not overwrite_confmat:
        confmat =  np.loadtxt(str(confmat_path)).astype(np.int64)
    else:
        # split keys into chunks for parallel execution
        keys = np.array_split(keys, n_jobs)
        confmats = Parallel(n_jobs=n_jobs)(delayed(_dist_get_unmatched_confmat)(
            scene_dir, keys[i], pred_space, label_space, pred_template,
            label_template, subsampling) for i in range(n_jobs))
        confmat = np.sum(confmats, axis=0)
        np.savetxt(str(confmat_path), confmat)
    matcher = LabelMatcher(pred_space, label_space)
    return matcher.match_confmat(confmat)


def evaluate_scene(scene_dir,
                   pred_space,
                   label_space,
                   keys=None,
                   subsampling=1,
                   pred_template='pred/{k}.png',
                   label_template='label_filt/{k}.png',
                   overwrite_confmat=False,
                   n_jobs=8):
    scene_dir = Path(scene_dir)
    if keys is None:
        files = glob(str(scene_dir / label_template.format(k='*')),
                     recursive=True)
        keys = sorted(
            int(re.search(label_template.format(k='(\d+)'), x).group(1))
            for x in files)
        keys = keys[::subsampling]
    
    log.info(
        f"getting confmat for {pred_template.split('/')[0]} in {scene_dir}")
    confmat = _get_confmat(scene_dir,
                           keys,
                           pred_space,
                           label_space,
                           pred_template,
                           label_template,
                           subsampling=subsampling,
                           overwrite_confmat=overwrite_confmat,
                           n_jobs=n_jobs)
    metrics = metrics_from_confmat(confmat)
    return metrics, confmat


def evaluate_scenes(scene_dirs,
                    pred_space,
                    label_space,
                    subsampling=1,
                    pred_template='pred/{k}.png',
                    label_template='label_filt/{k}.png',
                    overwrite_confmat=False,
                    n_jobs=8):
    confmat = None
    for k, scene_dir in enumerate(scene_dirs):

         

        _, c = evaluate_scene(scene_dir,
                              pred_space,
                              label_space,
                              pred_template=pred_template[k] if type(pred_template) is list else pred_template,
                              label_template=label_template,
                              subsampling=subsampling,
                              overwrite_confmat=overwrite_confmat,
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
