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

from segmentation_tools.label_mappings import LabelMatcher

logging.basicConfig(level="INFO")
log = logging.getLogger('Segmentation Evaluation')


def _dist_get_confmat(scene_dir, keys, pred_space, label_space, pred_template,
                      label_template):
    matcher = LabelMatcher(pred_space, label_space)
    confmat = np.zeros((len(matcher.right_ids), len(matcher.right_ids)),
                       dtype=np.int64)
    for k in tqdm(keys):
        pred = cv2.imread(str(scene_dir / pred_template.format(k=k)),
                          cv2.IMREAD_UNCHANGED)
        label = cv2.imread(str(scene_dir / label_template.format(k=k)),
                           cv2.IMREAD_UNCHANGED)
        confmat += matcher.confusion_matrix(pred, label)
    return confmat


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


def metrics_from_confmat(confmat):
    float_confmat = confmat.astype(float)
    metrics = {
        'iou':
        np.diag(float_confmat) /
        (float_confmat.sum(axis=1) + float_confmat.sum(axis=0) -
         np.diag(float_confmat)),
        'acc':
        np.diag(float_confmat) / float_confmat.sum(0),
    }
    metrics['iou'] = np.nan_to_num(metrics['iou'])  # fill with 0
    metrics['mIoU'] = metrics['iou'].mean()
    metrics['mAcc'] = metrics['acc'].mean()
    metrics['tAcc'] = np.diag(float_confmat).sum() / float_confmat.sum()
    return metrics


def evaluate_scene(scene_dir,
                   pred_space,
                   label_space,
                   keys=None,
                   pred_template='pred/{k}.png',
                   label_template='label_filt/{k}.png',
                   n_jobs=8):
    scene_dir = Path(scene_dir)
    if keys is None:
        files = glob(str(scene_dir / label_template.format(k='*')),
                     recursive=True)
        keys = sorted(
            int(re.search(label_template.format(k='(\d+)'), x).group(1))
            for x in files)
    log.info(
        f"getting confmat for {pred_template.split('/')[0]} in {scene_dir}")
    confmat = _get_confmat(scene_dir,
                           keys,
                           pred_space,
                           label_space,
                           pred_template,
                           label_template,
                           n_jobs=n_jobs)
    metrics = metrics_from_confmat(confmat)
    return metrics, confmat


def evaluate_scenes(scene_dirs,
                    pred_space,
                    label_space,
                    pred_template='pred/{k}.png',
                    label_template='label_filt/{k}.png',
                    n_jobs=8):
    confmat = None
    for scene_dir in scene_dirs:
        _, c = evaluate_scene(scene_dir,
                              pred_space,
                              label_space,
                              pred_template=pred_template,
                              label_template=label_template,
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
        img_template = 'label_filt/{k}.png'
        label_space = 'id'

    # check which predictors are present
    for subdir in scene_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith('pred_'):
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
            elif subdir.name == 'pred_ovseg_replica':
                pred_space = 'replicaid'
                pred_template = 'pred_ovseg_replica/{k}.png'
            elif subdir.name.startswith('pred_ovseg_w'):
                pred_space = 'wn199'
                pred_template = subdir.name + '/{k}.png'
            else:
                continue
        metrics, confmat = evaluate_scene(scene_dir,
                                          pred_space,
                                          label_space,
                                          pred_template=pred_template,
                                          label_template=label_template,
                                          n_jobs=flags.j)
