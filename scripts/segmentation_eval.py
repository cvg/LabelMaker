import sys, os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from joblib import Parallel, delayed

from segmentation_tools.label_mappings import LabelMatcher

logging.basicConfig(level="INFO")
log = logging.getLogger('Segmentation Evaluation')


def _get_confmat(scene_dir, keys, pred_space, label_space, pred_template,
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


def evaluate(scene_dir,
             keys,
             pred_space,
             label_space,
             pred_template='pred/{k}.png',
             label_template='label_filt/{k}.png',
             n_jobs=8):
    log.info('matching predictions')
    # split keys into chunks for parallel execution
    keys = np.array_split(keys, n_jobs)
    confmats = Parallel(n_jobs=n_jobs)(
        delayed(_get_confmat)(scene_dir, keys[i], pred_space, label_space,
                              pred_template, label_template)
        for i in range(n_jobs))
    confmat = np.sum(confmats, axis=0)
    float_confmat = confmat.astype(float)
    metrics = {
        'iou':
        np.diag(float_confmat) /
        (float_confmat.sum(axis=1) + float_confmat.sum(axis=0) -
         np.diag(float_confmat)),
        'acc':
        np.diag(float_confmat) / float_confmat.sum(),
    }
    metrics['mIoU'] = metrics['iou'].mean()
    metrics['mAcc'] = metrics['acc'].mean()
    return metrics, confmat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor')
    parser.add_argument('scene')
    parser.add_argument('--replica', default=False)
    parser.add_argument('--j', default=8)
    flags = parser.parse_args()
    scene_dir = Path(flags.scene)
    assert scene_dir.exists() and scene_dir.is_dir()
    if flags.replica:
        keys = sorted(
            int(x.name.split('.')[0].split('_')[1])
            for x in (scene_dir / 'rgb').iterdir())
        label_template = 'semantic_class/semantic_class_{k}.png'
        label_space = 'replicaid'
    else:
        keys = sorted(
            int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
        img_template = 'label_filt/{k}.png'
        label_space = 'id'
    # load predictor settings
    if flags.predictor == 'internimage':
        pred_template = 'pred_internimage/{k}.png'
        pred_space = 'ade20k'
    elif flags.predictor == 'cmx':
        pred_template = 'pred_cmx/{k}.png'
        pred_space = 'nyu40id'
    elif flags.predictor == 'ovseg_wn':
        pred_template = 'pred_ovseg_wordnet/{k}.png'
        pred_space = 'wn199'
    else:
        raise ValueError(f'unknown predictor {flags.predictor}')
    metrics, confmat = evaluate(scene_dir,
                                keys,
                                pred_space,
                                label_space,
                                pred_template=pred_template,
                                label_template=label_template,
                                n_jobs=flags.j)
    print(metrics, flush=True)
