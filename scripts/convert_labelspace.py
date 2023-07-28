import sys, os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import shutil
from joblib import Parallel, delayed

from segmentation_tools.label_mappings import LabelMatcher, PredictorVoting

logging.basicConfig(level="INFO")
log = logging.getLogger('Label Conversion')



def convert_labels(scene_dir, input_space, output_space, input_template, output_template,
        n_jobs=8):
    scene_dir = Path(scene_dir)
    assert scene_dir.exists() and scene_dir.is_dir()
    keys = sorted(
        int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
    output_dir = scene_dir / output_template.split('/')[0]
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=False)

    def conversion(k):
        votebox = PredictorVoting(output_space=output_space)
        input_frame = cv2.imread(
            str(scene_dir / input_template.format(k=k)),
            cv2.IMREAD_UNCHANGED)
        ade20k = []
        nyu40 = []
        wn199 = []
        scannet = []
        if input_space == 'ade20k':
            ade20k = [input_frame]
        elif input_space == 'nyu40id':
            nyu40 = [input_frame]
        elif input_space == 'wn199':
            wn199 = [input_frame]
        elif input_space == 'id':
            scannet = [input_frame]
        _, pred_vote = votebox.voting(
            ade20k_predictions=ade20k,
            nyu40_predictions=nyu40,
            wn199_predictions=wn199,
            scannet_predictions=scannet,
            )  # double even without flipping
        pred_vote[pred_vote == -1] = 0
        cv2.imwrite(str(scene_dir / output_template.format(k=k)), pred_vote)

    Parallel(n_jobs=n_jobs)(delayed(conversion)(k) for k in tqdm(keys))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', type=str)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('input_space', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('output_space', type=str)
    flags = parser.parse_args()


    input_template = flags.input_dir + '/{k}.png'
    output_template = flags.output_dir + '/{k}.png'
    convert_labels(flags.scene, flags.input_space, flags.output_space, input_template, output_template)

