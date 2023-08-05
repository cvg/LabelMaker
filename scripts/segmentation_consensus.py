import sys, os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import shutil
from joblib import Parallel, delayed

from segmentation_tools.label_mappings import LabelMatcher, get_ade150_to_scannet, \
    MatcherScannetWordnet199, get_scannet_from_wn199

logging.basicConfig(level="INFO")
log = logging.getLogger('Segmentation Consensus')


class PredictorVoting:

    def __init__(self, output_space='wn199-merged-v2'):
        #assert output_space == 'wn199'
        matcher_ade150 = LabelMatcher('ade20k', output_space)
        matcher_nyu40 = LabelMatcher('nyu40id', output_space)
        matcher_wn199 = LabelMatcher('wn199', output_space)
        matcher_scannet = LabelMatcher('id', output_space)
        self.output_space = output_space
        # build lookup tables for predictor voting
        # some class spaces vote for multiple options in the wordnet output space
        self.output_size = max(matcher_ade150.right_ids) + 1
        output_ids = np.arange(self.output_size)
        self.votes_from_ade150 = np.zeros((150, self.output_size),
                                          dtype=np.uint8)
        for ade150_id in range(150):
            multihot_matches = matcher_ade150.match(
                ade150_id * np.ones_like(output_ids), output_ids)
            multihot_matches[multihot_matches == -1] = 0
            multihot_matches[multihot_matches == -2] = 0
            self.votes_from_ade150[ade150_id] = multihot_matches

        self.votes_from_nyu40 = np.zeros((41, self.output_size),
                                         dtype=np.uint8)
        for nyu40_id in range(1, 41):
            multihot_matches = matcher_nyu40.match(
                nyu40_id * np.ones_like(output_ids), output_ids)
            multihot_matches[multihot_matches == -1] = 0
            multihot_matches[multihot_matches == -2] = 0
            self.votes_from_nyu40[nyu40_id] = multihot_matches

        self.votes_from_wn199 = np.zeros((200, self.output_size),
                                         dtype=np.uint8)
        for wn199_id in range(1, 189):
            multihot_matches = matcher_wn199.match(
                wn199_id * np.ones_like(output_ids), output_ids)
            multihot_matches[multihot_matches == -1] = 0
            multihot_matches[multihot_matches == -2] = 0
            self.votes_from_wn199[wn199_id] = multihot_matches

        scannet_dimensionality = max(matcher_scannet.left_ids) + 1
        self.votes_from_scannet = np.zeros(
            (scannet_dimensionality, self.output_size), dtype=np.uint8)
        for scannet_id in range(scannet_dimensionality):
            multihot_matches = matcher_scannet.match(
                scannet_id * np.ones_like(output_ids), output_ids)
            multihot_matches[multihot_matches == -1] = 0
            multihot_matches[multihot_matches == -2] = 0
            self.votes_from_scannet[scannet_id] = multihot_matches

    def voting(self,
               ade20k_predictions=[],
               nyu40_predictions=[],
               wn199_predictions=[],
               scannet_predictions=[]):
        """Voting scheme for combining multiple segmentation predictors.

        Args:
            ade20k_predictors (list): list of ade20k predictions
            nyu40_predictors (list): list of nyu40 predictions
            wn199_predictors (list): list of wn199 predictions
            scannet_predictions (list): list of scannet predictions

        Returns:
            np.ndarray: consensus prediction in the output space
        """
        shape = None
        if len(ade20k_predictions) > 0:
            shape = ade20k_predictions[0].shape[:2]
        elif len(nyu40_predictions) > 0:
            shape = nyu40_predictions[0].shape[:2]
        elif len(wn199_predictions) > 0:
            shape = wn199_predictions[0].shape[:2]
        elif len(scannet_predictions) > 0:
            shape = scannet_predictions[0].shape[:2]


        # build consensus prediction
        # first, each prediction votes for classes in the output space
        votes = np.zeros((shape[0], shape[1], self.output_size),
                         dtype=np.uint8)
        for pred in wn199_predictions:
            vote = self.votes_from_wn199[pred]
            vote[pred == -1] = 0
            votes += vote
        for pred in ade20k_predictions:
            votes += self.votes_from_ade150[pred]
        for pred in nyu40_predictions:
            votes += self.votes_from_nyu40[pred]
            
        for pred in scannet_predictions:
            votes += self.votes_from_scannet[pred]

        pred_vote = np.argmax(votes, axis=2)
        n_votes = votes[np.arange(shape[0])[:, None],
                        np.arange(shape[1]), pred_vote]
        #n_votes = np.amax(votes, axis=2)
        # # fastest check for ambiguous prediction: take the argmax in reverse order
        # alt_pred = (self.output_size - 1) - np.argmax(votes[:, :, ::-1],
        #                                               axis=2)
        # pred_vote[pred_vote != alt_pred] = -1
        return n_votes, pred_vote


def build_scannet_consensus(scene_dir,
                            n_jobs=8,
                            min_votes=2,
                            use_scannet=True,
                            scannet_weight=3,
                            no_ovseg=False,
                            no_mask3d=False,
                            no_cmx=False,
                            no_intern=False,
                            output_dir=None):
    
    scene_dir = Path(scene_dir)
    assert scene_dir.exists() and scene_dir.is_dir()
    keys = sorted(
        int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
    if use_scannet and output_dir is None:
        output_dir = scene_dir / f'pred_consensus_{scannet_weight}_scannet'
    elif output_dir is None:
        output_dir = scene_dir / 'pred_consensus_noscannet'
        scannet_weight=0
    else:
        output_dir = scene_dir / output_dir

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=False)

    def consensus(k):
        votebox = PredictorVoting(output_space='wn199-merged-v2')
        intern_ade150 = cv2.imread(
            str(scene_dir / 'pred_internimage' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        intern_ade150_flip = cv2.imread(
            str(scene_dir / 'pred_internimage_flip' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        cmx_nyu40 = cv2.imread(str(scene_dir / 'pred_cmx' / f'{k}.png'),
                               cv2.IMREAD_UNCHANGED)
        cmx_nyu40 = cv2.resize(cmx_nyu40, intern_ade150.shape[:2][::-1],
                               interpolation=cv2.INTER_NEAREST)
        cmx_nyu40_flip = cv2.imread(
            str(scene_dir / 'pred_cmx_flip' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        cmx_nyu40_flip = cv2.resize(cmx_nyu40_flip,
                                    intern_ade150.shape[:2][::-1],
                                    interpolation=cv2.INTER_NEAREST)
        ovseg_wn199 = cv2.imread(
            str(scene_dir / 'pred_ovseg_wn_nodef' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        ovseg_wn199_flip = cv2.imread(
            str(scene_dir / 'pred_ovseg_wn_nodef_flip' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        mask3d = cv2.imread(
            str(scene_dir / 'pred_mask3d_rendered' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        
        if use_scannet:
            scannet_labels = cv2.imread(str(scene_dir / 'label-filt' / f'{k}.png'),
                                        cv2.IMREAD_UNCHANGED)
        # n_votes, pred_vote = votebox.voting(
        #     ade20k_predictions=[intern_ade150, intern_ade150_flip],
        #     nyu40_predictions=[cmx_nyu40, cmx_nyu40_flip],
        #     wn199_predictions=[ovseg_wn199, ovseg_wn199_flip],
        #     scannet_predictions=[mask3d, mask3d]) # double even without flipping

        if not no_intern:
            ade20k_predictions = [intern_ade150, intern_ade150_flip]
        else:
            ade20k_predictions = []
        
        if not no_cmx:
            nyu40_predictions = [cmx_nyu40, cmx_nyu40_flip]
        else:
            nyu40_predictions = []
        
        if not no_ovseg:
            wn199_predictions = [ovseg_wn199, ovseg_wn199_flip]
        else:
            wn199_predictions = []
        
        if not no_mask3d:
            scannet_predictions = [mask3d, mask3d]
        else:
            scannet_predictions = []

        if use_scannet:
            scannet_prediction += [*(scannet_labels for _ in range(scannet_weight))]

        n_votes, pred_vote = votebox.voting(
            ade20k_predictions=ade20k_predictions,
            nyu40_predictions=nyu40_predictions,
            wn199_predictions=wn199_predictions,
            scannet_predictions=scannet_predictions)  # double even without flipping
        pred_vote[n_votes < min_votes] = 0
        pred_vote[pred_vote == -1] = 0
        cv2.imwrite(str(output_dir / f'{k}.png'), pred_vote)

    Parallel(n_jobs=n_jobs)(delayed(consensus)(k) for k in tqdm(keys))


def build_replica_consensus(scene_dir, n_jobs=16, min_votes=2, wn=False):
    scene_dir = Path(scene_dir)
    assert scene_dir.exists() and scene_dir.is_dir()
    keys = sorted(
        int(x.name.split('.')[0].split('_')[1])
        for x in (scene_dir / 'rgb').iterdir())
    if wn:
        output_dir = scene_dir / 'pred_wn_consensus'
    else:
        output_dir = scene_dir / 'pred_consensus'
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=False)

    def consensus(k):
        if wn:
            votebox = PredictorVoting(output_space='wn199-merged-v2')
        else:
            votebox = PredictorVoting(output_space='replicaid')
        intern_ade150 = cv2.imread(
            str(scene_dir / 'pred_internimage' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        intern_ade150_flip = cv2.imread(
            str(scene_dir / 'pred_internimage_flip' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        cmx_nyu40 = cv2.imread(str(scene_dir / 'pred_cmx' / f'{k}.png'),
                               cv2.IMREAD_UNCHANGED)
        cmx_nyu40_flip = cv2.imread(
            str(scene_dir / 'pred_cmx_flip' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        ovseg_wn199 = cv2.imread(
            str(scene_dir / 'pred_ovseg_wn_nodef' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        ovseg_wn199_flip = cv2.imread(
            str(scene_dir / 'pred_ovseg_wn_nodef_flip' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        mask3d = cv2.imread(
            str(scene_dir / 'pred_mask3d_rendered' / f'{k}.png'),
            cv2.IMREAD_UNCHANGED)
        # n_votes, pred_vote = votebox.voting(
        #     ade20k_predictions=[intern_ade150, intern_ade150_flip],
        #     nyu40_predictions=[cmx_nyu40, cmx_nyu40_flip],
        #     wn199_predictions=[ovseg_wn199, ovseg_wn199_flip],
        #     scannet_predictions=[mask3d, mask3d]) # double even without flipping
        n_votes, pred_vote = votebox.voting(
            ade20k_predictions=[
                intern_ade150, intern_ade150_flip, intern_ade150
            ],
            nyu40_predictions=[cmx_nyu40, cmx_nyu40_flip],
            wn199_predictions=[ovseg_wn199, ovseg_wn199_flip],
            scannet_predictions=[mask3d,
                                 mask3d])  # double even without flipping
        pred_vote[n_votes < min_votes] = 0
        pred_vote[pred_vote == -1] = 0
        cv2.imwrite(str(output_dir / f'{k}.png'), pred_vote)

    Parallel(n_jobs=n_jobs)(delayed(consensus)(k) for k in tqdm(keys))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replica', default=False)
    parser.add_argument('--use_scannet', action='store_true')
    parser.add_argument('--votes', default=5)
    parser.add_argument('--scannet_weight', default=3, type=int)
    parser.add_argument('--no_ovseg', action='store_true')
    parser.add_argument('--no_mask3d', action='store_true')
    parser.add_argument('--no_cmx', action='store_true')
    parser.add_argument('--no_intern', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('scene', type=str)
    flags = parser.parse_args()


    if flags.use_scannet:
        print("will use scannet annotations in consensus")

    if flags.replica:
        build_replica_consensus(flags.scene,
                                min_votes=int(flags.votes),
                                wn=True)
    else:
        build_scannet_consensus(flags.scene, 
                                min_votes=int(flags.votes), 
                                use_scannet=bool(flags.use_scannet),
                                scannet_weight=flags.scannet_weight,
                                no_ovseg=flags.no_ovseg,
                                no_mask3d=flags.no_mask3d,
                                no_cmx=flags.no_cmx,
                                no_intern=flags.no_intern,
                                output_dir=flags.output_dir)
