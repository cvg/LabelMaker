import os
import argparse
import logging
from tqdm import tqdm
import cv2
import numpy as np
import shutil
from joblib import Parallel, delayed

from labelmaker.label_mappings import LabelMatcher

# clean up imports
import gin
from typing import Union
from pathlib import Path

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
    self.votes_from_ade150 = np.zeros((150, self.output_size), dtype=np.uint8)
    for ade150_id in range(150):
      multihot_matches = matcher_ade150.match(
          ade150_id * np.ones_like(output_ids), output_ids)
      multihot_matches[multihot_matches == -1] = 0
      multihot_matches[multihot_matches == -2] = 0
      self.votes_from_ade150[ade150_id] = multihot_matches

    self.votes_from_nyu40 = np.zeros((41, self.output_size), dtype=np.uint8)
    for nyu40_id in range(1, 41):
      multihot_matches = matcher_nyu40.match(
          nyu40_id * np.ones_like(output_ids), output_ids)
      multihot_matches[multihot_matches == -1] = 0
      multihot_matches[multihot_matches == -2] = 0
      self.votes_from_nyu40[nyu40_id] = multihot_matches

    self.votes_from_wn199 = np.zeros((200, self.output_size), dtype=np.uint8)
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
    votes = np.zeros((shape[0], shape[1], self.output_size), dtype=np.uint8)
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

    top_two_pred = votes.argsort(axis=2)[:, :, -2:]

    first_pred = top_two_pred[:, :, 1]
    second_pred = top_two_pred[:, :, 0]

    num_first_votes = votes[
        np.arange(shape[0])[:, None],
        np.arange(shape[1]),
        first_pred,
    ]
    num_second_votes = votes[
        np.arange(shape[0])[:, None],
        np.arange(shape[1]),
        second_pred,
    ]
    # if second vote is zero, then assign the label to zero
    second_pred[num_second_votes == 0] = 0

    return first_pred, num_first_votes, second_pred, num_second_votes


VALID_LABEL_SPACES = ['ade20k', 'nyu40', 'scannet200', 'wordnet', 'scannet']


def consensus(k, folders, output_dir, min_votes):

  votebox = PredictorVoting(output_space='wn199-merged-v2')

  predictions = {label_space: [] for label_space in VALID_LABEL_SPACES}

  for folder in folders:
    assert folder.exists() and folder.is_dir()

    label_space = folder.name.split('_')[0]
    pred = cv2.imread(str(folder / f'{k}.png'), cv2.IMREAD_UNCHANGED)
    predictions[label_space].append(pred.copy())

  first_pred, num_first_votes, second_pred, num_second_votes = votebox.voting(
      ade20k_predictions=predictions['ade20k'],
      nyu40_predictions=predictions['nyu40'],
      wn199_predictions=predictions['wordnet'],
      scannet_predictions=predictions['scannet200']
  )  # double even without flipping

  pred_vote = first_pred.copy()
  pred_vote[num_first_votes < min_votes] = 0
  aux_data = np.stack([second_pred, num_first_votes, num_second_votes], axis=-1)

  cv2.imwrite(str(output_dir / f'{k}.png'), pred_vote.astype(np.uint8))
  cv2.imwrite(str(output_dir / f'{k}_aux.png'), aux_data.astype(np.uint8))


# this is needed for parallel execution
def wrapper_consensus(k, input_folders_str, output_dir_str, min_votes):
  input_folders = [Path(s) for s in input_folders_str]
  output_dir = Path(output_dir_str)
  consensus(k, input_folders, output_dir, min_votes)
  return 1


@gin.configurable
def run(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    n_jobs=-1,
    min_votes=2,
    custrom_vote_weight: bool = False,
):

  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  assert scene_dir.exists() and scene_dir.is_dir()

  output_dir = scene_dir / output_folder
  # check if output directory exists
  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)

  log.info('[consensus] loading model predictions')
  input_folders = [
      scene_dir / 'intermediate' / folder
      for folder in os.listdir(scene_dir / 'intermediate')
      if folder.split('_')[0] in VALID_LABEL_SPACES
  ]
  input_folders = []
  for folder in os.listdir(scene_dir / 'intermediate'):
    if folder.split('_')[0] in VALID_LABEL_SPACES and (
        scene_dir / 'intermediate' / folder).is_dir():
      if custrom_vote_weight:
        try:
          vote_weight = max(int(folder.split('_')[2]), 1)
        except:
          vote_weight = 1
      else:
        vote_weight = 1
      input_folders += [scene_dir / 'intermediate' / folder] * vote_weight

  # assert that all folders have the same number of files
  n_files = None
  for folder in input_folders:
    files = [
        f for f in os.listdir(scene_dir / 'intermediate' / folder)
        if f.endswith('.png')
    ]
    if n_files is None:
      n_files = len(files)
    else:
      assert n_files == len(
          files
      ), f'Number of files in {folder} does not match {n_files} vs. {len(files)}'

  keys = sorted([s.stem for s in (scene_dir / 'color').iterdir()])

  input_folders_str = [str(f) for f in input_folders]
  output_dir_str = str(output_dir)

  # Using Parallel to run the function in parallel
  results = Parallel(n_jobs=n_jobs)(delayed(wrapper_consensus)(
      k, input_folders_str, output_dir_str, min_votes) for k in tqdm(keys))


def arg_parser():
  parser = argparse.ArgumentParser(description='Run consensus segmentation')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help='Path to workspace directory. There should be a "color" folder.',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/consensus',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version',
  )
  parser.add_argument(
      '--custom_vote_weight',
      action="store_true",
      help=
      'Indicate vote in naming scheme, [label_space]_[model]_[vote]_[run_id] instead of [label_space]_[model]_[run_id]',
  )
  parser.add_argument("--n_jobs", type=int, default=-1)
  parser.add_argument('--config', help='Name of config file')

  return parser.parse_args()


if __name__ == '__main__':
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  run(
      scene_dir=args.workspace,
      output_folder=args.output,
      n_jobs=args.n_jobs,
      custrom_vote_weight=args.custom_vote_weight,
  )
