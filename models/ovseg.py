import os

# change default download location for nltk
os.environ['NLTK_DATA'] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'nltk_data'))

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Union

import cv2
import gin
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from labelmaker.label_data import get_ade150, get_replica, get_wordnet

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'ov-seg'))
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo

logging.basicConfig(level="INFO")
log = logging.getLogger('OV-Seg Segmentation')


class WordnetPromptTemplate:

  def __init__(self, template, add_synonyms=True):
    self.template = template
    self.add_synonyms = add_synonyms

  def format(self, noun):
    synset = wn.synset(noun)
    prompt = self.template.format(noun=synset.name().split('.')[0],
                                  definition=synset.definition())
    if self.add_synonyms and len(synset.lemma_names()) > 1:
      prompt += " It can also be called {}".format(", ".join(
          synset.lemma_names()[1:]))
    return prompt

  def __str__(self):
    return str(self.template)


def load_ovseg(
    device: Union[str, torch.device],
    custom_templates=None,
):
  cfg = get_cfg()
  add_deeplab_config(cfg)
  add_ovseg_config(cfg)
  cfg.merge_from_file(
      str(
          Path(__file__).parent / '..' / '3rdparty' / 'ov-seg' / 'configs' /
          'ovseg_swinB_vitL_demo.yaml'))
  cfg.merge_from_list([
      'MODEL.WEIGHTS',
      str(
          Path(__file__).parent / '..' / 'checkpoints' /
          'ovseg_swinbase_vitL14_ft_mpt.pth')
  ])

  # add device information
  cfg.MODEL.DEVICE = str(device)

  if custom_templates is not None:
    cfg.MODEL.CLIP_ADAPTER.TEXT_TEMPLATES = "predefined"
    cfg.MODEL.CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = custom_templates
  cfg.freeze()
  demo = VisualizationDemo(cfg)
  return demo


def process_image(
    model,
    img_path,
    class_names,
    threshold=0.7,
    flip=False,
):
  # use PIL, to be consistent with evaluation
  img = read_image(img_path, format="BGR")
  if flip:
    img = img[:, ::-1]
  predictions = model.predictor(img, class_names)
  blank_area = (predictions['sem_seg'][0] == 0).to('cpu').numpy()
  product, pred = torch.max(predictions['sem_seg'], dim=0)
  pred[product < threshold] = len(class_names)
  pred = pred.to('cpu').numpy().astype(int)
  pred[blank_area] = -1
  # the last text feature is the background / zero feature
  pred[pred == len(class_names)] = -1
  if flip:
    pred = pred[:, ::-1]
  return pred


def get_templates(classes):
  templates = None
  if classes == 'ade150':
    class_names = [x['name'] for x in get_ade150()]
  elif classes == 'replica':
    class_names = [x['name'] for x in get_replica()]
  elif classes == 'wordnet':
    sizeless_templates = [
        "a photo of a {size}{noun}, which is {definition}.",
        "a photo of a {size}{noun}, which can be defined as {definition}.",
        "a photo of a {size}{noun}, as in {definition}.",
        "This is a photo of a {size}{noun}, which is {definition}",
        "This is a photo of a {size}{noun}, which can be defined as {definition}",
        "This is a photo of a {size}{noun}, as in {definition}",
        "There is a {size}{noun} in the scene",
        "There is a {size}{definition} in the scene",
        "There is the {size}{noun} in the scene",
        "There is the {size}{definition} in the scene",
        "a photo of a {size}{noun} in the scene",
        "a photo of a {size}{definition} in the scene",
    ]
    templates = []
    for t in sizeless_templates:
      for s in ["", "small ", "medium ", "large "]:
        templates.append(
            WordnetPromptTemplate(
                t.format(size=s, noun="{noun}", definition="{definition}")))
    # the first class is the background class
    class_names = [x['name'] for x in get_wordnet()[1:]]
  elif classes == 'wn_nosyn':
    sizeless_templates = [
        "a photo of a {size}{noun}, which is {definition}.",
        "a photo of a {size}{noun}, which can be defined as {definition}.",
        "a photo of a {size}{noun}, as in {definition}.",
        "This is a photo of a {size}{noun}, which is {definition}",
        "This is a photo of a {size}{noun}, which can be defined as {definition}",
        "This is a photo of a {size}{noun}, as in {definition}",
        "There is a {size}{noun} in the scene",
        "There is a {size}{definition} in the scene",
        "There is the {size}{noun} in the scene",
        "There is the {size}{definition} in the scene",
        "a photo of a {size}{noun} in the scene",
        "a photo of a {size}{definition} in the scene",
    ]
    templates = []
    for t in sizeless_templates:
      for s in ["", "small ", "medium ", "large "]:
        templates.append(
            WordnetPromptTemplate(t.format(size=s,
                                           noun="{noun}",
                                           definition="{definition}"),
                                  add_synonyms=False))
    # the first class is the background class
    class_names = [x['name'] for x in get_wordnet()[1:]]
  elif classes == 'wn_nodef':
    sizeless_templates = [
        "a photo of a {size}{noun}",
        "a photo of a {size}{noun}",
        "a photo of a {size}{noun}",
        "This is a photo of a {size}{noun}.",
        "This is a photo of a {size}{noun}.",
        "This is a photo of a {size}{noun}.",
        "There is a {size}{noun} in the scene",
        "There is the {size}{noun} in the scene",
        "a photo of a {size}{noun} in the scene",
    ]
    templates = []
    for t in sizeless_templates:
      for s in ["", "small ", "medium ", "large "]:
        templates.append(WordnetPromptTemplate(t.format(size=s, noun="{noun}")))
    # the first class is the background class
    class_names = [x['name'] for x in get_wordnet()[1:]]
  elif classes == 'wn_nosyn_nodef':
    sizeless_templates = [
        "a photo of a {size}{noun}",
        "a photo of a {size}{noun}",
        "a photo of a {size}{noun}",
        "This is a photo of a {size}{noun}.",
        "This is a photo of a {size}{noun}.",
        "This is a photo of a {size}{noun}.",
        "There is a {size}{noun} in the scene",
        "There is the {size}{noun} in the scene",
        "a photo of a {size}{noun} in the scene",
    ]
    templates = []
    for t in sizeless_templates:
      for s in ["", "small ", "medium ", "large "]:
        templates.append(
            WordnetPromptTemplate(t.format(size=s, noun="{noun}"),
                                  add_synonyms=False))
    # the first class is the background class
    class_names = [x['name'] for x in get_wordnet()[1:]]
  else:
    raise ValueError(f'Unknown class set {classes}')

  return templates, class_names


@gin.configurable
def run(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    device: Union[str, torch.device] = 'cuda:0', # changing this to cuda default as all of us have it available. Otherwise, it will fail on machines without cuda
    classes='wn_nodef',
    flip=False,
):
  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  # check if scene_dir exists
  assert scene_dir.exists() and scene_dir.is_dir()

  input_color_dir = scene_dir / 'color'
  assert input_color_dir.exists() and input_color_dir.is_dir()

  output_dir = scene_dir / output_folder
  output_dir = output_dir + '_flip' if flip else output_dir
  if classes != 'wn_nodef':
    output_dir.replace('wn_nodef', classes)

  # check if output directory exists
  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)

  input_files = input_color_dir.glob('*')
  input_files = sorted(input_files, key=lambda x: int(x.stem.split('_')[-1]))

  log.info(f'[ov-seg] using {classes} classes')
  log.info(f'[ov-seg] inference in {str(input_color_dir)}')

  templates, class_names = get_templates(classes)

  log.info('[ov-seg] loading model')
  model = load_ovseg(device=device, custom_templates=templates)

  log.info('[ov-seg] inference')

  for file in tqdm(input_files):
    result = process_image(model, file, class_names, flip=flip)
    cv2.imwrite(str(output_dir / f'{file.stem}.png'),
                result + 1)  # if 0 is background


def arg_parser():
  parser = argparse.ArgumentParser(description='OVSeg Segmentation')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help=
      'Path to workspace directory. There should be a "color" folder inside.',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/wordnet_ovseg_1',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version',
  )
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == '__main__':
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  run(scene_dir=args.workspace, output_folder=args.output)
