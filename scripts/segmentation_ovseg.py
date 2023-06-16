import sys, os

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'ov-seg'))
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo

from nltk.corpus import wordnet as wn

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from segmentation_tools.label_data import get_ade150, get_wordnet, get_replica
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import numpy as np
import shutil

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


def load_ovseg(custom_templates=None):
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
            Path(__file__).parent / '..' / '3rdparty' /
            'ovseg_swinbase_vitL14_ft_mpt.pth')
    ])
    if custom_templates is not None:
        cfg.MODEL.CLIP_ADAPTER.TEXT_TEMPLATES = "predefined"
        cfg.MODEL.CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = custom_templates
    cfg.freeze()
    demo = VisualizationDemo(cfg)
    return demo


def process_image(model, img_path, class_names, threshold=0.7, flip=False):
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


def ovseg_inference(scene_dir,
                    keys,
                    flip=False,
                    classes='wordnet',
                    img_template='color/{k}.jpg'):
    log.info(f'[ov-seg] using {classes} classes')
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
                        t.format(size=s,
                                 noun="{noun}",
                                 definition="{definition}")))
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
                templates.append(
                    WordnetPromptTemplate(t.format(size=s, noun="{noun}")))
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

    log.info('[ov-seg] loading model')
    if templates is not None:
        model = load_ovseg(custom_templates=templates)
    else:
        model = load_ovseg()

    log.info('[ov-seg] inference')
    scene_dir = Path(scene_dir)
    assert scene_dir.exists() and scene_dir.is_dir()
    if flip:
        results_dir = scene_dir / f'pred_ovseg_{classes}_flip'
    else:
        results_dir = scene_dir / f'pred_ovseg_{classes}'
    shutil.rmtree(results_dir, ignore_errors=True)
    results_dir.mkdir(exist_ok=False)
    for k in tqdm(keys):
        img = str(scene_dir / img_template.format(k=k))
        result = process_image(model, img, class_names, flip=flip)
        cv2.imwrite(str(results_dir / f'{k}.png'),
                    result + 1)  # if 0 is background


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--replica', default=False)
    parser.add_argument('--classes', default='ade150')
    parser.add_argument('--flip', default=False)
    flags = parser.parse_args()
    scene_dir = Path(flags.scene)
    assert scene_dir.exists() and scene_dir.is_dir()
    if flags.replica:
        keys = sorted(
            int(x.name.split('.')[0].split('_')[1])
            for x in (scene_dir / 'rgb').iterdir())
        img_template = 'rgb/rgb_{k}.png'
    else:
        keys = sorted(
            int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
        img_template = 'color/{k}.png'
    log.info('running inference')
    ovseg_inference(scene_dir,
                    keys,
                    flip=flags.flip,
                    classes=flags.classes,
                    img_template=img_template)
