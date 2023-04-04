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

from segmentation_tools.label_data import get_ade150, get_wordnet
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import numpy as np

logging.basicConfig(level="INFO")
log = logging.getLogger('OV-Seg Segmentation')


class WordnetPromptTemplate:

    def __init__(self, template):
        self.template = template

    def format(self, noun):
        synset = wn.synset(noun)
        prompt = self.template.format(noun=synset.name().split('.')[0],
                                    definition=synset.definition())
        if len(synset.lemma_names()) > 1:
            prompt += " It can also be called {}".format(", ".join(synset.lemma_names()[1:]))
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


def process_image(model, img_path, class_names, threshold=0.7):
    # use PIL, to be consistent with evaluation
    img = read_image(img_path, format="BGR")
    predictions = model.predictor(img, class_names)
    blank_area = (predictions['sem_seg'][0] == 0).to('cpu').numpy()
    product, pred = torch.max(predictions['sem_seg'], dim=0)
    pred[product < threshold] = len(class_names)
    pred = pred.to('cpu').numpy().astype(int)
    pred[blank_area] = -1
    # the last text feature is the background / zero feature
    pred[pred == len(class_names)] = -1
    return pred


def ovseg_inference(scene_dir, keys, classes='wordnet', img_template='color/{k}.jpg'):
    log.info(f'using {flags.classes} classes')
    if classes == 'ade150':
        class_names = [x['name'] for x in get_ade150()]
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
                        t.format(size=s, noun="{noun}",
                                definition="{definition}")))
        class_names = [x['name'] for x in get_wordnet()]
    else:
        raise ValueError(f'Unknown class set {flags.classes}')

    log.info('loading model')
    if classes == 'wordnet':
        model = load_ovseg(custom_templates=templates)
    else:
        model = load_ovseg()

    scene_dir = Path(scene_dir)
    assert scene_dir.exists() and scene_dir.is_dir()
    (scene_dir / f'pred_ovseg_{flags.classes}').mkdir(exist_ok=True)
    for k in tqdm(keys):
        img = str(scene_dir / img_template.format(k=k))
        result = process_image(model, img, class_names)
        cv2.imwrite(str(scene_dir / f'pred_ovseg_{flags.classes}' / f'{k}.png'),
                    result + 1)  # png cannot save -1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--replica', default=False)
    parser.add_argument('--classes', default='ade150')
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
    ovseg_inference(scene_dir, keys, classes=flags.classes, img_template=img_template)
