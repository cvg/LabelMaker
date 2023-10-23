import argparse
import logging
import os
import shutil
import sys
from os.path import abspath, dirname, join
from pathlib import Path
from typing import List

import clip
import cv2
import gin
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
import torchvision.transforms as TS
from groundingdino.models import build_model
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from PIL import Image
from ram import get_transform
from ram.models import ram as get_ram
from ram.models.ram import RAM
from ram.utils.openset_utils import article, multiple_templates, processed_name
from segment_anything import SamPredictor, build_sam_hq
from skimage.measure import label as count_components
from skimage.morphology import binary_dilation
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(abspath(join(dirname(__file__), '..')))
from labelmaker.label_data import get_wordnet

logging.basicConfig(level="INFO")
log = logging.getLogger('Grounded SAM Segmentation')


def build_openset_label_embedding(
    categories,
    device: str = 'cpu',
):
  """
  modifiied from ram.utils.build_openset_label_embedding for better device and download control
  """
  model, _ = clip.load(
      "ViT-B/16",
      device=device,
  )
  templates = multiple_templates

  with torch.no_grad():
    openset_label_embedding = []
    for category in categories:
      texts = [
          template.format(processed_name(category, rm_dot=True),
                          article=article(category)) for template in templates
      ]
      texts = [
          "This is " +
          text if text.startswith("a") or text.startswith("the") else text
          for text in texts
      ]
      texts = clip.tokenize(texts).to(device=device)  # tokenize

      text_embeddings = model.encode_text(texts)
      text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
      text_embedding = text_embeddings.mean(dim=0)
      text_embedding /= text_embedding.norm()
      openset_label_embedding.append(text_embedding)

    openset_label_embedding = torch.stack(openset_label_embedding, dim=1)

  openset_label_embedding = openset_label_embedding.t()
  return openset_label_embedding


def load_grounding_dino(
    grounding_dino_config_path: str,
    grounding_dino_checkpoint_path: str,
    device: torch.device,
) -> GroundingDINO:
  args = SLConfig.fromfile(grounding_dino_config_path)
  args.device = device
  model = build_model(args)
  checkpoint = torch.load(grounding_dino_checkpoint_path)
  model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
  model = model.eval().to(device)
  return model


def load_grounded_sam(
    ram_ckpt: str,
    groundingdino_ckpt: str,
    sam_hq_ckpt: str,
    device: str,
):
  """
  grounded sam contains three model: RAM, Grounding DINO, SAM-HQ
  """
  # get tags
  wordnet = get_wordnet()
  wordnet.sort(key=lambda x: x.get('id'))

  categories = [
      item["name"].split(".")[0].replace("_", " ").strip()
      for item in wordnet[1:]  # skip the "uknown" class
  ]

  # clip is used here
  category_embedding = build_openset_label_embedding(categories, device=device)

  # load ram
  ram = get_ram(
      pretrained=ram_ckpt,
      image_size=384,
      vit="swin_l",
  ).to(device).eval()

  # change ram's closed set to wordnet
  ram.tag_list = np.array(categories)
  ram.label_embed = nn.Parameter(category_embedding.float().to(device))
  ram.num_class = len(categories)
  ram.class_threshold = torch.ones(ram.num_class) * 0.5
  ram.eval().to(device)

  ram_transform = get_transform(image_size=384)

  # load grounding dino
  grounding_dino_config_path = str(
      abspath(
          join(__file__, '../..', '3rdparty', "Grounded-Segment-Anything",
               "GroundingDINO", "groundingdino", "config",
               "GroundingDINO_SwinT_OGC.py")))
  grounding_dino = load_grounding_dino(
      grounding_dino_config_path=grounding_dino_config_path,
      grounding_dino_checkpoint_path=groundingdino_ckpt,
      device=device,
  )
  grounding_dino_transform = T.Compose([
      T.RandomResize([800], max_size=1333),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  # load sam
  sam = build_sam_hq(checkpoint=sam_hq_ckpt).to(device).eval()
  sam_predictor = SamPredictor(sam)

  sam_transform = TS.PILToTensor()

  return (
      ram,
      ram_transform,
      grounding_dino,
      grounding_dino_transform,
      sam_predictor,
      sam_transform,
  )


def get_ram_output(
    model: RAM,
    image,
):
  """
  Modified from ram.models.ram.RAM.generate_tag_openset. This function gives string output, I have to convert it back to label ids. This is unnecessary.
  """
  label_embed = torch.nn.functional.relu(model.wordvec_proj(model.label_embed))

  image_embeds = model.image_proj(model.visual_encoder(image))
  image_atts = torch.ones(image_embeds.size()[:-1],
                          dtype=torch.long).to(image.device)

  # recognized image tags using image-tag recogntiion decoder
  image_spatial_embeds = image_embeds[:, 1:, :]

  bs = image_spatial_embeds.shape[0]
  label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
  tagging_embed = model.tagging_head(
      encoder_embeds=label_embed,
      encoder_hidden_states=image_embeds,
      encoder_attention_mask=image_atts,
      return_dict=False,
      mode='tagging',
  )

  logits = model.fc(tagging_embed[0]).squeeze(-1)

  targets = torch.where(
      torch.sigmoid(logits) > model.class_threshold.to(image.device),
      torch.tensor(1.0).to(image.device),
      torch.zeros(model.num_class).to(image.device))

  tag = targets.cpu().numpy()
  tag[:, model.delete_tag_index] = 0
  indices = np.argwhere(tag == 1)[:, 1]

  return indices


def tags_to_caption(tags: List[str]) -> str:
  """
  Given a set of nouns, convert it into Grounding DINO compatible captions.
  """
  caption = ", ".join(tags)
  caption = caption.lower()
  caption = caption.strip()
  if not caption.endswith("."):
    caption = caption + "."

  return caption


def get_phrases_id_from_logit(
    logit: torch.Tensor,
    tags: List[str],
    tokenizer: AutoTokenizer,
    text_threshold: float = 0.2,
):
  """
    There is a one to many mapping from a box to possible tag's tokenzier, therefore, the original method is not guaranteed to output a single category. Here we match a bounding box with the most probable tag.
    Input:
        logit: N_box x max_token_length
    """

  # 101 [CLS]
  # 102 [SEP]
  # 1010 ','
  # 1012 '.'

  tokenized = tokenizer(tags_to_caption(tags))
  token_ids = np.array(tokenized['input_ids'])

  token_splits = np.split(np.arange(token_ids.shape[0]),
                          np.argwhere(token_ids == 1010).reshape(-1))

  effective_logit = logit[:, :token_ids.shape[0]]

  logit_mask = (effective_logit > text_threshold) * torch.from_numpy(
      ~np.isin(token_ids, [101, 102, 1010, 1012])).to(logit.device).reshape(
          1, -1)

  effective_logit = effective_logit * logit_mask

  tag_mask = torch.stack(
      [logit_mask[:, single_split].any(dim=1) for single_split in token_splits],
      dim=1,
  )

  tag_logit = torch.stack(
      [
          effective_logit[:, single_split].max(dim=1)[0]
          for single_split in token_splits
      ],
      dim=1,
  )

  tag_logit[~tag_mask] = -torch.inf

  pred_tag_idx = tag_logit.argmax(dim=1)
  pred_tag_mask = ~torch.isinf(tag_logit.max(dim=1)[0])
  # pred_tag = [tags[i] for i in pred_tag_idx]
  pred_score = tag_logit.max(dim=1)[0]

  return pred_tag_idx, pred_tag_mask, pred_score


def get_grounding_output(
    grounding_dino_model,
    image,
    tags,
    box_threshold,
    text_threshold,
    device="cpu",
):
  """
  Get bounding box, not the final output.
  """
  caption = tags_to_caption(tags=tags)
  grounding_dino_model = grounding_dino_model.to(device)
  image = image.to(device)
  with torch.no_grad():
    outputs = grounding_dino_model(image[None], captions=[caption])
  logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
  boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

  # filter output by logits
  logits_filt = logits.clone()
  boxes_filt = boxes.clone()
  filt_mask = logits_filt.max(dim=1)[0] > box_threshold
  logits_filt = logits_filt[filt_mask]  # num_filt, 256
  boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

  # tag prediction
  pred_tag_idx, valid_tag_filter, score = get_phrases_id_from_logit(
      logit=logits_filt,
      tags=tags,
      tokenizer=grounding_dino_model.tokenizer,
      text_threshold=text_threshold,
  )

  ## filter out failure decoding
  score = score[valid_tag_filter]
  boxes_filt = boxes_filt[valid_tag_filter]
  pred_tag_idx = pred_tag_idx[valid_tag_filter]

  return boxes_filt, score, pred_tag_idx


@torch.no_grad()
def process_image(
    ram: RAM,
    ram_transform,
    grounding_dino: GroundingDINO,
    grounding_dino_transform,
    sam_predictor: SamPredictor,
    sam_transform,
    img_path: str,
    device: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.2,
    iou_threshold: float = 0.5,
    sam_defect_threshold: int = 60,
    flip=False,
    debug: bool = False,
):

  img = Image.open(img_path).convert("RGB")

  # returns the semantic id (shifted id=0 is wall)
  ram_results = get_ram_output(
      image=ram_transform(img).unsqueeze(0).to(device),
      model=ram,
  )
  tags = ram.tag_list[ram_results].tolist()

  boxes_filt, scores, relative_label = get_grounding_output(
      grounding_dino_model=grounding_dino,
      image=grounding_dino_transform(img, None)[0],
      tags=tags,
      box_threshold=box_threshold,
      text_threshold=text_threshold,
      device=device,
  )
  # shifted semantic label
  labels = ram_results[relative_label].reshape(-1)

  # convert relative to absolute box
  size = img.size
  H, W = size[1], size[0]
  for i in range(boxes_filt.size(0)):
    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
    boxes_filt[i][2:] += boxes_filt[i][:2]

  # nms filtering
  nms_filter = torchvision.ops.nms(boxes_filt, scores,
                                   iou_threshold).numpy().tolist()
  boxes_filt = boxes_filt[nms_filter]
  scores_filt = scores[nms_filter]
  label_filt = labels[nms_filter]

  # forward sam
  image_array_sam = sam_transform(img).movedim(0, -1).numpy()

  transformed_boxes = sam_predictor.transform.apply_boxes_torch(
      boxes_filt, image_array_sam.shape[:2]).to(device)

  sam_predictor.set_image(image_array_sam)
  masks, _, _ = sam_predictor.predict_torch(
      point_coords=None,
      point_labels=None,
      boxes=transformed_boxes.to(device),
      multimask_output=False,
  )
  masks = masks.squeeze(1)  # n_mask, H, W

  # filter out defect segmentation
  # this gives the connected component of each masts,
  # if there are too many
  # this is probably a defect mask
  num_components = np.array([
      count_components(binary_dilation(mask.cpu().numpy()), return_num=True)[1]
      for mask in masks
  ])
  sam_defect_filter = num_components < sam_defect_threshold

  boxes_filt = boxes_filt[sam_defect_filter]
  scores_filt = scores_filt[sam_defect_filter]
  label_filt = label_filt[sam_defect_filter]
  masks_filt = masks[sam_defect_filter]

  if masks_filt.shape[0] == 0:
    # if there is no masks
    semantic_label = np.zeros(shape=(H, W), dtype=np.int64)

  else:
    # resolve intersection conflict
    # assigning intersections area to the instance with smallest area
    masks_area = masks_filt.count_nonzero(dim=[1, 2])

    # taking argmin onto this tensor returns the unshifted instance label
    temp = torch.cat(
        [
            (masks_area.max() + 1) * torch.ones(
                size=[1] + list(masks_filt.shape[1:]),
                device=masks_filt.device,
                dtype=masks_area.dtype,
            ),
            (masks_area.reshape(-1, 1, 1) * masks_filt +
             (masks_area.max() + 1) * (~masks_filt)),
        ],
        dim=0,
    )
    instance_id = temp.argmin(dim=0)  # H, W
    semantic_label = np.concatenate([[0], label_filt + 1])[
        instance_id.cpu().numpy(),
    ]  # H, W

    if flip:
      semantic_label = semantic_label[:, ::-1]

  if not debug:
    return semantic_label
  else:
    return semantic_label, {
        'num_components': num_components,
    }


@gin.configurable
def run(
    input_dir: Path,
    output_dir: Path,
    device: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.2,
    iou_threshold: float = 0.5,
    sam_defect_threshold: int = 30,
    flip=False,
):

  # check if output directory exists
  shutil.rmtree(output_dir, ignore_errors=True)
  output_dir = output_dir + '_flip' if flip else output_dir
  # makedirs instead of mkdir
  os.makedirs(str(output_dir), exist_ok=False)

  input_files = input_dir.glob('*')
  input_files = sorted(input_files, key=lambda x: int(x.stem.split('.')[0]))

  log.info(f'[Grounded SAM] inference in {str(input_dir)}')

  log.info('[Grounded SAM] loading model')
  ram_ckpt = abspath(
      join(__file__, '../..', 'checkpoints', 'ram_swin_large_14m.pth'))
  groundingdino_ckpt = abspath(
      join(__file__, '../..', 'checkpoints', 'groundingdino_swint_ogc.pth'))
  sam_hq_ckpt = abspath(join(__file__, '../..', 'checkpoints', 'sam_hq_vit_h.pth'))
  (
      ram,
      ram_transform,
      grounding_dino,
      grounding_dino_transform,
      sam_predictor,
      sam_transform,
  ) = load_grounded_sam(
      ram_ckpt=ram_ckpt,
      groundingdino_ckpt=groundingdino_ckpt,
      sam_hq_ckpt=sam_hq_ckpt,
      device=device,
  )
  log.info('[Grounded SAM] model loaded!')

  for file in tqdm(input_files):
    result = process_image(
        ram=ram,
        ram_transform=ram_transform,
        grounding_dino=grounding_dino,
        grounding_dino_transform=grounding_dino_transform,
        sam_predictor=sam_predictor,
        sam_transform=sam_transform,
        img_path=file,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        iou_threshold=iou_threshold,
        sam_defect_threshold=sam_defect_threshold,
        flip=flip,
    )
    cv2.imwrite(str(output_dir / f'{file.stem}.png'), result)


def arg_parser():
  parser = argparse.ArgumentParser(
      description='Grounded SAM Semantic Segmentation')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help='Path to workspace directory',
  )
  parser.add_argument(
      '--input',
      type=str,
      default='color',
      help='Name of input directory in the workspace directory',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/wordnet_groundedsam_1',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version'
  )
  parser.add_argument('--config', help='Name of config file')

  return parser.parse_args()


def main(args):

  # check if workspace exists
  workspace = Path(args.workspace)
  assert workspace.exists() and workspace.is_dir()

  # check if input directory exists
  input_dir = workspace / args.input
  assert input_dir.exists() and input_dir.is_dir()

  output_dir = workspace / args.output

  gin.parse_config_file(args.config)
  run(input_dir=input_dir, output_dir=output_dir)


if __name__ == '__main__':
  print(os.path.abspath(os.path.join(__file__, '../..', '3rdparty')))
  args = arg_parser()
  main(args)
