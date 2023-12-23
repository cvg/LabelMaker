import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Union

import cv2
import ffmpeg
import gin
import imgviz
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from labelmaker.consensus import VALID_LABEL_SPACES
from labelmaker.label_data import get_ade150, get_nyu40, get_replica, get_scannet200, get_scannet_all, get_wordnet

logging.basicConfig(level="INFO")
log = logging.getLogger('Label Visualization and Vieo Rendering')


def visualize_image(
    rgb: np.ndarray,
    label: np.ndarray,
    color_mapping: np.ndarray,
    label_mapping: List[str],
    alpha: float = 0.6,
    resize: float = 2.0,
    font_size: int = 25,
):
  assert rgb.shape[:2] == label.shape[:2]
  H, W = rgb.shape[:2]

  new_H = int(H * resize)
  new_W = int(W * resize)

  new_rgb = cv2.resize(rgb, (new_W, new_H))
  # should not use cv2.resize as it introduce interpolation...
  new_label = np.asarray(
      Image.fromarray(label).resize((new_W, new_H), Image.NEAREST))

  labeled_viz_img = imgviz.label2rgb(
      image=new_rgb,
      label=new_label,
      colormap=color_mapping,
      label_names=label_mapping,
      font_size=font_size,
      loc="centroid",
      alpha=alpha,
  )
  return labeled_viz_img


@gin.configurable
def batch_visualize_image(
    label_space: str,
    scene_dir: Union[str, Path],
    rgb_folder: Union[str, Path],
    label_folder: Union[str, Path],
    temp_save_folder: Union[str, Path],
    rgb_glob_expression: str = '*.jpg',
    path_to_key_fn: Callable = lambda x: int(x.stem),
    label_file_template: str = '{k:06d}.png',
    force_delete_temp: bool = True,
    alpha: float = 0.6,
    resize: float = 1.0,
    font_size: int = 13,
    n_jobs=4,
):
  # convert str to Path object
  scene_dir = Path(scene_dir)
  rgb_folder = Path(rgb_folder)
  label_folder = Path(label_folder)
  temp_save_folder = Path(temp_save_folder)

  # convert folder to path
  rgb_dir = scene_dir / rgb_folder
  label_dir = scene_dir / label_folder
  temp_save_dir = scene_dir / temp_save_folder

  # check
  assert scene_dir.exists() and scene_dir.is_dir()
  assert rgb_dir.exists() and rgb_dir.is_dir()
  assert label_dir.exists() and label_dir.is_dir()
  if force_delete_temp:
    shutil.rmtree(temp_save_dir, ignore_errors=True)
    os.makedirs(str(temp_save_dir), exist_ok=False)
  else:
    if not temp_save_dir.exists():
      os.makedirs(str(temp_save_dir), exist_ok=False)
    else:
      assert temp_save_dir.is_dir(
      ), "The temporary saving directory is actually a file!"
      is_empty = not any(temp_save_dir.iterdir())
      assert is_empty, "The temporary saving directory is not empty!"

  # check if all rgb files have their corresponding labels
  rgb_files = rgb_dir.glob(rgb_glob_expression)
  rgb_files = sorted(rgb_files, key=path_to_key_fn)
  keys = [path_to_key_fn(path) for path in rgb_files]
  label_files = [label_dir / label_file_template.format(k=key) for key in keys]
  assert all([path.exists() and path.is_file() for path in label_files
             ]), "Not all rgb files has their label, check your folder!"

  # get label set
  get_label_info_fn = {
      "ade20k": get_ade150,
      "nyu40": get_nyu40,
      "scannet200": get_scannet200,
      'wordnet': get_wordnet,
      "scannet": get_scannet_all,
  }
  assert label_space in get_label_info_fn.keys()
  label_info = get_label_info_fn[label_space]()

  id2name, id2color = {0: "unknown"}, {0: [0, 0, 0]}
  for item in label_info:
    id2name[item['id']] = item['name'].split('.')[0]
    id2color[item['id']] = item['color']

  id_range = np.array(list(id2name.keys())).max() + 1

  label_names = [""] * id_range
  label_color = np.zeros(shape=(id_range, 3), dtype=np.uint8)

  for idx in id2name.keys():
    label_names[idx] = id2name[idx]
    label_color[idx] = id2color[idx]

  # now begin processing
  log.info('[visualization] All folders and files checked!')
  log.info('[visualization] labelling image starts!')

  def warpper_viz_and_save_img(
      key: int,
      rgb_path: Path,
      label_path: Path,
  ):
    rgb_img = np.asarray(Image.open(str(rgb_path)).convert("RGB"))
    label_img = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

    viz_img = visualize_image(
        rgb=rgb_img,
        label=label_img,
        color_mapping=label_color,
        label_mapping=label_names,
        alpha=alpha,
        resize=resize,
        font_size=font_size,
    )
    Image.fromarray(viz_img).save(str(temp_save_dir / '{:06d}.png'.format(key)))

    return 1

  Parallel(n_jobs=n_jobs)(delayed(warpper_viz_and_save_img)(
      key,
      rgb_path,
      label_path,
  ) for key, rgb_path, label_path in tqdm(
      zip(
          keys,
          rgb_files,
          label_files,
      ),
      total=len(keys),
  ))

  log.info('[visualization] All image successfully labeled!')


@gin.configurable
def viz2video(
    scene_dir: Union[str, Path],
    image_folder: Union[str, Path],
    output_name: Union[str, Path],
    fps: int = 30,
    image_glob_expression="*.png",
    image_name_to_key_fn=lambda x: int(x.stem),
    verbose: bool = False,
    delete_image_folder: bool = False,
):
  scene_dir = Path(scene_dir)
  image_folder = Path(image_folder)
  output_name = Path(output_name)

  image_dir = scene_dir / image_folder
  output_path = scene_dir / output_name

  # check
  assert scene_dir.exists() and scene_dir.is_dir()
  assert image_dir.exists() and image_dir.is_dir()
  if output_path.exists():
    if not output_path.is_file():
      assert False
    os.remove(str(output_path))

  image_files = image_dir.glob(image_glob_expression)
  image_files = sorted(image_files, key=image_name_to_key_fn)

  # create
  temp_path = image_dir / 'ffmpeg_video_render.txt'
  with open(str(temp_path), "w") as f:
    for name in image_files:
      f.write("file '" + str(name) + "'\n")

  if not output_path.parent.exists():
    os.makedirs(str(output_path.parent))

  log.info('[video render] Starting to render video!')
  ffmpeg.input(
      str(temp_path),
      r=str(fps),
      f="concat",
      safe="0",
  ).output(
      str(output_path),
      vcodec="libx265",
      loglevel="info" if verbose else "quiet",
  ).run()
  os.remove(str(temp_path))
  log.info('[video render] Video render complete!')

  if delete_image_folder:
    log.info(f'[video render] Deleting input image folder: {str(image_dir)}')
    shutil.rmtree(image_dir)
