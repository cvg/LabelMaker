import argparse
import glob
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Union

import cv2
import gin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from torchvision import transforms
from tqdm import tqdm

from labelmaker.label_data import get_replica, get_scannet_all, get_wordnet
from labelmaker.visualisation import random_color

logging.basicConfig(level="INFO")
log = logging.getLogger(' sdfstudio data preprocessing ')


@gin.configurable
def sdfstudio_preprocessing(
    scene_dir: Union[str, Path],
    mono_depth_folder: Union[str, Path],
    mono_normal_folder: Union[str, Path],
    label_folder: Union[str, Path],
    output_folder: Union[str, Path],
    semantic_info: List[Dict[str, str]],
    image_size: int = 384,
    sampling: int = 1,
    depth_scale: float = 1000.0,
    force: bool = False,
):
  """
    preprocessing color, depth, semantic, mono_depth, mono_normal, pose, intrinsic before it is fed into NeuS-acc
    The scene is rescaled so that its pose is fit into a [-1, +1] cube.
  """
  scene_dir = Path(scene_dir)
  mono_depth_folder = Path(mono_depth_folder)
  mono_normal_folder = Path(mono_normal_folder)
  label_folder = Path(label_folder)
  output_folder = Path(output_folder)

  
  # check if output fodler exists and if same number of files as in color
  if not force and (scene_dir / output_folder).exists():
    if len(list((scene_dir / output_folder).glob('*_rgb.png'))) > 0:
        log.info(f" {output_folder} already exists, skipping")
        return

  # check if directories exists
  assert scene_dir.exists() and scene_dir.is_dir()

  input_color_dir = scene_dir / 'color'  # .jpg files
  assert input_color_dir.exists() and input_color_dir.is_dir()
  color_keys = set(x.stem for x in input_color_dir.glob('*.jpg'))

  input_depth_dir = scene_dir / 'depth'  # .png files
  assert input_depth_dir.exists() and input_depth_dir.is_dir()
  depth_keys = set(x.stem for x in input_depth_dir.glob('*.png'))

  input_pose_dir = scene_dir / 'pose'  # .txt files
  assert input_pose_dir.exists() and input_pose_dir.is_dir()
  pose_keys = set(x.stem for x in input_pose_dir.glob('*.txt'))

  input_intrinsic_dir = scene_dir / 'intrinsic'  # .txt files
  assert input_intrinsic_dir.exists() and input_intrinsic_dir.is_dir()
  intrinsic_keys = set(x.stem for x in input_intrinsic_dir.glob('*.txt'))

  input_mono_depth_dir = scene_dir / mono_depth_folder  # .png files
  assert input_mono_depth_dir.exists() and input_mono_depth_dir.is_dir()
  mono_depth_keys = set(x.stem for x in input_mono_depth_dir.glob('*.png'))

  input_mono_normal_dir = scene_dir / mono_normal_folder  # .npy files
  assert input_mono_normal_dir.exists() and input_mono_normal_dir.is_dir()
  mono_normal_keys = set(x.stem for x in input_mono_normal_dir.glob('*.npy'))

  input_label_dir = scene_dir / label_folder  # .png files
  assert input_label_dir.exists() and input_label_dir.is_dir()
  label_keys = set(x.stem for x in input_label_dir.glob('*.png'))


  # test if all file names are identical
  assert color_keys == depth_keys
  assert color_keys == pose_keys
  assert color_keys == intrinsic_keys
  assert color_keys == mono_depth_keys
  assert color_keys == mono_normal_keys
  assert color_keys == label_keys

  # frames to be used
  keys = list(color_keys)
  keys.sort()
  keys = keys[::sampling]

  # save paths
  pose_paths = [input_pose_dir / f"{k}.txt" for k in keys]
  color_paths = [input_color_dir / f"{k}.jpg" for k in keys]
  depth_paths = [input_depth_dir / f"{k}.png" for k in keys]
  label_paths = [input_label_dir / f"{k}.png" for k in keys]
  intrinsic_paths = [input_intrinsic_dir / f"{k}.txt" for k in keys]
  mono_depth_paths = [input_mono_depth_dir / f"{k}.png" for k in keys]
  mono_normal_paths = [input_mono_normal_dir / f"{k}.npy" for k in keys]

  # record original data information
  original_color_dim = Image.open(color_paths[0]).size
  original_depth_dim = Image.open(depth_paths[0]).size
  original_label_dim = Image.open(label_paths[0]).size
  original_mono_depth_dim = Image.open(mono_depth_paths[0]).size
  original_mono_normal_dim = np.load(mono_normal_paths[0]).shape[:2]

  # crop size
  color_crop_size = min(original_color_dim)
  depth_crop_size = min(original_depth_dim)
  label_crop_size = min(original_label_dim)
  mono_depth_crop_size = min(original_mono_depth_dim)
  mono_normal_crop_size = min(original_mono_normal_dim)

  log.info(f" original image dim: {original_color_dim}")

  # transform function
  color_trans_totensor = transforms.Compose([
      transforms.CenterCrop(color_crop_size),
      transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
  ])
  depth_trans_totensor = transforms.Compose([
      transforms.CenterCrop(depth_crop_size),
      transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
  ])
  label_trans_totensor = transforms.Compose([
      transforms.CenterCrop(label_crop_size),
      transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
  ])
  mono_depth_trans_totensor = transforms.Compose([
      transforms.ToTensor(),
      transforms.CenterCrop(mono_depth_crop_size),
      transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
  ])
  mono_normal_trans_totensor = transforms.Compose([
      transforms.ToTensor(),
      transforms.CenterCrop(mono_normal_crop_size),
      transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
  ])

  # load intrinsics
  intrinsics = np.stack(
      [np.loadtxt(path) for path in intrinsic_paths],
      axis=0,
  )  # (n, 3, 3)

  # load pose
  poses = np.stack(
      [np.loadtxt(path) for path in pose_paths],
      axis=0,
  )  # (n, 4, 4)

  # deal with invalid poses
  valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)

  # scaling of poses
  min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)  # (3,)
  max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)  # (3,)
  center = (min_vertices + max_vertices) / 2.0
  scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
  poses[:, :3, 3] -= center
  poses[:, :3, 3] *= scale
  log.info(
      f" scaling poses into a unit cube, original center :{center}, original scale: {scale}"
  )

  # inverse normalization
  scale_mat = np.eye(4).astype(np.float32)
  scale_mat[:3, 3] -= center
  scale_mat[:3] *= scale
  scale_mat = np.linalg.inv(scale_mat)

  # rescale intrinsics
  W, H = original_color_dim

  offset_x = (W - color_crop_size) * 0.5
  offset_y = (H - color_crop_size) * 0.5
  resize_factor = image_size / color_crop_size

  intrinsics[:, 0, 2] -= offset_x
  intrinsics[:, 1, 2] -= offset_y
  intrinsics[:, :2, :] *= resize_factor

  # semantic information
  class_hist = np.zeros(max(x['id'] for x in semantic_info) + 1, dtype=np.int64)

  # create output folder
  output_dir = scene_dir / output_folder
  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)

  # preprocessing and copying files
  frames = []
  for idx, (
      valid,
      pose,
      intrinsic,
      color_path,
      depth_path,
      label_path,
      mono_depth_path,
      mono_normal_path,
      k,
  ) in tqdm(enumerate(
      zip(
          valid_poses,
          poses,
          intrinsics,
          color_paths,
          depth_paths,
          label_paths,
          mono_depth_paths,
          mono_normal_paths,
          keys,
      )),
            total=len(keys)):

    if not valid:
      continue

    # saving paths, use png instead of jpg
    target_pose_path = output_dir / f"{idx:06d}_camtoworld.txt"
    target_color_path = output_dir / f"{idx:06d}_rgb.png"
    target_label_path = output_dir / f"{idx:06d}_label.png"
    target_depth_path = output_dir / f"{idx:06d}_sensor_depth.npy"
    target_depth_vis_path = output_dir / f"{idx:06d}_sensor_depth.png"  # for visualization
    target_mono_depth_path = output_dir / f"{idx:06d}_depth.npy"
    target_mono_normal_path = output_dir / f"{idx:06d}_normal.npy"

    # process color
    color = Image.open(str(color_path))
    color_tensor = color_trans_totensor(color)
    color_tensor.save(str(target_color_path))

    # process sensor depth
    depth = cv2.imread(str(depth_path), -1).astype(np.float32) / depth_scale
    depth_PIL = Image.fromarray(depth)
    new_depth = depth_trans_totensor(depth_PIL)
    new_depth = np.copy(np.asarray(new_depth))
    
    # scale depth as we normalize the scene to unit box
    new_depth *= scale
    plt.imsave(str(target_depth_vis_path), new_depth, cmap="viridis")
    np.save(str(target_depth_path), new_depth)

    # process pose
    np.savetxt(str(target_pose_path), pose)

    # process label
    label = Image.open(str(label_path))
    label = np.asarray(label).copy()
    #remove unknown ids
    label[label > class_hist.shape[0] - 1] = 0
    label = Image.fromarray(label)
    label_tensor = label_trans_totensor(label)
    label_tensor.save(str(target_label_path))

    # update semantic class histogram
    class_hist += np.bincount(
        np.asarray(label_tensor).flatten(),
        minlength=class_hist.shape[0],
    )

    # process mono depth
    mono_depth = cv2.imread(str(mono_depth_path), -1).astype(np.float32)
    monodepth_tensor = mono_depth_trans_totensor(mono_depth).squeeze()
    # this depth is scaled, we need to unscale it
    monodepth_tensor = monodepth_tensor - monodepth_tensor.min()
    monodepth_tensor = monodepth_tensor / monodepth_tensor.max()
    np.save(str(target_mono_depth_path), np.asarray(monodepth_tensor))

    # process mono normal
    mono_normal = np.load(str(mono_normal_path))
    mononormal_tensor = mono_normal_trans_totensor(mono_normal)
    np.save(str(target_mono_normal_path), np.asarray(mononormal_tensor))

    # saving, in format of relative path
    frame = {
        "rgb_path":
            str(target_color_path.relative_to(output_dir)),
        "camtoworld":
            pose.tolist(),
        "intrinsics":
            intrinsic.tolist(),
        "label_path":
            str(target_label_path.relative_to(output_dir)),
        "mono_depth_path":
            str(target_mono_depth_path.relative_to(output_dir)),
        "mono_normal_path":
            str(target_mono_normal_path.relative_to(output_dir)),
        "sensor_depth_path":
            str(target_depth_path.relative_to(output_dir)),
    }

    frames.append(frame)
    # Finished this loop!

    # scene bbox for the  scene
  scene_box = {
      "aabb": [[-1, -1, -1], [1, 1, 1]],
      "near": 0.05,
      "far": 2.5,
      "radius": 1.0,
      "collider_type": "box",
  }

  # meta data
  output_data = {
      "camera_model": "OPENCV",
      "height": image_size,
      "width": image_size,
      "has_mono_prior": True,
      "has_sensor_depth": True,
      "has_semantics": True,
      "pairs": None,
      "worldtogt": scale_mat.tolist(),
      "scene_box": scene_box,
  }

  # semantic info metadata
  for c in semantic_info:
    if 'color' not in c:
      c['color'] = random_color(rgb=True).tolist()
  output_data["semantic_classes"] = semantic_info

  # class histogram
  class_hist = class_hist.astype(np.float32)
  class_hist /= class_hist.sum()
  output_data["semantic_class_histogram"] = class_hist.tolist()

  # frames
  output_data["frames"] = frames

  # save meta data as json
  with open(output_dir / "meta_data.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)

  # another camera_path.json is used for rendering, it needs to be as dense as the original data
  render_keys = list(pose_keys)
  render_keys.sort()

  render_pose_paths = [input_pose_dir / f"{k}.txt" for k in render_keys]
  render_intrinsic_paths = [
      input_intrinsic_dir / f"{k}.txt" for k in render_keys
  ]

  # load render intrinsics
  render_intrinsics = np.stack(
      [np.loadtxt(path) for path in render_intrinsic_paths],
      axis=0,
  )  # (n, 3, 3)

  # load render pose
  render_poses = np.stack(
      [np.loadtxt(path) for path in render_pose_paths],
      axis=0,
  )  # (n, 4, 4)

  # record camera path
  camera_path = []
  for pose, intrinsic, k in tqdm(
      zip(render_poses, render_intrinsics, pose_keys),
      total=len(pose_keys),
  ):
    render_pose = pose.copy()
    render_pose[:3, 1:3] *= -1
    camera_path.append({
        "camera_to_world": render_pose.tolist(),
        "fx": intrinsic[0, 0],
        "fy": intrinsic[1, 1],
        "key": k,
    })

  # save camera path
  with open(output_dir / "camera_path.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "render_height": H,
            "render_width": W,
            "camera_path": camera_path,
            "seconds": 5.0,
        },
        f,
        indent=4,
    )


def arg_parser():
  parser = argparse.ArgumentParser(
      description=
      'Preprocess a scene in LabelMaker format into a sdfstudio dataset')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help=
      'Path to workspace directory. There should be a "color", "pose", "depth", "intrinsic" folder inside.',
  )
  parser.add_argument(
      '--mono_depth_folder',
      type=str,
      default='intermediate/depth_omnidata_1',
      help='Name of mono depth folder in the workspace directory',
  )
  parser.add_argument(
      '--mono_normal_folder',
      type=str,
      default='intermediate/normal_omnidata_1',
      help='Name of mono normal folder in the workspace directory',
  )
  parser.add_argument(
      '--label_folder',
      type=str,
      default='intermediate/consensus',
      help='Name of label folder in the workspace directory',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/sdfstudio_preprocessing',
      help=
      'Name of output folder in the workspace directory, to store data used in sdfstudio training and rendering',
  )
  parser.add_argument('--size', type=int, default=384)
  parser.add_argument('--sampling', type=int, default=1)
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == '__main__':
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  sdfstudio_preprocessing(
      scene_dir=args.workspace,
      mono_depth_folder=args.mono_depth_folder,
      mono_normal_folder=args.mono_normal_folder,
      label_folder=args.label_folder,
      output_folder=args.output,
      image_size=args.size,
      sampling=args.sampling,
      semantic_info=get_wordnet(),  # use this as default
  )
