import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Union

import cv2
import gin
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


import sys
import os

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录
parent_dir = os.path.dirname(current_dir)
# 获取父目录的父目录
grandparent_dir = os.path.dirname(parent_dir)

# 将父目录的父目录添加到 sys.path
sys.path.append(grandparent_dir)

from labelmaker.label_data import get_wordnet
from labelmaker.visualisation import random_color

logging.basicConfig(level="INFO")
log = logging.getLogger(' sdfstudio data preprocessing ')


@gin.configurable
def sdfstudio_preprocessing(
    scene_dir: Union[str, Path],
    mono_depth_folder: Union[str, Path],
    mono_normal_folder: Union[str, Path],
    label_folder: Union[str, Path],
    label_aux_folder: Union[str, Path],
    instance_folder: Union[str, Path],
    output_folder: Union[str, Path],
    semantic_info: List[Dict[str, str]],
    sampling: int = 1,
    depth_scale: float = 1000.0,
    force: bool = True,
    train_width: int = -1,
    train_height: int = -1,
):
  """
    preprocessing color, depth, semantic, mono_depth, mono_normal, pose, intrinsic before it is fed into NeuS-acc
    The scene is rescaled so that its pose is fit into a [-1, +1] cube.

    train_width, train_height, sampling are used to rescale all input, to prevent cpu memory OOM.
  """
  scene_dir = Path(scene_dir)
  mono_depth_folder = Path(mono_depth_folder)
  mono_normal_folder = Path(mono_normal_folder)
  label_folder = Path(label_folder)
  label_aux_folder = Path(label_aux_folder)
  instance_folder = Path(instance_folder)
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
  
  input_label_aux_dir = scene_dir / label_aux_folder  # .png files
  assert input_label_aux_dir.exists() and input_label_aux_dir.is_dir()  
  aux_label_keys = set(x.stem for x in input_label_aux_dir.glob('*aux.png'))
  
  input_instance_dir = scene_dir / instance_folder  # .png files
  assert input_instance_dir.exists() and input_instance_dir.is_dir()  
  instance_keys = set(x.stem for x in input_instance_dir.glob('*.png'))        


  # test if all file names are identical
  #   the associated instance masks were subsampled already, so don't check the instance_keys
  assert color_keys == depth_keys
  assert color_keys == pose_keys
  assert color_keys == intrinsic_keys
  assert color_keys == mono_depth_keys
  assert color_keys == mono_normal_keys  
  assert color_keys == label_keys
#   assert color_keys == instance_keys

  # frames to be used
  keys = list(color_keys)
  keys.sort()
  keys = keys[::sampling]

  # save paths
  pose_paths = [input_pose_dir / f"{k}.txt" for k in keys]
  color_paths = [input_color_dir / f"{k}.jpg" for k in keys]
  depth_paths = [input_depth_dir / f"{k}.png" for k in keys]
  label_paths = [input_label_dir / f"{k}.png" for k in keys]
  aux_label_paths = [input_label_aux_dir / f"{k}_aux.png" for k in keys]
  instance_paths = [input_instance_dir / f"{k}.png" for k in keys]
  intrinsic_paths = [input_intrinsic_dir / f"{k}.txt" for k in keys]
  mono_depth_paths = [input_mono_depth_dir / f"{k}.png" for k in keys]
  mono_normal_paths = [input_mono_normal_dir / f"{k}.npy" for k in keys]

  # record original data information
  original_color_dim = Image.open(color_paths[0]).size
  log.info(f" original image dim: {original_color_dim}")

  # resize
  if train_width > 0 and train_height > 0:
    log.info(
        f" All input will be resized during training: {(train_width, train_height)}"
    )
  else:
    train_width, train_height = original_color_dim

  # transform function
  image_size = (train_height, train_width)
  color_trans_totensor = transforms.Compose([
      transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
  ])
  depth_trans_totensor = transforms.Compose([
      transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
  ])
  label_trans_totensor = transforms.Compose([
      transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
  ])
  instance_trans_totensor = transforms.Compose([
      transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
  ])
  mono_depth_trans_totensor = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
  ])
  mono_normal_trans_totensor = transforms.Compose([
      transforms.ToTensor(),
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

  W, H = original_color_dim

  # adjust intrinsic matrix due to resizing
  intrinsics[:, 0, :] *= train_width / W
  intrinsics[:, 1, :] *= train_height / H

  # semantic information
  class_hist = np.zeros(max(x['id'] for x in semantic_info) + 1, dtype=np.int64)

  # create output folder
  output_dir = scene_dir / output_folder
  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)
  
  num_instances = 0

  # preprocessing and copying files
  frames = []
  for idx, (
      valid,
      pose,
      intrinsic,
      color_path,
      depth_path,
      label_path,
      aux_label_path,
      instance_path,
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
          aux_label_paths,
          instance_paths,
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
    target_label_aux_path = output_dir / f"{idx:06d}_label_aux.png"
    target_instance_path = output_dir / f"{idx:06d}_instance.png"
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
    
    # process auxiliary label
    aux_label = Image.open(str(aux_label_path))
    aux_label = np.asarray(aux_label).copy()
    # aux_label = cv2.imread(str(aux_label_path))
    #remove unknown ids
    aux_label[:,:,0][aux_label[:,:,0] > class_hist.shape[0] - 1] = 0
    aux_label = cv2.resize(aux_label, (train_width, train_height), interpolation=cv2.INTER_NEAREST)   
    success = cv2.imwrite(str(target_label_aux_path), aux_label)
    # aux_label_tensor = label_trans_totensor(Image.fromarray(aux_label))
    # aux_label_tensor.save(str(target_label_aux_path))

    # update semantic class histogram
    class_hist += np.bincount(
        np.asarray(label_tensor).flatten(),
        minlength=class_hist.shape[0],
    )
    class_hist += np.bincount(
        aux_label[:,:,0].flatten(),
        minlength=class_hist.shape[0],
    )
    
    # process instance labels
    instance = Image.open(str(instance_path))
    instance = np.asarray(instance).copy()
    num_instances = max(num_instances, int(instance.max())+1)
    instance = Image.fromarray(instance)
    instance_tensor = instance_trans_totensor(instance)
    instance_tensor.save(str(target_instance_path))
    # process mono depth
    mono_depth = cv2.imread(str(mono_depth_path), -1).astype(np.float32)
    monodepth_tensor = mono_depth_trans_totensor(mono_depth).squeeze()
    # this depth is scaled, we need to unscale it
    monodepth_tensor = monodepth_tensor - monodepth_tensor.min()
    monodepth_tensor = monodepth_tensor / (monodepth_tensor.max() + 1e-8)
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
        "aux_label_path":
            str(target_label_aux_path.relative_to(output_dir)),
        "instance_path":
            str(target_instance_path.relative_to(output_dir)),
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
      "height": train_height,
      "width": train_width,
      "has_mono_prior": True,
      "has_sensor_depth": True,
      "has_semantics": True,
      "has_auxiliary_semantics": True,
      "has_instances": True,
      "pairs": None,
      "num_instances": num_instances,
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
  )  # (n, 3, 3), this is the original intrinsic, not changed

  # load render pose
  render_poses = np.stack(
      [np.loadtxt(path) for path in render_pose_paths],
      axis=0,
  )  # (n, 4, 4)
  render_poses[:, :3, 3] -= center
  render_poses[:, :3, 3] *= scale

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
      '--instance_folder',
      type=str,
      default='sam_associated',
      help='Name of instance folder in the workspace directory',
  ) 
  parser.add_argument(
      '--label_aux_folder',
      type=str,
      default='intermediate/consensus_aux',
      help='Name of label folder in the workspace directory',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/sdfstudio_preprocessing',
      help=
      'Name of output folder in the workspace directory, to store data used in sdfstudio training and rendering',
  )
  parser.add_argument('--sampling', type=int, default=1)
  parser.add_argument('--train_width', type=int, default=-1)
  parser.add_argument('--train_height', type=int, default=-1)
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
      label_aux_folder=args.label_aux_folder,
      instance_folder=args.instance_folder,
      output_folder=args.output,
      sampling=args.sampling,
      semantic_info=get_wordnet(),  # use this as default
      train_width=args.train_width,
      train_height=args.train_height,
  )
