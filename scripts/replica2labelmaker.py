import argparse
import json
import logging
import math
import os
import shutil
import sys
from os.path import abspath, dirname, exists, join

import cv2
import gin
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from tqdm import trange

sys.path.append(abspath(join(dirname(__file__), '..')))
from utils_3d import fuse_mesh


def get_intrinsics(H: int, W: int):
  # defined here: https://github.com/Harry-Zhi/semantic_nerf/blob/b79f9c3640b62350e9c167a66c273c2121428ce1/SSR/training/trainer.py#L55C1-L55C1
  # replica use a 90 degree fov camera

  hfov = 90

  fx = W / 2.0 / math.tan(math.radians(hfov / 2.0))
  fy = fx
  cx = (W - 1.0) / 2.0
  cy = (H - 1.0) / 2.0

  return np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


@gin.configurable
def process_replica(
    scan_dir: str,
    target_dir: str,
    sdf_trunc: float,
    voxel_length: float,
    depth_trunc: float,
):
  logger = logging.getLogger('Replica Process')
  logger.setLevel(logging.DEBUG)
  consoleHeader = logging.StreamHandler()
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  consoleHeader.setFormatter(formatter)
  logger.addHandler(consoleHeader)

  logger.info(
      "Processing Replica scan to LabelMaker format, from {} to {}...".format(
          scan_dir, target_dir))

  color_dir = join(scan_dir, 'rgb')
  depth_dir = join(scan_dir, 'depth')
  label_dir = join(scan_dir, 'semantic_class')
  pose_file = join(scan_dir, 'traj_w_c.txt')

  assert exists(color_dir), "rgb attribute not downloaded!"
  assert exists(depth_dir), "depth attribute not downloaded!"
  assert exists(label_dir), "semantic_class attribute not downloaded!"
  assert exists(pose_file), "traj_w_c.txt attribute not downloaded!"

  color_file_list = os.listdir(color_dir)
  depth_file_list = os.listdir(depth_dir)
  label_file_list = os.listdir(label_dir)

  # get file name idx
  color_idx, color_inv = np.unique(
      np.array([
          int(name.split('_')[1].split('.png')[0]) for name in color_file_list
      ]),
      return_index=True,
  )
  depth_idx, depth_inv = np.unique(
      np.array([
          int(name.split('_')[1].split('.png')[0]) for name in depth_file_list
      ]),
      return_index=True,
  )
  label_idx, label_inv = np.unique(
      np.array([
          int(name.split('_')[-1].split('.png')[0]) for name in label_file_list
      ]),
      return_index=True,
  )

  # load poses
  poses = np.loadtxt(pose_file, delimiter=' ').reshape(-1, 4, 4)

  # check if indexes are the same and the number is the same as poses
  assert (color_idx == depth_idx).all() and (
      color_idx == label_idx).all() and color_idx.shape[0] == poses.shape[0]

  # get correspondence to original file
  num_frame = int(poses.shape[0])
  rows = []
  for i in range(num_frame):
    frame_id = '{:06d}'.format(i)
    color_pth = color_file_list[color_inv[i]]
    depth_pth = depth_file_list[depth_inv[i]]
    label_pth = label_file_list[label_inv[i]]
    rows.append([frame_id, color_pth, depth_pth, label_pth])

  # write to new file
  shutil.rmtree(target_dir, ignore_errors=True)
  os.makedirs(target_dir, exist_ok=True)
  os.makedirs(join(target_dir, 'color'), exist_ok=True)
  os.makedirs(join(target_dir, 'depth'), exist_ok=True)
  os.makedirs(join(target_dir, 'intrinsic'), exist_ok=True)
  os.makedirs(join(target_dir, 'pose'), exist_ok=True)
  os.makedirs(join(target_dir, 'gt_label'), exist_ok=True)

  # first write correspondence list
  fields = [
      'frame_id',
      'original_color_path',
      'original_depth_path',
      'original_ground_truth_label_path',
  ]
  correspondence_list = [dict(zip(fields, row)) for row in rows]
  json_object = json.dumps(correspondence_list, indent=4)
  with open(join(target_dir, 'correspondence.json'), 'w') as jsonfile:
    jsonfile.write(json_object)
  logger.info("Saved old and new files correspondence to {}.".format(
      join(target_dir, 'correspondence.json')))

  logger.info("Transfering files...")
  for idx in trange(num_frame):
    frame_id, color_pth, depth_pth, label_pth = rows[idx]

    # save color
    tgt_color_pth = join(target_dir, 'color', frame_id + '.jpg')
    color_img = Image.open(join(color_dir, color_pth))
    color_img.save(tgt_color_pth)
    h, w, _ = np.asarray(color_img).shape

    # save pose
    tgt_pose_pth = join(target_dir, 'pose', frame_id + '.txt')
    np.savetxt(tgt_pose_pth, poses[idx])

    # process and save intr
    tgt_intrinsic_pth = join(target_dir, 'intrinsic', frame_id + '.txt')
    intrinsic = get_intrinsics(h, w)
    np.savetxt(tgt_intrinsic_pth, intrinsic)

    # process and save depth
    depth = cv2.imread(join(depth_dir, depth_pth), cv2.IMREAD_UNCHANGED)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    tgt_depth_pth = join(target_dir, 'depth', frame_id + '.png')
    cv2.imwrite(tgt_depth_pth, depth)

    # process and save ground truth label
    label = cv2.imread(join(label_dir, label_pth), cv2.IMREAD_UNCHANGED)
    label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    tgt_label_pth = join(target_dir, 'gt_label', frame_id + '.png')
    cv2.imwrite(tgt_label_pth, label)

  logger.info("File transfer finished!")

  logger.info("Fusing RGBD images into TSDF Volmue...")
  fuse_mesh(
      scan_dir=target_dir,
      sdf_trunc=sdf_trunc,
      voxel_length=voxel_length,
      depth_trunc=depth_trunc,
      depth_scale=1000.0,
  )  # depth_scale is a fixed value in Replica, no need to pass an argument in cli
  logger.info("Fusion finished! Saving to file as {}".format(
      join(target_dir, 'mesh.ply')))


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--scan_dir", type=str)
  parser.add_argument("--target_dir", type=str)
  parser.add_argument("--sdf_trunc", type=float, default=0.04)
  parser.add_argument("--voxel_length", type=float, default=0.008)
  parser.add_argument("--depth_trunc", type=float, default=3.0)
  parser.add_argument('--config', help='Name of config file')

  return parser.parse_args()


if __name__ == "__main__":
  args = arg_parser()
  if args.config is not None:
    gin.parse_config_file(args.config)
  process_replica(
      scan_dir=args.scan_dir,
      target_dir=args.target_dir,
      sdf_trunc=args.sdf_trunc,
      voxel_length=args.voxel_length,
      depth_trunc=args.depth_trunc,
  )
