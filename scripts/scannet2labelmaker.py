import argparse
import json
import logging

import os
import shutil
from pathlib import Path
import sys
from os.path import abspath, dirname, exists, join

import cv2
import gin
import numpy as np
from PIL import Image

from tqdm import trange

sys.path.append(abspath(join(dirname(__file__), '..')))
from utils_3d import fuse_mesh


@gin.configurable
def process_scannet(
    scan_dir: str,
    target_dir: str,
    sdf_trunc: float,
    voxel_length: float,
    depth_trunc: float,
):
  logger = logging.getLogger('ScanNetProcess')
  logger.setLevel(logging.DEBUG)
  consoleHeader = logging.StreamHandler()
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  consoleHeader.setFormatter(formatter)
  logger.addHandler(consoleHeader)

  logger.info(
      "Processing ScanNet scan to LabelMaker format, from {} to {}...".format(
          scan_dir, target_dir))

  color_dir = join(scan_dir, 'data/color')
  depth_dir = join(scan_dir, 'data/depth')
  pose_dir = join(scan_dir, "data/pose")
  intrinsic_path = join(scan_dir, "data/intrinsic/intrinsic_color.txt")

  assert exists(color_dir)
  assert exists(depth_dir)
  assert exists(intrinsic_path)
  assert exists(pose_dir)

  color_file_list = list(Path(color_dir).glob("*.jpg"))
  depth_file_list = list(Path(depth_dir).glob("*.png"))
  pose_file_list = list(Path(pose_dir).glob("*.txt"))

  color_file_list.sort(key=lambda e: int(e.stem))
  depth_file_list.sort(key=lambda e: int(e.stem))
  pose_file_list.sort(key=lambda e: int(e.stem))

  color_idx_set = set()
  for p in color_file_list:
    color_idx_set.add(int(p.stem))

  depth_idx_set = set()
  for p in depth_file_list:
    depth_idx_set.add(int(p.stem))

  pose_idx_set = set()
  for p in pose_file_list:
    pose_idx_set.add(int(p.stem))

  assert color_idx_set == depth_idx_set
  assert color_idx_set == pose_idx_set

  # make dirs for target folder
  # write to new file
  os.makedirs(target_dir, exist_ok=True)
  # delete the following folders or path and create new one.
  # do not delete others as their might be intermediate result
  shutil.rmtree(join(target_dir, 'color'), ignore_errors=True)
  os.makedirs(join(target_dir, 'color'), exist_ok=True)
  shutil.rmtree(join(target_dir, 'depth'), ignore_errors=True)
  os.makedirs(join(target_dir, 'depth'), exist_ok=True)
  shutil.rmtree(join(target_dir, 'intrinsic'), ignore_errors=True)
  os.makedirs(join(target_dir, 'intrinsic'), exist_ok=True)
  shutil.rmtree(join(target_dir, 'pose'), ignore_errors=True)
  os.makedirs(join(target_dir, 'pose'), exist_ok=True)
  shutil.rmtree(join(target_dir, 'correspondence.sjon'), ignore_errors=True)
  shutil.rmtree(join(target_dir, 'mesh.ply'), ignore_errors=True)

  intrinsic = np.loadtxt(intrinsic_path, delimiter=" ",
                         dtype=np.float32)[:3, :3]

  correspondence_list = []
  logger.info("Transfering files...")
  for i in trange(len(color_file_list)):
    color_path = color_file_list[i]
    pose_path = pose_file_list[i]
    depth_path = depth_file_list[i]

    # read pose
    pose_mat = np.loadtxt(str(pose_path), delimiter=" ", dtype=np.float32)
    extrinsic_mat = np.linalg.inv(pose_mat)
    z_in_camera_view = extrinsic_mat[:3, 2]
    angle_to_image_up = np.arctan2(
        -z_in_camera_view[0],
        -z_in_camera_view[1],
    )
    z_direction = np.floor(angle_to_image_up * 2 / np.pi + 8.5).astype(int) % 4

    # read rgb
    color_img = Image.open(str(color_path))
    h, w, _ = np.asarray(color_img).shape

    # read and process depth
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    # save
    frame_id = '{:06d}'.format(i)
    tgt_color_pth = join(target_dir, 'color', frame_id + '.jpg')
    color_img.save(tgt_color_pth)

    tgt_pose_pth = join(target_dir, 'pose', frame_id + '.txt')
    np.savetxt(tgt_pose_pth, pose_mat)

    tgt_intrinsic_pth = join(target_dir, 'intrinsic', frame_id + '.txt')
    np.savetxt(tgt_intrinsic_pth, intrinsic)

    tgt_depth_pth = join(target_dir, 'depth', frame_id + '.png')
    cv2.imwrite(tgt_depth_pth, depth)

    correspondence_list.append({
        "frame_id": frame_id,
        "angel_z_up": float(angle_to_image_up),
        "z_direction": int(z_direction),
        "H": h,
        "W": w,
    })

  json_object = json.dumps(correspondence_list, indent=4)
  with open(join(target_dir, 'correspondence.json'), 'w') as jsonfile:
    jsonfile.write(json_object)

  logger.info("File transfer finished!")

  logger.info("Fusing RGBD images into TSDF Volmue...")
  fuse_mesh(
      scan_dir=target_dir,
      sdf_trunc=sdf_trunc,
      voxel_length=voxel_length,
      depth_trunc=depth_trunc,
      depth_scale=1000.0,
  )
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
  process_scannet(
      scan_dir=args.scan_dir,
      target_dir=args.target_dir,
      sdf_trunc=args.sdf_trunc,
      voxel_length=args.voxel_length,
      depth_trunc=args.depth_trunc,
  )
