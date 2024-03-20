import argparse
import json
import logging
import os
import shutil
import sys
import glob
from os.path import abspath, dirname, exists, join
from typing import List, Optional

import cv2
import gin
import numpy as np
import open3d as o3d

from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from tqdm import trange
import matplotlib.pyplot as plt

from copy import copy

sys.path.append(abspath(join(dirname(__file__), '..')))
from utils_3d import fuse_mesh


def extract_pose(data):
  return np.asarray(data['cameraPoseARFrame'])

def extract_intrinsics(data):
  return np.asarray(data['intrinsics'])

def render_depth(world_to_cam, intrinsics, mesh, resolution):
  rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        width_px=resolution[1],
        height_px=resolution[0],
        intrinsic_matrix=intrinsics[:3, :3],
        extrinsic_matrix=world_to_cam,  # world to camera
    )

  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(mesh)
  vis = scene.cast_rays(rays)

  depth = vis['t_hit'].numpy()
  return depth


@gin.configurable
def process_scanner3d(
    scan_dir: str,
    target_dir: str,
    sdf_trunc: float,
    voxel_length: float,
    depth_trunc: float,
    resize: Optional[List] = None,
):

  logger = logging.getLogger('Scanner3DProcess')
  logger.setLevel(logging.DEBUG)
  consoleHeader = logging.StreamHandler()
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  consoleHeader.setFormatter(formatter)
  logger.addHandler(consoleHeader)

  logger.info(
      "Processing Scanner3D scan to LabelMaker format, from {} to {}...".
      format(scan_dir, target_dir))

  color_dir = join(scan_dir)

  color_file_list = glob.glob(join(color_dir, 'frame_*.jpg'))
  color_file_list = sorted([os.path.basename(f) for f in color_file_list], key=lambda x: int(x.split('_')[1].split('.jpg')[0]))

  traj_file_list = glob.glob(join(color_dir, 'frame_*.json'))
  traj_file_list = sorted([os.path.basename(f) for f in traj_file_list], key=lambda x: int(x.split('_')[1].split('.json')[0]))

  # write to new file
  shutil.rmtree(target_dir, ignore_errors=True)
  os.makedirs(target_dir, exist_ok=True)
  os.makedirs(join(target_dir, 'color'), exist_ok=True)
  os.makedirs(join(target_dir, 'depth'), exist_ok=True)
  os.makedirs(join(target_dir, 'intrinsic'), exist_ok=True)
  os.makedirs(join(target_dir, 'pose'), exist_ok=True)

  # load scanner 3d mesh
  mesh = o3d.io.read_triangle_mesh(scan_dir + '/export.obj')
  mesh.compute_vertex_normals()
  mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

  num_frame = len(color_file_list)
  logger.info("Transfering files...")
  for idx in trange(num_frame):
    color_pth, traj_pth = color_file_list[idx], traj_file_list[idx]

    traj_data = json.load(open(join(color_dir, traj_pth)))

    pose_mat = extract_pose(traj_data).reshape((4, 4))

    rotation = pose_mat[:3, :3]

    # rotate the camera and flip the axis
    rotation[2, :] = -rotation[2, :]
    rotation[1, :] = -rotation[1, :]
    rotation[0, :] = -rotation[0, :]
    pose_mat[:3, :3] = rotation
    pose_mat[:, 0] = -pose_mat[:, 0]


    # save color
    tgt_color_pth = join(target_dir, 'color',
                         str(idx) + '.jpg')  # png -> jpg, compressed
    color_img = Image.open(join(color_dir, color_pth))
    if resize is not None:
      original_size = (color_img.width, color_img.height)
      color_img = color_img.resize(resize)
    color_img.save(tgt_color_pth)

    intr = extract_intrinsics(traj_data).reshape((3, 3))

    if resize is not None:
      w, h = resize
      width_factor = w / float(original_size[0])
      height_factor = h / float(original_size[1])
      intr[0, 0] *= width_factor
      intr[1, 1] *= height_factor
      intr[0, 2] *= width_factor
      intr[1, 2] *= height_factor
    else:
      h, w, _ = np.asarray(color_img).shape
    depth = render_depth(np.linalg.inv(pose_mat), intr, mesh, (h, w))
    depth = depth * 1000
    depth = depth.astype(np.uint16)

    # save pose
    tgt_pose_pth = join(target_dir, 'pose', str(idx) + '.txt')
    np.savetxt(tgt_pose_pth, pose_mat)

    # process and save intr
    tgt_intrinsic_pth = join(target_dir, 'intrinsic', str(idx) + '.txt')
    np.savetxt(tgt_intrinsic_pth, intr)

    tgt_depth_pth = join(target_dir, 'depth', str(idx) + '.png')
    cv2.imwrite(tgt_depth_pth, depth)

  logger.info("File transfer finished!")

  logger.info("Fusing RGBD images into TSDF Volmue...")
  fuse_mesh(
      scan_dir=target_dir,
      sdf_trunc=sdf_trunc,
      voxel_length=voxel_length,
      depth_trunc=depth_trunc,
      depth_scale=1000.0,
  )  # depth_scale is a fixed value in ARKitScene, no need to pass an argument in cli
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
  process_scanner3d(
      scan_dir=args.scan_dir,
      target_dir=args.target_dir,
      sdf_trunc=args.sdf_trunc,
      voxel_length=args.voxel_length,
      depth_trunc=args.depth_trunc,
      resize=[640, 480],
  )
