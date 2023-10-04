import argparse
import json
import logging
import os
import shutil
import sys
from os.path import abspath, dirname, exists, join

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from tqdm import trange

sys.path.append(abspath(join(dirname(__file__), '..')))
from utils_3d import fuse_pointcloud


def get_closest_timestamp(reference_timestamps: np.ndarray,
                          target_timestamps: np.ndarray):
  """
  This function returns:
    min_time_delta: for each time in reference_timetamps, the minimum time difference (dt) w.r.t target_timestamps
    target_index: the index of element in target_timestamps that gives minimum dt
    minimum_margin: the time difference of minimum timestamps and second minimum, used for checking uniqueness of minima
  """
  time_delta = np.abs(
      reference_timestamps.reshape(-1, 1) - target_timestamps.reshape(1, -1))

  min_two_idx = time_delta.argsort(axis=1)[:, :2]
  target_index = min_two_idx[:, 0]
  min_time_delta = time_delta[np.arange(target_index.shape[0]), target_index]
  minimum_margin = time_delta[np.arange(target_index.shape[0]),
                              min_two_idx[:, 1]] - min_time_delta

  return min_time_delta, target_index, minimum_margin


def load_intrinsics(file):
  # as define here https://github.com/apple/ARKitScenes/blob/951af73d20406acf608061c16774f770c61b1405/threedod/benchmark_scripts/utils/tenFpsDataLoader.py#L46
  w, h, fx, fy, hw, hh = np.loadtxt(file)
  return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


def process_arkit(
    scan_dir: str,
    target_dir: str,
    sdf_trunc: float,
    voxel_length: float,
    depth_trunc: float,
):

  logger = logging.getLogger('ARKitProcess')
  logger.setLevel(logging.DEBUG)
  consoleHeader = logging.StreamHandler()
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  consoleHeader.setFormatter(formatter)
  logger.addHandler(consoleHeader)

  logger.info(
      "Processing ARKitScene scan to LabelMaker format, from {} to {}...".
      format(scan_dir, target_dir))

  color_dir = join(scan_dir, 'lowres_wide')
  intrinsic_dir = join(scan_dir, 'lowres_wide_intrinsics')

  depth_dir = join(scan_dir, 'lowres_depth')
  confidence_dir = join(scan_dir, 'confidence')

  trajectory_file = join(scan_dir, 'lowres_wide.traj')

  assert exists(color_dir), "lowres_wide attribute not downloaded!"
  assert exists(depth_dir), "lowres_depth attribute not downloaded!"
  assert exists(confidence_dir), "confidence attribute not downloaded!"
  assert exists(
      intrinsic_dir), "lowres_wide_intrinsics attribute not downloaded!"
  assert exists(trajectory_file), "lowres_wide.traj attribute not downloaded!"

  color_file_list = os.listdir(color_dir)
  depth_file_list = os.listdir(depth_dir)
  confidence_file_list = os.listdir(confidence_dir)
  intr_file_list = os.listdir(intrinsic_dir)

  # ts stands for timestamps, inv stands for inverse
  color_ts, color_inv = np.unique(
      np.array([
          float(name.split('_')[1].split('.png')[0]) for name in color_file_list
      ]),
      return_index=True,
  )
  depth_ts, depth_inv = np.unique(
      np.array([
          float(name.split('_')[1].split('.png')[0]) for name in depth_file_list
      ]),
      return_index=True,
  )
  confidence_ts, confidence_inv = np.unique(
      np.array([
          float(name.split('_')[1].split('.png')[0])
          for name in confidence_file_list
      ]),
      return_index=True,
  )
  intrinsic_ts, intrinsic_inv = np.unique(
      np.array([
          float(name.split('_')[1].split('.pincam')[0])
          for name in intr_file_list
      ]),
      return_index=True,
  )

  # load trajactory
  trajectory_data = np.loadtxt(trajectory_file, delimiter=' ')
  trajectory_ts = trajectory_data[:, 0]  # already sorted

  # synchronization
  logger.info("Synchronizing timestamps...")
  dt_max = 1 / 60 / 2  # half of frame time step

  # we compare all with respect to depth
  # if the matched timestamp and second matched timestamp have difference less than 1 milisecond,
  # we regard this case as the matching is not unique, and throw a warning.
  margin_threshold = 1e-3
  color_dt, color_idx, color_margin = get_closest_timestamp(depth_ts, color_ts)
  if color_margin.min() < margin_threshold:
    logger.warn(
        "Found multiple color timestamps matching in timestamps: {}".format(
            depth_ts[color_margin < margin_threshold].tolist()))

  confidence_dt, confidence_idx, confidence_margin = get_closest_timestamp(
      depth_ts, confidence_ts)
  if confidence_margin.min() < margin_threshold:
    logger.warn(
        "Found multiple confidence timestamps matching in timestamps: {}".
        format(depth_ts[confidence_margin < margin_threshold].tolist()))

  intrinsic_dt, intrinsic_idx, intrinsic_margin = get_closest_timestamp(
      depth_ts, intrinsic_ts)
  if intrinsic_margin.min() < margin_threshold:
    logger.warn(
        "Found multiple intrinsic timestamps matching in timestamps: {}".format(
            depth_ts[intrinsic_margin < margin_threshold].tolist()))

  depth_idx = np.arange(depth_ts.shape[0])

  # we also want to interpolate pose, so we have to filter out times outside trajectory timestamp
  timestamp_filter = (color_dt < dt_max) * (confidence_dt < dt_max) * (
      intrinsic_dt < dt_max) * (depth_ts >= trajectory_ts.min()) * (
          depth_ts <= trajectory_ts.max())

  timestamp = depth_ts[timestamp_filter]
  logger.info("Synchronization finished!")

  # interpolate pose
  logger.info("Interpolating poses...")
  rots = Rotation.from_rotvec(trajectory_data[:, 1:4])
  rot_spline = RotationSpline(trajectory_ts, rots)

  x_spline = CubicSpline(trajectory_ts, trajectory_data[:, 4])
  y_spline = CubicSpline(trajectory_ts, trajectory_data[:, 5])
  z_spline = CubicSpline(trajectory_ts, trajectory_data[:, 6])

  num_frame = timestamp_filter.sum()

  pose_mat = np.zeros(shape=(num_frame, 4, 4))
  pose_mat[:, 3, 3] = 1.0
  pose_mat[:, :3, :3] = rot_spline(timestamp).as_matrix()
  pose_mat[:, :3, 3] = np.stack(
      [x_spline(timestamp),
       y_spline(timestamp),
       z_spline(timestamp)], axis=1)
  logger.info("Pose interpolation finished!")

  # get correspondence to original file
  rows = []
  for i in range(num_frame):
    frame_id = '{:06d}'.format(i)
    color_pth = color_file_list[color_inv[color_idx[timestamp_filter][i]]]
    depth_pth = depth_file_list[depth_inv[depth_idx[timestamp_filter][i]]]
    confdc_pth = confidence_file_list[confidence_inv[
        confidence_idx[timestamp_filter][i]]]
    intr_pth = intr_file_list[intrinsic_inv[intrinsic_idx[timestamp_filter][i]]]
    rows.append([frame_id, color_pth, depth_pth, confdc_pth, intr_pth])

  # write to new file
  shutil.rmtree(target_dir, ignore_errors=True)
  os.makedirs(target_dir, exist_ok=True)
  os.makedirs(join(target_dir, 'color'), exist_ok=True)
  os.makedirs(join(target_dir, 'depth'), exist_ok=True)
  os.makedirs(join(target_dir, 'intrinsic'), exist_ok=True)
  os.makedirs(join(target_dir, 'pose'), exist_ok=True)

  # first write correspondence list
  fields = [
      'frame_id', 'original_color_path', 'original_depth_path',
      'original_confidence_path', 'original_intrinsic_path'
  ]
  correspondence_list = [dict(zip(fields, row)) for row in rows]
  json_object = json.dumps(correspondence_list, indent=4)
  with open(join(target_dir, 'correspondence.json'), 'w') as jsonfile:
    jsonfile.write(json_object)
  logger.info("Saved old and new files correspondence to {}.".format(
      join(target_dir, 'correspondence.json')))

  logger.info("Transfering files...")
  for idx in trange(num_frame):
    frame_id, color_pth, depth_pth, confdc_pth, intr_pth = rows[idx]

    # save color
    tgt_color_pth = join(target_dir, 'color',
                         frame_id + '.jpg')  # png -> jpg, compressed
    color_img = Image.open(join(color_dir, color_pth))
    color_img.save(tgt_color_pth)
    h, w, _ = np.asarray(color_img).shape

    # save pose
    tgt_pose_dir = join(target_dir, 'pose', frame_id + '.txt')
    np.savetxt(tgt_pose_dir, pose_mat[idx])

    # process and save intr
    tgt_intrinsic_dir = join(target_dir, 'intrinsic', frame_id + '.txt')
    np.savetxt(tgt_intrinsic_dir, load_intrinsics(join(intrinsic_dir,
                                                       intr_pth)))

    # process and save depth
    depth = cv2.imread(join(depth_dir, depth_pth), cv2.IMREAD_UNCHANGED)
    confdc = cv2.imread(join(confidence_dir, confdc_pth), cv2.IMREAD_UNCHANGED)

    depth[confdc < 2] = 0
    # depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST) # no need to resie as we use lowres color

    tgt_depth_dir = join(target_dir, 'depth', frame_id + '.png')
    cv2.imwrite(tgt_depth_dir, depth)

  logger.info("File transfer finished!")

  logger.info("Fusing RGBD images into TSDF Volmue...")
  fuse_pointcloud(
      scan_dir=target_dir,
      sdf_trunc=sdf_trunc,
      voxel_length=voxel_length,
      depth_trunc=depth_trunc,
      depth_scale=1000.0,
  )  # depth_scale is a fixed value in ARKitScene, no need to pass an argument in cli
  logger.info("Fusion finished! Saving to file as {}".format(
      join(scan_dir, 'pointcloud.ply')))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--scan_dir", type=str)
  parser.add_argument("--target_dir", type=str)
  parser.add_argument("--sdf_trunc", type=float, default=0.06)
  parser.add_argument("--voxel_length", type=float, default=0.02)
  parser.add_argument("--depth_trunc", type=float, default=3.0)
  flags = parser.parse_args()

  assert exists(str(flags.scan_dir))
  process_arkit(
      scan_dir=flags.scan_dir,
      target_dir=flags.target_dir,
      sdf_trunc=flags.sdf_trunc,
      voxel_length=flags.voxel_length,
      depth_trunc=flags.depth_trunc,
  )
