import argparse
import csv
import os
import shutil
from os.path import exists, join

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from tqdm import trange


def get_closest_timestamp(timestamps: np.ndarray,
                          target_timestamps: np.ndarray):
  time_delta = np.abs(
      timestamps.reshape(-1, 1) - target_timestamps.reshape(1, -1))

  min_time_delta = time_delta.min(axis=1)
  target_index = time_delta.argmin(axis=1)

  return min_time_delta, target_index


def load_intrinsics(file):
  # as define here https://github.com/apple/ARKitScenes/blob/951af73d20406acf608061c16774f770c61b1405/threedod/benchmark_scripts/utils/tenFpsDataLoader.py#L46
  w, h, fx, fy, hw, hh = np.loadtxt(file)
  return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


def process_arkit(scan_path: str, target_dir: str):

  color_dir = join(scan_path, 'lowres_wide')
  intr_dir = join(scan_path, 'lowres_wide_intrinsics')

  depth_dir = join(scan_path, 'lowres_depth')
  confdc_dir = join(scan_path, 'confidence')

  traj_pth = join(scan_path, 'lowres_wide.traj')

  assert exists(color_dir), "lowres_wide attribute not downloaded!"
  assert exists(depth_dir), "lowres_depth attribute not downloaded!"
  assert exists(confdc_dir), "confidence attribute not downloaded!"
  assert exists(intr_dir), "lowres_wide_intrinsics attribute not downloaded!"
  assert exists(traj_pth), "lowres_wide.traj attribute not downloaded!"

  color_pth_list = os.listdir(color_dir)
  depth_pth_list = os.listdir(depth_dir)
  confdc_pth_list = os.listdir(confdc_dir)
  intr_pth_list = os.listdir(intr_dir)

  color_ts, color_inv = np.unique(
      np.array([
          float(name.split('_')[1].split('.png')[0]) for name in color_pth_list
      ]),
      return_index=True,
  )
  depth_ts, depth_inv = np.unique(
      np.array([
          float(name.split('_')[1].split('.png')[0]) for name in depth_pth_list
      ]),
      return_index=True,
  )
  confdc_ts, confdc_inv = np.unique(
      np.array([
          float(name.split('_')[1].split('.png')[0]) for name in confdc_pth_list
      ]),
      return_index=True,
  )
  intr_ts, intr_inv = np.unique(
      np.array([
          float(name.split('_')[1].split('.pincam')[0])
          for name in intr_pth_list
      ]),
      return_index=True,
  )

  # load trajactory
  traj_data = np.loadtxt(traj_pth, delimiter=' ')
  traj_ts = traj_data[:, 0]  # already sorted

  # synchronization
  print('Synchronizing...', end='')
  dt_max = 1 / 60 / 2  # half of frame time step

  # we compare all with respect to depth
  color_dt, color_idx = get_closest_timestamp(depth_ts, color_ts)
  confdc_dt, confdc_idx = get_closest_timestamp(depth_ts, confdc_ts)
  intr_dt, intr_idx = get_closest_timestamp(depth_ts, intr_ts)
  depth_idx = np.arange(depth_ts.shape[0])

  # we also want to interpolate pose, so we have to filter out times outside trajectory timestamp
  timestamp_filter = (color_dt < dt_max) * (confdc_dt < dt_max) * (
      intr_dt < dt_max) * (depth_ts >= traj_ts.min()) * (depth_ts
                                                         <= traj_ts.max())

  timestamp = depth_ts[timestamp_filter]
  print('Done!')

  # interpolate pose
  print('Interpolating poses...', end='')
  rots = Rotation.from_rotvec(traj_data[:, 1:4])
  rot_spline = RotationSpline(traj_ts, rots)

  x_spline = CubicSpline(traj_ts, traj_data[:, 4])
  y_spline = CubicSpline(traj_ts, traj_data[:, 5])
  z_spline = CubicSpline(traj_ts, traj_data[:, 6])

  num_frame = timestamp_filter.sum()

  pose_mat = np.zeros(shape=(num_frame, 4, 4))
  pose_mat[:, 3, 3] = 1.0
  pose_mat[:, :3, :3] = rot_spline(timestamp).as_matrix()
  pose_mat[:, :3, 3] = np.stack(
      [x_spline(timestamp),
       y_spline(timestamp),
       z_spline(timestamp)], axis=1)
  print('Done!')

  # get correspondence to original file
  fields = [
      'labelmaker_id', 'original_color', 'original_depth',
      'original_confidence', 'original_intrinsic'
  ]
  rows = []
  for i in range(num_frame):
    lb_id = '{:06d}'.format(i)
    color_pth = color_pth_list[color_inv[color_idx[timestamp_filter][i]]]
    depth_pth = depth_pth_list[depth_inv[depth_idx[timestamp_filter][i]]]
    confdc_pth = confdc_pth_list[confdc_inv[confdc_idx[timestamp_filter][i]]]
    intr_pth = intr_pth_list[intr_inv[intr_idx[timestamp_filter][i]]]
    rows.append([lb_id, color_pth, depth_pth, confdc_pth, intr_pth])

  # write to new file
  shutil.rmtree(target_dir, ignore_errors=True)
  os.makedirs(target_dir, exist_ok=True)
  os.makedirs(join(target_dir, 'color'), exist_ok=True)
  os.makedirs(join(target_dir, 'depth'), exist_ok=True)
  os.makedirs(join(target_dir, 'intr'), exist_ok=True)
  os.makedirs(join(target_dir, 'pose'), exist_ok=True)

  # first write correspondence list
  with open(join(target_dir, 'corres.csv'), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)

  print('Transfering files...')
  for idx in trange(num_frame):
    lb_id, color_pth, depth_pth, confdc_pth, intr_pth = rows[idx]

    # save color
    tgt_color_pth = join(target_dir, 'color',
                         lb_id + '.jpg')  # png -> jpg, compressed
    color_img = Image.open(join(color_dir, color_pth))
    color_img.save(tgt_color_pth)
    h, w, _ = np.asarray(color_img).shape

    # save pose
    tgt_pose_dir = join(target_dir, 'pose', lb_id + '.txt')
    np.savetxt(tgt_pose_dir, pose_mat[idx])

    # process and save intr
    tgt_intr_dir = join(target_dir, 'intr', lb_id + '.txt')
    np.savetxt(tgt_intr_dir, load_intrinsics(join(intr_dir, intr_pth)))

    # process and save depth
    depth = cv2.imread(join(depth_dir, depth_pth), cv2.IMREAD_UNCHANGED)
    confdc = cv2.imread(join(confdc_dir, confdc_pth), cv2.IMREAD_UNCHANGED)

    depth[confdc < 2] = 0
    # depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST) # no need to resie as we use lowres color

    tgt_depth_dir = join(target_dir, 'depth', lb_id + '.png')
    cv2.imwrite(tgt_depth_dir, depth)

  print('Done!')


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("scan_pth", type=str)
  parser.add_argument("target_dir", type=str)
  flags = parser.parse_args()

  assert exists(str(flags.scan_pth))
  process_arkit(scan_path=flags.scan_pth, target_dir=flags.target_dir)
