import sys, os
import matplotlib.pyplot as plt
import logging
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
import shutil

logging.basicConfig(level="INFO")
log = logging.getLogger('pose conversion')


def convert_poses(scene_dir):
    scene_dir = Path(scene_dir)
    log.info(f'converting poses for {scene_dir}')
    assert scene_dir.exists() and scene_dir.is_dir()
    poses = np.loadtxt(scene_dir / 'traj_w_c.txt', delimiter=' ').reshape(-1, 4, 4)

    pose_dir = scene_dir / 'pose'
    shutil.rmtree(pose_dir, ignore_errors=True)
    pose_dir.mkdir(exist_ok=False)
    for k in range(poses.shape[0]):
        # I checked, scannet is also camera to world
        np.savetxt(pose_dir / f'{k}.txt', poses[k])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=str)
    flags = parser.parse_args()

    base_dir = Path(flags.basedir)
    assert base_dir.exists() and base_dir.is_dir()

    log.info('converting poses from semantic-nerf replica format to scannet format')

    for room in base_dir.iterdir():
        if not room.is_dir():
            continue
        for sequence in room.iterdir():
            # check that it is a sequence
            if not str(sequence.name).startswith('Sequence'):
                continue
            convert_poses(base_dir / room / sequence)
