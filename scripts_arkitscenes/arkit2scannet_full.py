import numpy as np
from PIL import Image
import argparse
import os
from pathlib import Path
import logging
import shutil
from tqdm import tqdm
import cv2

log = logging.getLogger(__name__)


def axis_angle_to_mat(angles):
    matrix, jac = cv2.Rodrigues(angles)
    return matrix

def load_intrinsics(file):
    # as define here https://github.com/apple/ARKitScenes/blob/951af73d20406acf608061c16774f770c61b1405/threedod/benchmark_scripts/utils/tenFpsDataLoader.py#L46
    w, h, fx, fy, hw, hh = np.loadtxt(file)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


def load_poses(file):
    trajectory = {}
    with open(file, 'r') as f:
        for line in f:
            elements = line.rstrip().split(' ')
            
            timestamp = round(float(elements[0]), 3)
            rotation = axis_angle_to_mat(np.asarray([float(e) for e in elements[1:4]]))
            translation = np.asarray([float(e) for e in elements[4:]])
            
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, -1] = translation
            
            trajectory[timestamp] = {'rotation': rotation, 'translation': translation, 'pose': pose}
    return trajectory


def get_closest_timestamp(timestamp, pose_timestamps, max_delta=0.05):
    closest_timestamp = np.infty
    closest_delta = np.infty
    
    for ts in pose_timestamps:
        delta = abs(timestamp - ts)
        if delta < closest_delta:
            closest_timestamp = ts
            closest_delta = delta
    
    if closest_delta < max_delta:
        return closest_timestamp
    else:
        return np.infty
            


def process_arkit(scene_dir, depth_keys):

    scene_dir = Path(scene_dir)
    assert scene_dir.exists()
    imgdir = scene_dir / "color"
    shutil.rmtree(imgdir, ignore_errors=True)
    imgdir.mkdir(exist_ok=False)

    depthdir = scene_dir / "depth"
    shutil.rmtree(depthdir, ignore_errors=True)
    depthdir.mkdir(exist_ok=False)
    
    posedir = scene_dir / "pose"
    shutil.rmtree(posedir, ignore_errors=True)
    posedir.mkdir(exist_ok=False)


    rgb_keys = sorted(
        x.name.split('.png')[0] for x in (scene_dir / 'vga_wide').iterdir())
    
    poses = load_poses(scene_dir / 'lowres_wide.traj')
    pose_timestamps = list(poses)

    intrinsics_loaded = False

    # iterate over all rgb keys and if there is a depth key close enough add
    # the depth as well otherwise add zero depth

    matched_frames = 0
    already_matched = {k: False for k in depth_keys}

    for i, k in enumerate(tqdm(rgb_keys)):
        no_match = False

        # get the closest depth key
        for j, ck in enumerate(depth_keys):
            if ck >= k:
                key_before = depth_keys[j - 1]
                key_after = ck
                break

        if already_matched[key_before]:
            key_before = key_after

        if already_matched[key_after]:
            no_match = True
        

        # now decide whether to take before or after
        # based on which one is closer
        depth_key = min(
            key_before,
            key_after,
            key=lambda x: abs(float(x.split('_')[1]) - float(k.split('_')[1])))
        

        diff = abs(float(k.split('_')[-1]) - float(depth_key.split('_')[-1]))
        if diff < 0.025:
            already_matched[depth_key] = True
        else:
            no_match = True

        rgb = Image.open(scene_dir / "vga_wide" / f"{k}.png")

        if not intrinsics_loaded:
            intrinsics = load_intrinsics(scene_dir / 'vga_wide_intrinsics' / f"{k}.pincam")
            np.savetxt(scene_dir / 'intrinsics.txt', intrinsics)
            intrinsics_loaded = True
        
        # # rotate
        # rgb = rgb.transpose(Image.ROTATE_270)
        timestamp = float(k.replace('.png', '').split('_')[-1])
        c_ts = get_closest_timestamp(timestamp, pose_timestamps)
    
        if c_ts == np.infty:
            # print('No matching pose')
            continue

        rgb.save(imgdir / f"{i}.jpg")
        rgb = np.asarray(rgb)
        h, w, _ = rgb.shape

        if no_match:
            continue
            depth = np.zeros((h, w), dtype=np.uint16)
        depth = cv2.imread(str(scene_dir / "lowres_depth" / f"{depth_key}.png"),
                        cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(scene_dir / "confidence" / f"{depth_key}.png"),
                                cv2.IMREAD_UNCHANGED)

        depth[confidence < 2] = 0
       
        
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)


        matched_frames += 1

        # # rotate
        # depth = depth.T[:, ::-1]

        cv2.imwrite(str(scene_dir / 'depth' / f"{i}.png"), depth)

        

        pose = poses[c_ts]['pose']
        pose = np.linalg.inv(pose) # need to invert for scannet format
        np.savetxt(posedir / f"{i}.txt", pose)

    print(f"Matched {matched_frames} frames")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_dir", type=str)
    parser.add_argument("--sample", type=int, default=1)
    flags = parser.parse_args()

    scene_dir = Path(str(flags.scene_dir)).resolve()
    assert scene_dir.exists()
    keys = sorted(
        x.name.split('.png')[0]
        for x in (scene_dir / 'lowres_depth').iterdir())
    # now subsample
    keys = keys[::int(flags.sample)]
    print(f"Processing {len(keys)} frames")

    process_arkit(scene_dir, keys)
