import argparse
import glob
import json
import os
from pathlib import Path
import logging
from PIL import Image
import shutil
import re
import pandas as pd
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
import torch

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from segmentation_tools.label_data import get_replica, get_scannet_all, get_wordnet
from segmentation_tools.visualisation import random_color

logging.basicConfig(level="INFO")
log = logging.getLogger('sdfstudio data preprocessing')


def load_from_colmap(path, invert_pose=False):
    images = []
    with open(os.path.join(path, 'images.txt'), 'r') as file:
        line_count = 0
        for line in file:
            if len(line) <= 1:
                line_count += 1
                continue
            if line[0] == '#':
                continue
            if line_count % 2 == 0:
                images.append(line.rstrip().split(' '))
            line_count += 1

    pose_by_name = {}
    for i, image in enumerate(images):
        translation = np.array([float(image[5]), float(image[6]), float(image[7])])
        rotation = np.array([float(image[1]), float(image[2]), float(image[3]), float(image[4])])
        q = Quaternion(rotation)
        mat = q.rotation_matrix
        pose = np.eye(4)
        pose[:3, :3] = mat
        pose[:3, 3] = translation
        if invert_pose:
            pose = np.linalg.inv(pose)
        pose_by_name[image[9]] = pose

    return pose_by_name


def sdfstudio_preprocessing(scene_dirs,
                            image_size=384,
                            img_template='rgb/{k}.jpg',
                            depth_template='depth/{k}.png',
                            mono_depth_template='omnidata_depth/{k}.png',
                            mono_normal_template='omnidata_normal/{k}.npy',
                            label_template='pred_consensus/{k}.png',
                            groups_template='pred_sam/{k}.png',
                            use_scannet_pose=False,
                            semantic_info=[],
                            sampling=1):
    scene_keys = {}
    color_paths = []
    depth_paths = []
    for scene_dir in scene_dirs:
        scene_dir_path = Path(scene_dir)
        assert scene_dir_path.exists()
        files = glob.glob(str(scene_dir_path / img_template.format(k='*')),
                          recursive=True)
        keys = sorted(
            int(re.search(img_template.format(k='(\d+)'), x).group(1))
            for x in files)
        keys = keys[::sampling]
        scene_keys[scene_dir] = keys

        color_paths.extend(
            list(scene_dir_path / img_template.format(k=k) for k in keys))
        depth_paths.extend(
            list(scene_dir_path / depth_template.format(k=k) for k in keys))

    original_image_dim = Image.open(color_paths[0]).size
    image_crop_size = min(original_image_dim)
    original_depth_dim = Image.open(depth_paths[0]).size
    depth_crop_size = min(original_depth_dim)
    original_label_dim = Image.open(
        str(scene_dir_path / label_template.format(k=keys[0]))).size
    label_crop_size = min(original_label_dim)
    print(f"original image dim: {original_image_dim}")

    trans_totensor = transforms.Compose([
        transforms.CenterCrop(image_crop_size),
        transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
    ])

    depth_trans_totensor = transforms.Compose([
        # transforms.Resize(original_image_dim, interpolation=PIL.Image.NEAREST),
        transforms.CenterCrop(depth_crop_size),
        transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
    ])

    label_trans_totensor = transforms.Compose([
        # transforms.Resize(original_image_dim, interpolation=PIL.Image.NEAREST),
        transforms.CenterCrop(label_crop_size),
        transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
    ])

    monocue_dim = Image.open(
        str(scene_dir_path / mono_depth_template.format(k=keys[0]))).size
    monocue_crop_size = min(monocue_dim)
    monodepth_trans_totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(monocue_crop_size),
        transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
    ])

    monocue_dim = np.load(
        str(scene_dir_path / mono_normal_template.format(k=keys[0]))).shape[:2]
    monocue_crop_size = min(monocue_dim)
    mononormal_trans_totensor = transforms.Compose([
        # transforms.Resize(original_image_dim, interpolation=PIL.Image.NEAREST),
        transforms.ToTensor(),
        transforms.CenterCrop(monocue_crop_size),
        transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
    ])

    # groups for patch loss
    original_groups_dim = Image.open(str(scene_dir_path / groups_template.format(k=keys[0]))).size
    groups_crop_size = min(original_groups_dim)
    groups_trans_totensor = transforms.Compose([
        transforms.CenterCrop(groups_crop_size),
        transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
    ])
    # # load color
    # color_path = input_path / "frames" / "color"
    # color_paths = sorted(glob.glob(os.path.join(color_path, "*.jpg")), key=lambda x: int(os.path.basename(x)[:-4]))

    # # load depth
    # depth_path = input_path / "frames" / "depth"
    # depth_paths = sorted(glob.glob(os.path.join(depth_path, "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))

    # load intrinsic
    # intrinsic_path = input_path / "frames" / "intrinsic" / "intrinsic_color.txt"
    camera_intrinsic = np.loadtxt(scene_dir_path / 'intrinsic' /
                                  'intrinsic_color.txt')
    orig_camera_intrinsic = camera_intrinsic.copy()

    # load pose
    poses = []
    which_scenedir = []
    which_key = []
    for scene_dir in scene_dirs:
        scene_dir_path = Path(scene_dir)
        if not use_scannet_pose:
            # colmap_poses = pd.read_csv(
            #     scene_dir_path / 'refinedpose' / 'images.txt',
            #     sep=' ',
            #     header=None,
            #     index_col=9,
            #     names=[
            #         'idx', 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz', 'camera_id',
            #         'rest'
            #     ])
            colmap_poses = load_from_colmap(str(scene_dir_path / 'refinedpose'), invert_pose=True)
        for k in scene_keys[scene_dir]:
            if use_scannet_pose:
                c2w = np.loadtxt(scene_dir_path / 'pose' / f'{k}.txt')
            else:
                # row = colmap_poses.loc[f'{k}.jpg']
                # qvec = np.array(tuple(map(float, row[['qx', 'qy', 'qz', 'qw']])))
                # R = Rotation.from_quat(qvec).as_matrix()
                # # R = Quaternion(*map(float, row[['qw', 'qx', 'qy', 'qz']])).rotation_matrix
                # tvec = np.array(tuple(map(float, row[['tx', 'ty', 'tz']])))
                # t = tvec.reshape([3, 1])
                # c2w = np.eye(4)
                # c2w[:3, :3] = R
                # c2w[:3, 3] = t.squeeze()
                # c2w = np.linalg.inv(c2w)
                c2w = colmap_poses[f'{k}.jpg']
                # c2w = np.concatenate(
                #     [np.concatenate([R.T, -R.T @ t], 1),
                #      np.array([[0, 0, 0, 1]])], 0)
                # c2w = np.concatenate(
                #     [np.concatenate([R, t], 1),
                #      np.array([[0, 0, 0, 1]])], 0)
            poses.append(c2w)
            which_scenedir.append(scene_dir)
            which_key.append(k)
    assert len(poses) == sum(len(v) for v in scene_keys.values())
    assert len(color_paths) == sum(len(v) for v in scene_keys.values())
    poses = np.array(poses)

    # deal with invalid poses
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)


    center = (min_vertices + max_vertices) / 2.0
    scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
    # center = np.zeros(3)
    # scale = 1.0
    print(center, scale)

    # we should normalize pose to unit cube
    poses[:, :3, 3] -= center
    poses[:, :3, 3] *= scale

    # inverse normalization
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] -= center
    scale_mat[:3] *= scale
    scale_mat = np.linalg.inv(scale_mat)

    # copy image
    W, H = original_image_dim

    # center crop by 2 * image_size
    # offset_x = (W - image_size * 2) * 0.5
    # offset_y = (H - image_size * 2) * 0.5
    offset_x = (W - image_crop_size) * 0.5
    offset_y = (H - image_crop_size) * 0.5
    camera_intrinsic[0, 2] -= offset_x
    camera_intrinsic[1, 2] -= offset_y
    # resize from 384*2 to 384
    resize_factor = image_size / image_crop_size
    camera_intrinsic[:2, :] *= resize_factor

    class_hist = np.zeros((len(semantic_info)), dtype=np.int64)

    K = camera_intrinsic

    frames = []
    camera_path_per_scenedir = {
        s: []
        for s in scene_dirs
    }  # saves full trajectory for output renderings regardless of sampling
    out_index = 0
    output_path = Path(scene_dirs[0]) / 'sdfstudio'
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(exist_ok=False)
    for idx, (valid, pose, image_path, depth_path, scene_dir, k) in enumerate(
            zip(valid_poses, poses, color_paths, depth_paths, which_scenedir,
                which_key)):
        # sdfstudio dataset transforms the rotation matrix, we need to do this here for
        # rendering
        render_pose = pose.copy()
        render_pose[:3, 1:3] *= -1
        camera_path_per_scenedir[scene_dir].append({
            "camera_to_world":
            render_pose.tolist(),
            "fx":
            orig_camera_intrinsic[0, 0],
            "fy":
            orig_camera_intrinsic[1, 1],
            "key": k,
        })

        # if idx % sampling != 0:
        #     continue
        if not valid:
            continue

        target_image = output_path / f"{out_index:06d}_rgb.png"
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        img_tensor.save(target_image)

        # load depth
        target_depth_image = output_path / f"{out_index:06d}_sensor_depth.png"
        depth = cv2.imread(str(depth_path), -1).astype(np.float32) / 1000.0

        depth_PIL = Image.fromarray(depth)
        new_depth = depth_trans_totensor(depth_PIL)
        new_depth = np.copy(np.asarray(new_depth))
        # scale depth as we normalize the scene to unit box
        new_depth *= scale
        plt.imsave(target_depth_image, new_depth, cmap="viridis")
        np.save(str(target_depth_image).replace(".png", ".npy"), new_depth)
        np.savetxt(str(output_path / f"{out_index:06d}_camtoworld.txt"), pose)

        # load label
        scene_dir_path = Path(scene_dir)
        label = Image.open(str(scene_dir_path / label_template.format(k=k)))
        label_tensor = label_trans_totensor(label)
        label_path = output_path / f"{out_index:06d}_label.png"
        label_tensor.save(str(label_path))

        # semantic class histogram
        class_hist += np.bincount(np.asarray(label_tensor).flatten(), minlength=len(semantic_info))

        # load mono depth
        mono_depth = cv2.imread(str(scene_dir_path / mono_depth_template.format(k=k)), -1).astype(np.float32)
        monodepth_tensor = monodepth_trans_totensor(mono_depth).squeeze()
        # this depth is scaled, we need to unscale it
        monodepth_tensor = monodepth_tensor - monodepth_tensor.min()
        monodepth_tensor = monodepth_tensor / monodepth_tensor.max()
        monodepth_path = output_path / f"{out_index:06d}_depth.npy"
        np.save(str(monodepth_path), np.asarray(monodepth_tensor))
        # load mono normal
        mono_normal = np.load(str(scene_dir_path / mono_normal_template.format(k=k)))
        mononormal_tensor = mononormal_trans_totensor(mono_normal)
        mononormal_path = output_path / f"{out_index:06d}_normal.npy"
        np.save(str(mononormal_path), np.asarray(mononormal_tensor))

        # load patch groups
        groups = Image.open(str(scene_dir_path / groups_template.format(k=k)))
        groups_tensor = groups_trans_totensor(groups)
        groups_path = output_path / f"{out_index:06d}_groups.png"
        groups_tensor.save(str(groups_path))

        rgb_path = str(target_image.relative_to(output_path))
        frame = {
            "rgb_path": rgb_path,
            "camtoworld": pose.tolist(),
            "intrinsics": K.tolist(),
            "label_path": str(label_path.relative_to(output_path)),
            "group_path": str(groups_path.relative_to(output_path)),
            "mono_depth_path": rgb_path.replace("_rgb.png", "_depth.npy"),
            "mono_normal_path": rgb_path.replace("_rgb.png", "_normal.npy"),
            "sensor_depth_path": rgb_path.replace("_rgb.png",
                                                  "_sensor_depth.npy"),
        }

        frames.append(frame)
        out_index += 1

    # scene bbox for the scannet scene
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
    class_hist = class_hist.astype(np.float32)
    class_hist /= class_hist.sum()
    output_data["semantic_class_histogram"] = class_hist.tolist()

    output_data["frames"] = frames

    # save as json
    with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    # save camera path
    for scene_dir in scene_dirs:
        (Path(scene_dir) / "sdfstudio").mkdir(exist_ok=True)
        with open(Path(scene_dir) / "sdfstudio" / "camera_path.json",
                  "w",
                  encoding="utf-8") as f:
            json.dump(
                {
                    "render_height": H,
                    "render_width": W,
                    "camera_path": camera_path_per_scenedir[scene_dir],
                    "seconds": 5.0,
                },
                f,
                indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="preprocess scannet dataset to sdfstudio dataset")
    parser.add_argument('scenes', nargs='+')
    parser.add_argument('--size', default=384)
    parser.add_argument('--sampling', default=10)
    parser.add_argument('--scannetpose', default=False, action='store_true')
    parser.add_argument('--scannetlabel', action='store_true')
    flags = parser.parse_args()

    img_template = 'color/{k}.jpg'
    depth_template = 'depth/{k}.png'
    semantic_info = get_scannet_all()

    if bool(flags.scannetlabel):
        label_template='label-filt/{k}.png'
    else:
        label_template='pred_consensus/{k}.png'

    print(flags.scenes)

    sdfstudio_preprocessing(scene_dirs=flags.scenes,
                            image_size=int(flags.size),
                            sampling=int(flags.sampling),
                            img_template=img_template,
                            depth_template=depth_template,
                            label_template=label_template,
                            use_scannet_pose=bool(flags.scannetpose),
                            semantic_info=semantic_info)
