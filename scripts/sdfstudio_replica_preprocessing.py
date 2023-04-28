import argparse
import glob
import json
import os
from pathlib import Path
import logging
from PIL import Image
import shutil
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms

logging.basicConfig(level="INFO")
log = logging.getLogger('sdfstudio data preprocessing')


def sdfstudio_preprocessing(scene_dir,
                            image_size=384,
                            img_template='rgb/{k}.jpg',
                            depth_template='depth/{k}.png',
                            sampling=1):
    scene_dir = Path(scene_dir)
    assert scene_dir.exists(), f"scene_dir {scene_dir} does not exist"
    # output_path = Path(args.output_path)  # "data/custom/scannet_scene0050_00"
    # input_path = Path(args.input_path)  # "/home/yuzh/Projects/datasets/scannet/scene0050_00"
    files = glob.glob(str(scene_dir / img_template.format(k='*')), recursive=True)
    keys = sorted(
        int(re.search(img_template.format(k='(\d+)'), x).group(1))
        for x in files)

    color_paths = list(scene_dir / img_template.format(k=k) for k in keys)
    depth_paths = list(scene_dir / depth_template.format(k=k) for k in keys)

    original_image_dim = Image.open(color_paths[0]).size
    image_crop_size = min(original_image_dim)
    original_depth_dim = Image.open(depth_paths[0]).size
    depth_crop_size = min(original_depth_dim)
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


    # # load color
    # color_path = input_path / "frames" / "color"
    # color_paths = sorted(glob.glob(os.path.join(color_path, "*.jpg")), key=lambda x: int(os.path.basename(x)[:-4]))

    # # load depth
    # depth_path = input_path / "frames" / "depth"
    # depth_paths = sorted(glob.glob(os.path.join(depth_path, "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))

    # load intrinsic
    # intrinsic_path = input_path / "frames" / "intrinsic" / "intrinsic_color.txt"
    camera_intrinsic = np.loadtxt(scene_dir / 'intrinsic' /
                                  'intrinsic_color.txt')

    # load pose
    poses = []
    for k in keys:
        c2w = np.loadtxt(scene_dir / 'pose' / f'{k}.txt')
        poses.append(c2w)
    poses = np.array(poses)

    # deal with invalid poses
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

    center = (min_vertices + max_vertices) / 2.0
    scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
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

    K = camera_intrinsic

    frames = []
    out_index = 0
    output_path = scene_dir / 'sdfstudio'
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(exist_ok=False)
    for idx, (valid, pose, image_path, depth_path) in enumerate(
            zip(valid_poses, poses, color_paths, depth_paths)):

        if idx % sampling != 0:
            continue
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

        rgb_path = str(target_image.relative_to(output_path))
        frame = {
            "rgb_path": rgb_path,
            "camtoworld": pose.tolist(),
            "intrinsics": K.tolist(),
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
        "pairs": None,
        "worldtogt": scale_mat.tolist(),
        "scene_box": scene_box,
    }

    output_data["frames"] = frames

    # save as json
    with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="preprocess scannet dataset to sdfstudio dataset")
    parser.add_argument('scene')
    parser.add_argument('--replica', default=False)
    parser.add_argument('--size', default=384)
    parser.add_argument('--sampling', default=10)
    flags = parser.parse_args()

    if flags.replica:
        img_template = 'rgb/rgb_{k}.png'
        depth_template = 'depth/depth_{k}.png'
    else:
        img_template = 'color/{k}.jpg'
        depth_template = 'depth/{k}.png'

    sdfstudio_preprocessing(scene_dir=flags.scene,
                            image_size=flags.size,
                            sampling=int(flags.sampling),
                            img_template=img_template,
                            depth_template=depth_template)
