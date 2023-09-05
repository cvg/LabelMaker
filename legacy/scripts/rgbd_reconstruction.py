import sys, os
import open3d as o3d
from PIL import Image
import json

import logging
from joblib import Parallel, delayed
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level="INFO")
log = logging.getLogger('Depth to HHA conversion')


def integrate_frames(scene_dir,
                     keys,
                     intrinsics_matrix,
                     depth_template='depth/{k}.png',
                     img_template='color/{k}.png',
                     pose_template='pose/{k}.txt',
                     depth_scale=1000.0,
                     n_jobs=8):
    log.info(f'running depth to hha conversion for scene {scene_dir}')
    scene_dir = Path(scene_dir)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.000001,
        # voxel_length=4.0 / 512.0,
        # sdf_trunc=0.04,
        sdf_trunc=0.0000001,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    first_image = Image.open(str(scene_dir / img_template.format(k=keys[0])))
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=first_image.width,
        height=first_image.height,
        intrinsic_matrix=intrinsics_matrix)

    pointclouds = []
    for k in tqdm(keys[::10]):
        assert (scene_dir / img_template.format(k=k)).exists()
        color = o3d.io.read_image(str(scene_dir / img_template.format(k=k)))
        assert (scene_dir / depth_template.format(k=k)).exists()
        if depth_template.endswith('.npy'):
            depth = np.load(str(scene_dir / depth_template.format(k=k)))
            depth = o3d.geometry.Image(depth)
        else:
            depth = o3d.io.read_image(
                str(scene_dir / depth_template.format(k=k)))
        cam_to_world = np.loadtxt(scene_dir / pose_template.format(k=k))
        world_to_cam = np.linalg.inv(cam_to_world)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_trunc=10.0,
            depth_scale=depth_scale,
            convert_rgb_to_intensity=False)
        # volume.integrate(rgbd, intrinsics, world_to_cam)
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsics, world_to_cam)
        pointclouds.append(pc)
    # mesh = volume.extract_triangle_mesh()
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    o3d.visualization.draw_geometries(pointclouds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--replica', default=False)
    parser.add_argument('--sdfstudio', default=False)
    parser.add_argument('--j', default=8)
    flags = parser.parse_args()
    scene_dir = Path(flags.scene)
    assert scene_dir.exists() and scene_dir.is_dir()
    if flags.replica:
        keys = sorted(
            int(x.name.split('.')[0].split('_')[1])
            for x in (scene_dir / 'rgb').iterdir())
        img_template = 'rgb/rgb_{k}.png'
        # focal length is just guess-copied from scannet
        intrinsics = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
        depth_template = 'depth/depth_{k}.png'
        pose_template = 'pose/{k}.txt'
        # depth is already complete
        depth_completion_template = depth_template
        depth_scale = 1000.0
    elif flags.sdfstudio:
        keys = sorted(
            int(x.name.split('.')[0].split('_')[0])
            for x in (scene_dir / 'sdfstudio').iterdir()
            if x.name.endswith('rgb.png'))
        img_template = 'sdfstudio/{k:06d}_rgb.png'
        depth_template = 'sdfstudio/{k:06d}_sensor_depth.npy'
        pose_template = 'sdfstudio/{k:06d}_camtoworld.txt'
        with open(scene_dir / 'sdfstudio' / 'meta_data.json') as f:
            metadata = json.load(f)
            intrinsics = np.array(metadata['frames'][0]['intrinsics'])[:3, :3]
        depth_completion_template = 'omnidata_depth/{k}.png'
        depth_scale = 1.0
    else:
        keys = sorted(
            int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
        img_template = 'color/{k}.png'
        intrinsics = np.loadtxt(
            str(scene_dir / 'intrinsic/intrinsic_depth.txt'))[:3, :3]
        depth_template = 'depth/{k}.png'
        pose_template = 'pose/{k}.txt'
        depth_completion_template = 'omnidata_depth/{k}.png'
        depth_scale = 1000.0
    integrate_frames(scene_dir,
                     keys,
                     intrinsics,
                     depth_template=depth_template,
                     img_template=img_template,
                     pose_template=pose_template,
                     depth_scale=depth_scale,
                     n_jobs=flags.j)
