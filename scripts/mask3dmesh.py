import sys, os
import matplotlib.pyplot as plt
import logging
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
from copy import deepcopy
from plyfile import PlyData, PlyElement
from segmentation_tools.label_data import get_ade150
import shutil

logging.basicConfig(level="INFO")
log = logging.getLogger('Mask 3D Mesh preprocessing')


def read_mesh_vertices(filename):
    """
    PLY mesh reader from ScanNet repository. This is the only one that can read the weird
    quad-mesh file format.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        # convert quad faces into triangles
        triangles = np.zeros(shape=[plydata['face'].count * 2, 3],
                             dtype=np.int32)
        for i, face in enumerate(plydata['face'].data):
            triangles[2 * i, :] = [face[0][0], face[0][1], face[0][2]]
            triangles[2 * i + 1, :] = [face[0][0], face[0][2], face[0][3]]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def convert_scene(scene_dir, keys, mesh_path, mask3d_path, intrinsic, resolution, quadmesh=True):
    scene_dir = Path(scene_dir)
    mask3d_path = Path(mask3d_path)
    assert scene_dir.exists() and scene_dir.is_dir()
    prediction_file = scene_dir / 'mask3d.txt'

    if not prediction_file.exists():
        prediction_file = next(mask3d_path.glob('*3dod_mesh.txt'))

    if not prediction_file.exists():
        prediction_file = next(mask3d_path.glob('*.txt')).absolute()
    if not prediction_file.exists():
        log.error(f'No prediction file found in {scene_dir}')
        return
    
    with open(prediction_file) as f:
        instances = [x.strip().split(' ') for x in f.readlines()]

    # read mesh
    mesh_path = Path(mesh_path)
    assert mesh_path.exists()
    if quadmesh:
        mesh = read_mesh_vertices(str(mesh_path))
    else:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    labelinfo = get_ade150()
    colors = np.array([x['color'] for x in labelinfo])
    colors = colors / 255.0

    objects = []
    scenes = []
    # render = o3d.visualization.rendering.OffscreenRenderer(640, 480)
    geoid_to_classid = {}

    for i, inst in enumerate(instances):
        # check confidence
        if float(inst[2]) < 0.5:
            continue
        scene = o3d.t.geometry.RaycastingScene()
        filepath = mask3d_path / inst[0]
        mask = np.loadtxt(filepath).astype(bool)
        obj = deepcopy(mesh)
        obj.remove_vertices_by_mask(np.logical_not(mask))
        obj.paint_uniform_color(colors[int(inst[1]) % 150])
        # obj.paint_uniform_color((0.5, 0.5, 0.00001 * int(inst[1])))
        objects.append(obj)
        obj_in_scene = o3d.t.geometry.TriangleMesh.from_legacy(obj)
        # print(inst[1])
        geoid_to_classid[i] = int(inst[1])
        # render.scene.add_geometry(f"object{i}", obj, materials[int(inst[1])])
        scene.add_triangles(obj_in_scene)
        scenes.append(scene)
    # o3d.visualization.draw_geometries(objects)
    prediction_dir = scene_dir / 'pred_mask3d_rendered'
    shutil.rmtree(prediction_dir, ignore_errors=True)
    prediction_dir.mkdir(exist_ok=False)

    for k in tqdm(keys):
        cam_to_world = np.loadtxt(scene_dir / 'pose' / f'{k}.txt')
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = cam_to_world[:3, :3].T
        world_to_cam[:3, 3] = -world_to_cam[:3, :3] @ cam_to_world[:3, 3]

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            width_px=resolution[1],
            height_px=resolution[0],
            intrinsic_matrix=intrinsic[:3, :3],
            extrinsic_matrix=world_to_cam,  # world to camera
        )
        pixelid_to_instance = []
        segmentation = -1 * np.ones(resolution).astype(int)
        rendered_distance = np.zeros(resolution)
        for i, scene in enumerate(scenes):
            vis = scene.cast_rays(rays)
            geometry_ids = vis['geometry_ids'].numpy().astype(int)
            pixelid_to_instance.append([i])
            mask = geometry_ids == 0
            # check if this instance occludes a previous instance
            occluding_previous_pred = np.logical_and(
                rendered_distance > vis['t_hit'].numpy() + 0.05, mask)
            segmentation[occluding_previous_pred] = len(
                pixelid_to_instance) - 1
            rendered_distance[occluding_previous_pred] = vis['t_hit'].numpy(
            )[occluding_previous_pred]
            mask = np.logical_and(mask,
                                  np.logical_not(occluding_previous_pred))
            # now check if this instance gets occluded
            occluded_by_previous_pred = np.logical_and(
                rendered_distance <= vis['t_hit'].numpy() + 0.05,
                rendered_distance != 0)
            mask[occluded_by_previous_pred] = False
            # now deal with the case where there is no overlap with other ids
            update_mask = np.logical_and(mask, segmentation == -1)
            segmentation[update_mask] = len(pixelid_to_instance) - 1
            rendered_distance[update_mask] = vis['t_hit'].numpy()[update_mask]
            mask[update_mask] = False
            # finally, there are cases where already another instance was rendered at the same position
            for overlapping_id in np.unique(segmentation[np.logical_and(
                    mask, segmentation != -1)]):
                # check if this already overlaps with something else
                if len(pixelid_to_instance[overlapping_id]) > 1:
                    # merge
                    pixelid_to_instance[overlapping_id] = list(
                        set(pixelid_to_instance[overlapping_id] + [i]))
                else:
                    # new multi-instance
                    pixelid_to_instance.append(
                        [pixelid_to_instance[overlapping_id][0], i])
                    segmentation[np.logical_and(
                        mask, segmentation ==
                        overlapping_id)] = len(pixelid_to_instance) - 1
        semantic_segmentation = np.zeros(resolution).astype(int)
        for i, ids in enumerate(pixelid_to_instance):
            if len(ids) == 1:
                semantic_segmentation[segmentation == i] = instances[ids[0]][1]
            else:
                max_confidence = -1
                max_id = -1
                for j in ids:
                    if float(instances[j][2]) > max_confidence:
                        max_confidence = float(instances[j][2])
                        max_id = instances[j][1]
                semantic_segmentation[segmentation == i] = max_id
        cv2.imwrite(str(prediction_dir / f'{k}.png'), semantic_segmentation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replica', action='store_true')
    parser.add_argument('--arkitscenes', action='store_true')
    parser.add_argument('scene', type=str)
    flags = parser.parse_args()
    scene_dir = Path(flags.scene)
    if flags.replica:
        keys = sorted(
            int(x.name.split('.')[0].split('_')[1])
            for x in (scene_dir / 'rgb').iterdir())
    else:
        keys = sorted(
            int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
    if flags.replica:
        room_dir = scene_dir.parent
        mesh_path = next(
            x for x in room_dir.iterdir()
            if x.name.startswith('mesh_semantic') and x.name.endswith('.ply'))
        intrinsic = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
        resolution = (480, 640)
    elif flags.arkitscenes:
        room_dir = scene_dir
        mesh_path = next(
            x for x in room_dir.iterdir()
            if x.name.endswith('3dod_mesh.ply') )
        intrinsic = np.loadtxt(scene_dir / 'intrinsic' / 'intrinsic_color.txt')
        resolution = (480, 640)

    else:
        room_dir = scene_dir
        mesh_path = next(
            x for x in room_dir.iterdir()
            if x.name.endswith('vh_clean_2.ply') )
        intrinsic = np.loadtxt(scene_dir / 'intrinsic' / 'intrinsic_color.txt')
        resolution = (968, 1296)
    assert mesh_path.exists()
    convert_scene(flags.scene, keys, mesh_path,
                  room_dir / 'pred_mask3d', intrinsic, resolution, quadmesh=flags.replica)
