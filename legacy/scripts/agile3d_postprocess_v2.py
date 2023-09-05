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
from segmentation_tools.label_data import get_wordnet
import shutil
import skimage

o3d.visualization.webrtc_server.enable_webrtc()

logging.basicConfig(level="INFO")
log = logging.getLogger('Agile3D postprocessing')

WORDNET_UNKNOWNS = [
    'shovel', 'mammal', 'sponge', 'soda_stream', 'tetra_pack', 'can', 'magnet',
    'vessel', 'whiteboard', 'bike'
]

WORDNET_SPECIAL = {
    'books': 'book.n.11',
    'purse': 'bag.n.04',
    'bin': 'ashcan.n.01',
    'remote': 'remote_control.n.01',
    'shoes': 'shoe.n.01',
    'painting': 'picture.n.01', # mergings
    'rug': 'mat.n.01', # mergings
    'hanging': 'picture.n.01', # mergings
    'rack': 'bookshelf.n.01', # mergings
}


def parse_objectclass(name):
    # remove fileending from name
    name = name.split('.')[0]
    # remove object_ prefix
    if name.startswith('object_'):
        name = name[7:]
    # remove trailing numbers
    while name[-1].isdigit():
        name = name[:-1]
    wn = get_wordnet()
    # for v in wn:
    #     print(v)
    name_to_wnid = {v['name'].split('.')[0]: v['id'] for v in wn}

    # for k, v in name_to_wnid.items():
    #     print(k, v)

    for unknown in WORDNET_UNKNOWNS:
        name_to_wnid[unknown] = name_to_wnid['unknown']
    for k, v in WORDNET_SPECIAL.items():
        name_to_wnid[k] = next(x['id'] for x in wn if x['name'] == v)
    classid = name_to_wnid[name]
    return classid


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


def convert_scene(scene_dir, keys, agile3d_path):
    scene_dir = Path(scene_dir)
    agile3d_path = Path(agile3d_path)
    assert scene_dir.exists() and scene_dir.is_dir()
    instances = list(
        x.name for x in agile3d_path.iterdir()
        if x.name.startswith('object_') and x.name.endswith('.npy'))

    # read mesh
    mesh_path = agile3d_path / '3dpoints.ply'
    assert mesh_path.exists()
    # mesh = read_mesh_vertices(str(mesh_path))
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    pcd = o3d.io.read_point_cloud(str(mesh_path))
    print(f"pcd shape: {np.asarray(pcd.points).shape}")

    labelinfo = get_wordnet()
    colors = np.random.uniform(0, 1, (len(instances), 3))
    pcd_colors = np.asarray(pcd.colors)
    point_label_counter = np.zeros(np.asarray(pcd.points).shape[0], dtype=int)
    point_label = np.zeros(np.asarray(pcd.points).shape[0], dtype=int)
    point_label

    # make sure we only use points that are labelled once
    for i, inst in enumerate(instances):
        mask = np.load(str(agile3d_path / inst)) == 1
        label_id = int(parse_objectclass(inst))
        counter_mask = np.logical_and(point_label != label_id, mask)
        point_label_counter[counter_mask] += 1
        point_label[mask] = int(parse_objectclass(inst))



    # remove points that are labelled multiple times
    point_label[point_label_counter > 1] = 0
    np.savetxt(str(scene_dir / 'agile3d_wn_point_label.txt'), point_label)
    pcd_colors[:] = 0.5
    objects = []
    mesh_objects = []
    scenes = []
    geoid_to_classid = {}
    
    point_colors = np.zeros(np.asarray(pcd.points).shape)
    for i, inst in enumerate(np.unique(point_label)):
        if inst > 0:
            print(inst)
            m_ = point_label == inst
            point_colors[m_] = colors[i]
            break
    
    scene = o3d.t.geometry.RaycastingScene()

    for i, inst in enumerate(instances):
        mask = np.load(str(agile3d_path / inst)) == 1
        mask = np.logical_and(mask, point_label_counter == 1)
        # print(f"mask shape: {mask.shape}, mask size {np.sum(mask)}, color {colors[i]}")
        pcd_colors[mask] = colors[i]
        # obj_pcd = o3d.geometry.PointCloud()
        # obj_pcd.points = o3d.utility.Vector3dVector(
        #     np.asarray(pcd.points)[mask])
        # obj_pcd.normals = o3d.utility.Vector3dVector(
        #     np.asarray(mesh.vertex_normals)[mask])
        # obj.colors = o3d.utility.Vector3dVector(np.tile(colors[i][None, :], (np.sum(mask), 1)))
        # obj.paint_uniform_color((0.5, 0.5, 0.00001 * int(inst[1])))
        obj = deepcopy(mesh)
        obj.remove_vertices_by_mask(np.logical_not(mask))
        obj.paint_uniform_color(colors[i])
        # objects.append(obj)
        obj_in_scene = o3d.t.geometry.TriangleMesh.from_legacy(obj)
        # print(inst[1])

        if 'normals' not in obj_in_scene.triangle:
            print('Skipping', inst, 'because it has no triangles')
            continue

        # render.scene.add_geometry(f"object{i}", obj, materials[int(inst[1])])
        inst_id = scene.add_triangles(obj_in_scene)
        geoid_to_classid[inst_id] = int(parse_objectclass(inst))
        del obj
    #     scenes.append(scene)

    # scene = o3d.t.geometry.RaycastingScene()
    # obj = deepcopy(mesh)
    # obj.vertex_colors = o3d.utility.Vector3dVector(point_colors)
    # obj_in_scene = o3d.t.geometry.TriangleMesh.from_legacy(obj)
    # scene.add_triangles(obj_in_scene)
    scenes.append(scene)



    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries(objects)
    prediction_dir = scene_dir / 'label_agile3d'
    shutil.rmtree(prediction_dir, ignore_errors=True)
    prediction_dir.mkdir(exist_ok=False)

    intrinsic = np.loadtxt(scene_dir / 'intrinsic' / 'intrinsic_color.txt')
    for k in tqdm(keys):
        cam_to_world = np.loadtxt(scene_dir / 'pose' / f'{k}.txt')
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = cam_to_world[:3, :3].T
        world_to_cam[:3, 3] = -world_to_cam[:3, :3] @ cam_to_world[:3, 3]

        img_resolution = (968, 1296)

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            width_px=img_resolution[1],
            height_px=img_resolution[0],
            intrinsic_matrix=intrinsic[:3, :3],
            extrinsic_matrix=world_to_cam,  # world to camera
        )
        pixelid_to_instance = []
        segmentation = -1 * np.ones(img_resolution).astype(int)
        rendered_distance = np.zeros(img_resolution)

        vis = scenes[0].cast_rays(rays)

        geometry_ids = vis['geometry_ids'].numpy().astype(int)
        pixelid_to_instance.append([i])

            # mask = geometry_ids == 0
            # # check if this instance occludes a previous instance
            # occluding_previous_pred = np.logical_and(
            #     rendered_distance > vis['t_hit'].numpy() + 0.05, mask)
            # segmentation[occluding_previous_pred] = len(
            #     pixelid_to_instance) - 1
            # rendered_distance[occluding_previous_pred] = vis['t_hit'].numpy(
            # )[occluding_previous_pred]
            # mask = np.logical_and(mask,
            #                       np.logical_not(occluding_previous_pred))
            # # now check if this instance gets occluded
            # occluded_by_previous_pred = np.logical_and(
            #     rendered_distance <= vis['t_hit'].numpy() + 0.05,
            #     rendered_distance != 0)
            # mask[occluded_by_previous_pred] = False
            # # now deal with the case where there is no overlap with other ids
            # update_mask = np.logical_and(mask, segmentation == -1)
            # segmentation[update_mask] = len(pixelid_to_instance) - 1
            # rendered_distance[update_mask] = vis['t_hit'].numpy()[update_mask]
            # mask[update_mask] = False
            # # finally, there are cases where already another instance was rendered at the same position
            # for overlapping_id in np.unique(segmentation[np.logical_and(
            #         mask, segmentation != -1)]):
            #     # check if this already overlaps with something else
            #     if len(pixelid_to_instance[overlapping_id]) > 1:
            #         # merge
            #         pixelid_to_instance[overlapping_id] = list(
            #             set(pixelid_to_instance[overlapping_id] + [i]))
            #     else:
            #         # new multi-instance
            #         pixelid_to_instance.append(
            #             [pixelid_to_instance[overlapping_id][0], i])
            #         segmentation[np.logical_and(
            #             mask, segmentation ==
            #             overlapping_id)] = len(pixelid_to_instance) - 1
        semantic_segmentation = np.zeros(img_resolution).astype(int)
        for i, id in enumerate(np.unique(geometry_ids)):
            if id == 4294967295:
                continue
            semantic_segmentation[geometry_ids == id] = geoid_to_classid[id]
            # else:
            #     max_confidence = -1
            #     max_id = -1
            #     for j in ids:
            #         if float(instances[j][2]) > max_confidence:
            #             max_confidence = float(instances[j][2])
            #             max_id = instances[j][1]
            #     semantic_segmentation[segmentation == i] = max_id

        pred_sam = cv2.imread(str(scene_dir / 'pred_sam' / f'{k}.png'),
                              cv2.IMREAD_UNCHANGED)
        # go through all SAM segments and check if they match an agile3d segment
        for i in np.unique(pred_sam):
            # split the sam masks into connected components
            masks = skimage.measure.label(pred_sam == i)
            # for each connected component, check if it matches an agile3d segment
            # skip the first segment since this is background
            for j in np.unique(masks)[1:]:
                mask = masks == j
                bins = np.bincount(semantic_segmentation[mask])
                bins = bins[1:]  # remove unknown
                if bins.size == 0:
                    break
                max_fraction = np.max(bins) / np.sum(bins)
                if max_fraction > 0.95:
                    semantic_segmentation[mask] = np.argmax(bins) + 1
        cv2.imwrite(str(prediction_dir / f'{k}.png'), semantic_segmentation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', type=str)
    parser.add_argument('agile3d', type=str)
    flags = parser.parse_args()
    scene_dir = Path(flags.scene)
    agile3d_path = Path(flags.agile3d)
    keys = sorted(
        int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
    convert_scene(flags.scene, keys, agile3d_path)
