import argparse
import os
import torch
import logging
import shutil
import gin
import cv2

import MinkowskiEngine as ME
import open3d as o3d
import numpy as np
import albumentations as A

from mask3d import get_model
from mask3d.datasets.scannet200.scannet200_constants import (
    SCANNET_COLOR_MAP_200,
    VALID_CLASS_IDS_200,
)

from copy import deepcopy
from pathlib import Path
from labelmaker.label_data import get_ade150
from tqdm import tqdm


logging.basicConfig(level="INFO")
logger = logging.getLogger('Mask 3D Mesh preprocessing')

@gin.configurable
def run_mask3d(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model()
    model = model.to(device)
    # model.eval()

    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    input_file = os.path.join(args.workspace, args.input)

    # load point cloud
    mesh = o3d.io.read_triangle_mesh(input_file)
    
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    colors = colors * 255.

    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

    coords = np.floor(points / 0.02)
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coords,
        features=colors,
        return_index=True,
        return_inverse=True)

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates,
                                                feats=features)
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )

    # run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)

    del data
    torch.cuda.empty_cache()

    # parse predictions
    logits = outputs["pred_logits"]
    masks = outputs["pred_masks"]

    # reformat predictions
    logits = logits[0].detach().cpu()


    masks = masks[0].detach().cpu()

    labels = []
    confidences = []
    masks_binary = []

    for i in range(len(logits)):
        p_labels = torch.softmax(logits[i], dim=-1)
        p_masks = torch.sigmoid(masks[:, i])
        l = torch.argmax(p_labels, dim=-1)
        c_label = torch.max(p_labels)
        m = p_masks > 0.5
        c_m = p_masks[m].sum() / (m.sum() + 1e-8)
        c = c_label * c_m
        if l < 200 and c > 0.5:
            labels.append(l.item())
            confidences.append(c.item())
            masks_binary.append(
                m[inverse_map]
            )  # mapping the mask back to the original point cloud

    # save labelled mesh
    mesh_labelled = o3d.geometry.TriangleMesh()
    mesh_labelled.vertices = mesh.vertices
    mesh_labelled.triangles = mesh.triangles
     
    labels_mapped = np.zeros((len(mesh.vertices), 1))
    colors_mapped = np.zeros((len(mesh.vertices), 3))

    for i, (l, c, m) in enumerate(zip(
        *sorted(zip(confidences, labels, masks_binary), reverse=False))):
        labels_mapped[m == 1] = l
        if l == 0:
            l_ = -1 + 2  # label offset is 2 for scannet 200, 0 needs to be mapped to -1 before (see trainer.py in Mask3D)
        else:
            l_ = l + 2
        # print(VALID_CLASS_IDS_200[l_], SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l_]], l_, CLASS_LABELS_200[l_])
        colors_mapped[m == 1] = SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l_]]

        # colors_mapped[mask_mapped == 1] = SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l]]
    
    output_dir = os.path.join(args.workspace, args.output)
    os.makedirs(output_dir, exist_ok=True)
    mesh_labelled.vertex_colors = o3d.utility.Vector3dVector(
        colors_mapped.astype(np.float32) / 255.)
    o3d.io.write_triangle_mesh(f'{output_dir}/mesh_labelled.ply',
                             mesh_labelled)

    mask_path = os.path.join(args.workspace, args.output, 'pred_mask')
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
        
    with open(os.path.join(args.workspace, args.output, 'predictions.txt'), 'w') as f:
        for i, (l, c, m) in enumerate(zip(
            *sorted(zip(confidences, labels, masks_binary), reverse=False))):
            mask_file = f'pred_mask/{str(i).zfill(3)}.txt'
            f.write(f'{mask_file} {VALID_CLASS_IDS_200[l]} {c}\n')
            np.savetxt(f'{args.workspace}/{args.output}/pred_mask/{str(i).zfill(3)}.txt', m.numpy(), fmt='%d')


@gin.configurable
def run_rendering(args, resolution=(192, 256)):
    
    scene_dir = Path(args.workspace)
    mask3d_path = Path(scene_dir / args.output)
    
    assert scene_dir.exists() and scene_dir.is_dir()
    
    prediction_file = mask3d_path / 'predictions.txt'

    if not prediction_file.exists():
        logger.error(f'No prediction file found in {scene_dir}')
        return
    
    with open(prediction_file) as f:
        instances = [x.strip().split(' ') for x in f.readlines()]

    # read mesh
    mesh_path = Path(scene_dir / args.input)
    assert mesh_path.exists()
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
    prediction_dir = scene_dir / 'pred_mask3d_rendered_ours'
    shutil.rmtree(prediction_dir, ignore_errors=True)
    prediction_dir.mkdir(exist_ok=False)

    keys = [x.stem for x in scene_dir.glob('pose/*.txt')]
    for k in tqdm(keys):
        cam_to_world = np.loadtxt(scene_dir / 'pose' / f'{k}.txt')
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = cam_to_world[:3, :3].T
        world_to_cam[:3, 3] = -world_to_cam[:3, :3] @ cam_to_world[:3, 3]

        intrinsics = np.loadtxt(scene_dir / 'intrinsic' / f'{k}.txt')
        
        
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            width_px=resolution[1],
            height_px=resolution[0],
            intrinsic_matrix=intrinsics[:3, :3],
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

        # print('saving', str(prediction_dir / f'{k}.png'))
        cv2.imwrite(str(mask3d_path / f'{k}.png'), semantic_segmentation)

def main(args):
    run_mask3d(args)
    run_rendering(args)


def arg_parser():
    parser = argparse.ArgumentParser(description='Mask3D Segmentation')
    parser.add_argument('--workspace',
                        type=str,
                        required=True,
                        help='Path to workspace directory')
    parser.add_argument(
        '--input',
        type=str,
        default='mesh.ply',
        help='Name of input directory in the workspace directory')
    parser.add_argument(
        '--output',
        type=str,
        default='intermediate/scannet200_mask3d_1',
        help=
        'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version'
    )
    parser.add_argument('--config', help='Name of config file')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
