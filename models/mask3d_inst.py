import argparse
import logging
import os
import shutil
from copy import deepcopy
from os.path import abspath, dirname, exists, join, relpath
from pathlib import Path
from typing import Union

import albumentations as A
import cv2
import gin
import mask3d.conf
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
from hydra.experimental import compose, initialize
from mask3d import InstanceSegmentation
from mask3d.datasets.scannet200.scannet200_constants import SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200
from mask3d.utils.utils import (
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
    load_checkpoint_with_missing_or_exsessive_keys,
)
from omegaconf import OmegaConf
from tqdm import tqdm

from labelmaker.label_data import get_ade150

logging.basicConfig(level="INFO")
logger = logging.getLogger('Mask 3D Mesh preprocessing')


def get_model(checkpoint_path: str):
  # Initialize the directory with config files

  # get the config from file
  conf_path = relpath(dirname(mask3d.conf.__file__),
                      start=abspath(dirname(__file__)))
  print(conf_path, abspath(__file__))
  with initialize(config_path=conf_path):
    # Compose a configuration
    cfg = compose(config_name="config_base_instance_segmentation.yaml")
    # print(OmegaConf.to_yaml(cfg))

  # these are copied from official config of scannet_val
  # # general
  cfg.general.checkpoint = checkpoint_path
  cfg.general.num_targets = 201
  cfg.general.train_mode = False
  cfg.general.eval_on_segments = True
  cfg.general.topk_per_image = 300
  cfg.general.use_dbscan = True
  cfg.general.dbscan_eps = 0.95
  cfg.general.export_threshold = 0.001

  # # data
  cfg.data.num_labels = 200
  cfg.data.test_mode = "test"

  # # model
  cfg.model.num_queries = 150

  model = InstanceSegmentation(cfg)

  if cfg.general.backbone_checkpoint is not None:
    cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
        cfg, model)
  if cfg.general.checkpoint is not None:
    cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

  return model


def run_mask3d(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    device: Union[str, torch.device] = 'cpu',
):
  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  assert scene_dir.exists() and scene_dir.is_dir()

  model_ckpt = abspath(
      join(__file__, '../../checkpoints/mask3d_scannet200_benchmark.ckpt'))
  model = get_model(checkpoint_path=model_ckpt)
  model = model.to(device)
  # model.eval()

  color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
  color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
  normalize_color = A.Normalize(mean=color_mean, std=color_std)

  # load point cloud
  input_mesh_path = str(scene_dir / 'mesh.ply')
  mesh = o3d.io.read_triangle_mesh(input_mesh_path)

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
      return_inverse=True,
  )

  sample_coordinates = coords[unique_map]
  coordinates = [torch.from_numpy(sample_coordinates).int()]
  sample_features = colors[unique_map]
  features = [torch.from_numpy(sample_features).float()]

  coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
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
          m[inverse_map])  # mapping the mask back to the original point cloud

  # save labelled mesh
  mesh_labelled = o3d.geometry.TriangleMesh()
  mesh_labelled.vertices = mesh.vertices
  mesh_labelled.triangles = mesh.triangles

  labels_mapped = np.zeros((len(mesh.vertices), 1))
  colors_mapped = np.zeros((len(mesh.vertices), 3))

  for i, (l, c, m) in enumerate(
      sorted(zip(labels, confidences, masks_binary), reverse=False)):
    labels_mapped[m == 1] = l
    if l == 0:
      l_ = -1 + 2  # label offset is 2 for scannet 200, 0 needs to be mapped to -1 before (see trainer.py in Mask3D)
    else:
      l_ = l + 2
    # print(VALID_CLASS_IDS_200[l_], SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l_]], l_, CLASS_LABELS_200[l_])
    colors_mapped[m == 1] = SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l_]]

    # colors_mapped[mask_mapped == 1] = SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l]]

  output_dir = scene_dir / output_folder
  shutil.rmtree(output_dir, ignore_errors=True)
  os.makedirs(str(output_dir), exist_ok=False)

  mesh_labelled.vertex_colors = o3d.utility.Vector3dVector(
      colors_mapped.astype(np.float32) / 255.)
  o3d.io.write_triangle_mesh(
      f'{str(output_dir)}/mesh_labelled.ply',
      mesh_labelled,
  )

  # mask_path = os.path.join(args.scene_dir, args.output, 'pred_mask')
  # if not os.path.exists(mask_path):
  #   os.makedirs(mask_path)

  mask_path = output_dir / 'pred_mask'
  mask_path.mkdir(exist_ok=True)

  with open(str(output_dir / 'predictions.txt'), 'w') as f:
    for i, (l, c, m) in enumerate(
        sorted(zip(labels, confidences, masks_binary), reverse=False)):
      mask_file = f'pred_mask/{str(i).zfill(3)}.txt'
      f.write(f'{mask_file} {VALID_CLASS_IDS_200[l]} {c}\n')
      np.savetxt(
          f'{str(scene_dir)}/{str(output_folder)}/pred_mask/{str(i).zfill(3)}.txt',
          m.numpy(),
          fmt='%d')


def run_rendering(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    resolution=(192, 256),
):

  scene_dir = Path(scene_dir)
  output_folder = Path(output_folder)

  assert scene_dir.exists() and scene_dir.is_dir()

  input_pose_folder = scene_dir / 'pose'
  assert input_pose_folder.exists() and input_pose_folder.is_dir()

  output_dir = scene_dir / output_folder

  prediction_file = output_dir / 'predictions.txt'
  if not prediction_file.exists():
    logger.error(f'No prediction file found in {scene_dir}')
    return

  with open(prediction_file) as f:
    instances = [x.strip().split(' ') for x in f.readlines()]

  # read mesh
  mesh_path = scene_dir / "mesh.ply"
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
    filepath = output_dir / inst[0]
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
      segmentation[occluding_previous_pred] = len(pixelid_to_instance) - 1
      rendered_distance[occluding_previous_pred] = vis['t_hit'].numpy(
      )[occluding_previous_pred]
      mask = np.logical_and(mask, np.logical_not(occluding_previous_pred))
      # now check if this instance gets occluded
      occluded_by_previous_pred = np.logical_and(
          rendered_distance <= vis['t_hit'].numpy() + 0.05, rendered_distance
          != 0)
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
              mask,
              segmentation == overlapping_id)] = len(pixelid_to_instance) - 1
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

    cv2.imwrite(str(output_dir / f'{k}.png'), semantic_segmentation)


@gin.configurable
def run(
    scene_dir: Union[str, Path],
    output_folder: Union[str, Path],
    device: Union[str, torch.device] = 'cpu',
    render_resolution=(192, 256),
):
  run_mask3d(
      scene_dir=scene_dir,
      output_folder=output_folder,
      device=device,
  )
  run_rendering(
      scene_dir=scene_dir,
      output_folder=output_folder,
      resolution=render_resolution,
  )


def arg_parser():
  parser = argparse.ArgumentParser(description='Mask3D Segmentation')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help=
      'Path to workspace directory. There should be a "mesh.ply" file and "pose" folder inside.',
  )
  parser.add_argument(
      '--output',
      type=str,
      default='intermediate/scannet200_mask3d_1',
      help=
      'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version.'
  )
  parser.add_argument('--config', help='Name of config file')
  return parser.parse_args()


if __name__ == "__main__":
  args = arg_parser()
  gin.parse_config_file(args.config)
  run(scene_dir=args.workspace, output_folder=args.output)
