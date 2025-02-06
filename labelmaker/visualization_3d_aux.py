import os
import argparse

import numpy as np
import open3d as o3d

from labelmaker.label_data import get_nyu40, get_scannet200, get_wordnet, get_ade150

COLOR_MAPS = {
    'ade20k': get_ade150,
    'scannet200': get_scannet200,
    'nyu40': get_nyu40,
    'wordnet': get_wordnet,
    'consensus': get_wordnet,
    "sdfstudio": get_wordnet,
}

def read_mesh(scene_path):
    mesh_path = os.path.join(scene_path, 'mesh.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return mesh
  
def save_mesh(mesh, scene_path):
    mesh_path = os.path.join(scene_path, 'mesh_colored_aux.ply')
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    return mesh
  
def read_labels(scene_path):
  labels = np.loadtxt(os.path.join(scene_path, 'labels_aux.txt'))
  return labels

def colorize_labels(labels, color_map='consensus'):
  n = labels.shape[0]
  colors = np.zeros((n, 3))
  cmap = COLOR_MAPS[color_map]()
  
  for i in np.unique(labels):
    colors[labels == i] = cmap[int(i)]['color']
  
  colors = colors / 255.
  return colors
  
def colorize_mesh(mesh, colors):
  mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
  return mesh

def main(args):
  mesh = read_mesh(args.workspace)
  os.makedirs(os.path.join(args.workspace, args.output), exist_ok=True)
  labels = read_labels(args.workspace)
  colors = colorize_labels(labels)
  mesh = colorize_mesh(mesh, colors)
  save_mesh(mesh, os.path.join(args.workspace, args.output))

def arg_parser():
    parser = argparse.ArgumentParser(description='Lift 2D labels to 3D labels')
    parser.add_argument('--workspace', type=str)
    parser.add_argument('--output', type=str, default='vis_3d')
    return parser.parse_args()

if __name__ == '__main__':
  args = arg_parser()
  main(args)