import os 

import open3d as o3d
import numpy as np


PATH = '/home/weders/scratch/scratch/scannetter'
SCENE = 'scene0000_00'

labels = np.loadtxt(os.path.join(PATH, SCENE, 'pred_consensus_5_scannet_labels_3d.txt'))
colors = np.loadtxt(os.path.join(PATH, SCENE, 'colors_consensus.txt'))

mesh = o3d.io.read_triangle_mesh(os.path.join(PATH, SCENE, 'scene0000_00_vh_clean.ply'))

## Load saved camera parameters

# Visualize with the loaded camera pose

vis = o3d.visualization.Visualizer()
vis.create_window()
ctr = vis.get_view_control()
param = o3d.io.read_pinhole_camera_parameters("camera_pose_2.json")
vis.add_geometry(mesh)
# ctr.convert_from_pinhole_camera_parameters(param)
# ctr = vis.get_view_control()
ctr.set_front([0, 0, -1])
ctr.set_lookat([0, 0, 0])
ctr.set_up([0, -1, 0])
ctr.set_zoom(0.5)

vis.run()
vis.destroy_window()