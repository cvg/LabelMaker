import os
import glob
import random
import argparse
import numpy as np
import open3d as o3d

def main(args):
    os.makedirs(args.output, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(args.scene_mesh)
    points = np.asarray(mesh.vertices).copy()
    mesh_labelled = o3d.geometry.TriangleMesh()
    mesh_labelled.vertices = mesh.vertices
    mesh_labelled.triangles = mesh.triangles
    
    label_index = []
    with open(os.path.join(os.path.join(args.mask3d, 'predictions.txt')), "r") as file:
        for line in file:
            parts = line.split() 
            if len(parts) >= 2: 
                label_index.append(int(parts[1]))
    
    color_list = random.sample([(r, g, b) for r in range(256) for g in range(256) for b in range(256)], len(label_index))
    colors = np.zeros((len(points), 3), dtype=np.uint8)

    prediction = np.zeros(len(points))

    pred_mask_mask3d = sorted(glob.glob(os.path.join(os.path.join(args.mask3d, 'pred_mask'), '*.txt')))
    for i, mask in enumerate(pred_mask_mask3d):

        with open(mask, "r") as file:
            for line_number, line in enumerate(file, start=1): 
                line = line.strip()  
                if line == "1":
                    colors[line_number - 1] = color_list[i]
                    prediction[line_number - 1] = label_index[i]

    np.savetxt(f'{str(args.output)}/labels_mask3d.txt', prediction, fmt="%d")
    mesh_labelled.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.)
    o3d.io.write_triangle_mesh(f'{str(args.output)}/mesh_mask3d.ply', mesh_labelled)


def arg_parser():
    parser = argparse.ArgumentParser(description='Create Mask3d mesh and label')
    parser.add_argument('--mask3d', type=str, default= './scene/scene_name/intermediate/scannet200_mask3d_1')
    parser.add_argument('--scene_mesh', type=str, default= './scene/scene_name/mesh.ply')
    parser.add_argument('--output', type=str, default= './scene/scene_name/mask3d_mesh')
    return parser.parse_args()

if __name__ == '__main__':
  args = arg_parser()
  main(args)