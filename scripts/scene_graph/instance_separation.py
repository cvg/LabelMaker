import warnings
warnings.filterwarnings('ignore')  

import random
import hdbscan
import argparse
import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
from collections import defaultdict

def main(args):

    STUFF_OBJECTS = ['floor', 'wall', 'ceiling']
    df = pd.read_csv(Path(args.mapping_file))
    
    mesh_path = Path(args.workspace) / 'point_lifted_mesh.ply'

    assert mesh_path.exists(), f"mesh file {mesh_path} not found"
    pcd = o3d.io.read_point_cloud(str(mesh_path)) 
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) 
    
    label_path = Path(args.workspace) / 'labels.txt'
    assert label_path.exists(), f"Label file {label_path} not found"
    labels = np.loadtxt(str(label_path), dtype=int)
    assert len(labels) == len(pcd.points), "Mismatch between points and labels"

    print("Processing ...")

    color_instances = defaultdict(lambda: {'points': [], 'indices': []})
    for i, color in enumerate(colors):
        color_tuple = tuple((color * 255).astype(int))
        if color_tuple != (0,0,0): 
            color_instances[color_tuple]['points'].append(points[i])
            color_instances[color_tuple]['indices'].append(i)

    color_to_label = {tuple(np.round(color * 255).astype(int)): label for color, label in zip(colors, labels)}
    color_list = random.sample([(r, g, b) for r in range(256) for g in range(256) for b in range(256)], len(np.unique(colors, axis=0)) * 10)
    color_final = np.zeros((len(points), 3), dtype=np.uint8)
    
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points)

    counter = 0 
    for color, data in color_instances.items():
        points_array = np.array(data['points'])
        point_indices = np.array(data['indices'])

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points_array)
        pcd2_colors = np.zeros((points_array.shape[0], 3))

        category = df.loc[df[args.mapping_column] == color_to_label.get(color, "Unknown"), "category"] 
        if category.iloc[0] in STUFF_OBJECTS:
            color_final[point_indices] = [0, 0, 0]
        
        elif points_array.shape[0] > 100:  
            clusterer = hdbscan.HDBSCAN(
            min_cluster_size=100,
            min_samples=10,
            cluster_selection_epsilon=0.1
            )
            cluster_labels = clusterer.fit_predict(points_array)

            unique_clusters = set(cluster_labels)
            for i, cluster_id in enumerate(unique_clusters):
                if cluster_id != -1: 
                    cluster_mask = cluster_labels == cluster_id
                    cluster_points = points_array[cluster_mask]
                    cluster_indices = point_indices[cluster_mask]
                    
                    if cluster_points.shape[0] < 50:
                        color_final[cluster_indices] = [0, 0, 0]  
                        pcd2_colors[cluster_mask] = [0, 0, 0]
                        continue
                    
                    pcd2_colors[cluster_mask] = np.array(color_list[counter + i]) / 255.0 
                    color_final[cluster_indices] = color_list[counter + i]
                    counter += 1
            pcd2.colors = o3d.utility.Vector3dVector(pcd2_colors)

    pcd_new.colors = o3d.utility.Vector3dVector(color_final / 255.0 )
    print('Processing done!')
    o3d.io.write_point_cloud(args.save_path, pcd_new)
    o3d.visualization.draw_geometries([pcd_new])
    

def arg_parser():
    parser = argparse.ArgumentParser(description='Instance Separation')
    parser.add_argument('--workspace', type=str)
    parser.add_argument('--mapping_column', type=str, default='wn199-merged-v2')
    parser.add_argument('--mapping_file', type=str, default='../../labelmaker/mappings/label_mapping.csv')
    parser.add_argument('--save_path', type=str, default="./mesh_instances.ply")

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    main(args)
