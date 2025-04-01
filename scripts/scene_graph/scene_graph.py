import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
import open3d as o3d
import networkx as nx
from pathlib import Path
from collections import defaultdict
from scipy.spatial import cKDTree


def get_spatial_relationship(bbox1, bbox2, height_threshold=0.2, proximity_threshold=0.5):
    """Determine the spatial relationship between two bounding boxes."""
    center1 = (bbox1[0] + bbox1[1]) / 2
    center2 = (bbox2[0] + bbox2[1]) / 2
    proximity = np.linalg.norm(center1 - center2)

    min1, max1 = bbox1  
    min2, max2 = bbox2 

    top1, bottom1 = max1[2], min1[2]
    top2, bottom2 = max2[2], min2[2]
    
    if proximity < proximity_threshold:
        if -height_threshold< bottom1 - top2  and top1 > top2:
            return "Above"
        
        elif -height_threshold< bottom2 - top1 and top1 < top2:
            return "Under"

        elif (abs(center1[0] - center2[0]) < proximity_threshold and
                    abs(center1[1] - center2[1]) < proximity_threshold):
            return "Next to"
        
    return None 


def calculate_volume(bbox):
    """Calculate approximate volume of point cloud using bounding box."""
    bbox_min, bbox_max = bbox
    dimensions = bbox_max - bbox_min

    return np.prod(dimensions)


def filter_point_cloud_by_color(pcd, target_color):
    """Extracts points from the point cloud that match the target color."""
    points = np.asarray(pcd.points)  
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None  

    if colors is None:
        print("No color data found in the point cloud.")
        return None

    target_color = np.array(target_color) / 255.0
    mask = np.all(np.isclose(colors, target_color, atol=0.01), axis=1)
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcd


def create_bounding_box_for_color(pcd, target_color):
    """Creates a bounding box for the points in the point cloud that match the target color."""
    filtered_pcd = filter_point_cloud_by_color(pcd, target_color)
    if filtered_pcd is None or len(filtered_pcd.points) == 0:
        print(f"No points found for color {target_color}")
        return None

    bbox = filtered_pcd.get_axis_aligned_bounding_box()
    bbox.color = (0.5, 0.5, 0.5) 
    return bbox


def create_sphere(center, radius=0.05, color=(0.5, 0.5, 0.5)):
    """Creates sphere at the given center to represent the bounding box center."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(center)
  
    return sphere


def visualize_scene_graph(pcd, color_bboxes, scene_graph):
    """Visualizes all scene graph nodes and edges."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    processed_nodes = set()
    for color1, color2, _ in scene_graph.edges(data=True):
        for node in [color1, color2]:
            if node not in processed_nodes:
                processed_nodes.add(node)
                center = (color_bboxes[node][0] + color_bboxes[node][1]) / 2
                vis.add_geometry(create_sphere(center))
                vis.add_geometry(create_bounding_box_for_color(pcd, node))

        center1 = (color_bboxes[color1][0] + color_bboxes[color1][1]) / 2
        center2 = (color_bboxes[color2][0] + color_bboxes[color2][1]) / 2

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([center1, center2])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[1, 1, 1]])  
        vis.add_geometry(line)
    vis.run()
    vis.destroy_window()


def pairwise_relationships(pcd, scene_graph, df, args):
    """Outputs the relationship of each node pair of the scene graph."""
    for color1, color2, data in scene_graph.edges(data=True):

        category1 = df.loc[df[args.mapping_column] == color_to_label.get(color1, "Unknown"), "category"] 
        category2 = df.loc[df[args.mapping_column] == color_to_label.get(color2, "Unknown"), "category"]
        relation = data['relation']
        
        pcd1 = filter_point_cloud_by_color(pcd, color1)
        pcd2 = filter_point_cloud_by_color(pcd, color2)

        if not category1.empty and not category2.empty:
            if len(pcd1.points) > 300 and len(pcd2.points) > 300:
                print(f"{category1.iloc[0]} is {relation} the {category2.iloc[0]}.")

                
        
def main(args):

    global color_to_label, SMALL_OBJECTS, STUFF_OBJECTS

    df = pd.read_csv(Path(args.mapping_file))

    small_object_ids = [28, 22, 49, 63, 65, 130, 105, 230, 157, 103, 202, 79, 168, 214, 399, 88, 
                        1174, 562, 169, 572, 392, 1187, 115, 286, 919, 1125, 92, 1207, 228, 297, 
                        1228, 1241, 1247, 301, 1260, 1269, 15, 1276, 212, 1280, 1301]
    stuff_object_ids = [1, 3, 41]

    if args.mapping_column == "wn199-merged-v2":
        SMALL_OBJECTS = set(df.loc[df["id"].isin(small_object_ids), "wn199-merged-v2"])
        STUFF_OBJECTS = set(df.loc[df["id"].isin(stuff_object_ids), "wn199-merged-v2"])
    else:
        SMALL_OBJECTS = set(small_object_ids)
        STUFF_OBJECTS = set(stuff_object_ids)

    workspace = Path(args.workspace)

    input_mesh_path = workspace / 'mesh_instances.ply'
    assert input_mesh_path.exists(), f"Mesh file {input_mesh_path} not found"
    pcd = o3d.io.read_point_cloud(str(input_mesh_path)) 
    label_path = workspace / 'labels.txt'
    assert label_path.exists(), f"Label file {label_path} not found"
    labels = np.loadtxt(str(label_path), dtype=int)

    assert len(labels) == len(pcd.points), "Mismatch between points and labels"

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    color_to_label = {tuple(np.round(color * 255).astype(int)): label for color, label in zip(colors, labels)}


    color_instances = defaultdict(list)
    point_indices = defaultdict(list) 

    for i, color in enumerate(colors):
        color_tuple = tuple(np.round(color * 255).astype(int))
        if color_tuple != (0,0,0):  
            color_instances[color_tuple].append(points[i])
            point_indices[color_tuple].append(i)

    color_bboxes = {}
    color_points = {}

    color_mask = np.full(len(points), False) 
    for color, pts in list(color_instances.items()):
        points_array = np.array(pts)

        if points_array.shape[0] > 0:
            bbox_min = points_array.min(axis=0)
            bbox_max = points_array.max(axis=0)
            color_bboxes[color] = (bbox_min, bbox_max)
            color_points[color] = points_array

            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(points_array)
            
            _, inlier_indices = pcd_temp.remove_statistical_outlier(nb_neighbors=300, std_ratio=1.0)
            global_inlier_indices = {point_indices[color][i] for i in inlier_indices}
            
            for i in point_indices[color]:
                if i not in global_inlier_indices:
                    color_mask[i] = True

    colors[color_mask] = [0, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    scene_graph = nx.Graph()
    for color in color_instances.keys():
        label = color_to_label.get(color, "Unknown")
        if color != (0, 0, 0):
            if label in STUFF_OBJECTS:
                continue
            if label not in SMALL_OBJECTS:
                if calculate_volume(color_bboxes[color]) > 0.01 and np.array(color_instances[color]).shape[0] > 500 :
                        scene_graph.add_node(color, label=label)
            elif label in SMALL_OBJECTS and np.array(color_instances[color]).shape[0] > 200:
                scene_graph.add_node(color, label=label)

    nodes = list(scene_graph.nodes) 
    node_centers = np.array([(color_bboxes[node][0] + color_bboxes[node][1]) / 2 for node in nodes])
    tree = cKDTree(node_centers)

    for i, color1 in enumerate(nodes):
        nearby_indices = tree.query_ball_point(node_centers[i], r=2.0) 
        for j in nearby_indices:
            color2 = nodes[j]
            if color1 != color2:
                relation = get_spatial_relationship(color_bboxes[color1], color_bboxes[color2])
                if relation:
                    scene_graph.add_edge(color1, color2, relation=relation)

    pairwise_relationships(pcd, scene_graph, df, args)
    visualize_scene_graph(pcd, color_bboxes, scene_graph)


def arg_parser():
    parser = argparse.ArgumentParser(description='Create Scene Graph')
    parser.add_argument('--workspace', type=str)
    parser.add_argument('--mapping_column', type=str, default='wn199-merged-v2')
    parser.add_argument('--mapping_file', type=str, default='../../labelmaker/mappings/label_mapping.csv')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    main(args)
