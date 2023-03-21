import os
import cv2
import gin

import argparse


import open3d as o3d
import numpy as np


from copy import deepcopy
from PIL import Image

PATH = '/home/weders/scratch/scratch/03-PEOPLE/weders/datasets/scannet/scans/scene0000_00/data/'
frames = sorted(os.listdir(os.path.join(PATH, 'color')), key=lambda x: int(x.split('.')[0]))

# relative registration prototypes

from dataclasses import dataclass

@dataclass
class Node:
    idx: int
    pcl: o3d.geometry.PointCloud
    pose: np.ndarray
    edges: []
        
@dataclass
class Edge:
    idx: int
    information: np.ndarray
    transformation: np.ndarray
    uncertain: bool

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/weders/scratch/scratch/03-PEOPLE/weders/datasets/scannet/scans/')
    parser.add_argument('--scene', type=str, default='scene0000_00')
    args = parser.parse_args()
    return args


class RelativeRegistration:

    def __init__(self, 
                 root_dir, 
                 scene,
                 icp_threshold=0.1,
                 icp_max_iter=100,):

        self.root_dir = root_dir
        self.scene = scene
        self.icp_threshold = icp_threshold

    def init(self):

        self.frames = sorted(os.listdir(os.path.join(self.root_dir, self.scene, 'data', 'color')), key=lambda x: int(x.split('.')[0]))[::20]

        self.intrinsics_file = os.path.join(self.root_dir, self.scene, 'data', 'intrinsic', 'intrinsic_depth.txt')
        self.intrinsics = np.loadtxt(self.intrinsics_file)[:3, :3]
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, intrinsic_matrix=self.intrinsics)

        self.nodes = []
    
    def run(self):

        # initializing relative pose with identity
        trans_init = np.eye(4)

        for idx, frame in enumerate(self.frames):
    
    
            pose_file = os.path.join(self.root_dir, self.scene, 'data', 'pose', frame.replace('jpg', 'txt'))
            image_file = os.path.join(self.root_dir, self.scene, 'data', 'color', frame)
            depth_file = os.path.join(self.root_dir, self.scene, 'data', 'depth', frame.replace('jpg', 'png'))
    
            pose = np.loadtxt(pose_file)
            pose = np.linalg.inv(pose)
            
            image = np.asarray(Image.open(image_file)).astype(np.uint8)
            depth = np.asarray(Image.open(depth_file)).astype(np.float32) / 1000.
    
            # resizing image to depth shape
            h, w = depth.shape
            image = cv2.resize(image, (w, h))
            
            image = o3d.geometry.Image(image)
            depth = o3d.geometry.Image(depth)
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(image, 
                                                                    depth, 
                                                                    depth_scale=1, 
                                                                    depth_trunc=3., 
                                                                    convert_rgb_to_intensity=False)
    
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics, pose)
            
            
            if len(self.nodes) > 0:
                reg_p2l = o3d.pipelines.registration.registration_icp(self.nodes[idx - 1].pcl, 
                                                                      pcd, 
                                                                      self.icp_threshold, 
                                                                      trans_init, 
                                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint())
                
                transformation_icp = reg_p2l.transformation
                information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(self.nodes[-1].pcl, 
                                                                                                      pcd, 
                                                                                                      self.icp_threshold, 
                                                                                                      transformation_icp)       
                print(reg_p2l)
                print("Transformation is:")
                print(reg_p2l.transformation)

            
            
            if idx > 0:
                edges = [Edge(idx=idx-1, information=information_icp, transformation=transformation_icp, uncertain=False)]
            else:
                edges = []
                
            node = Node(idx=idx, pcl=pcd, pose=pose, edges=edges)
            self.nodes.append(node)
            # if idx == 10:
            #     break
    
class AbsoluteRegistration:
    def __init__(self):
        pass
    
    def init(self, nodes):
        self.pose_graph = o3d.pipelines.registration.PoseGraph()

        for node in nodes:
            self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(node.pose))
            
            for edge in node.edges:
                print(node.idx, edge.idx)
                self.pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(node.idx,
                                                                                 edge.idx,
                                                                                 edge.transformation,
                                                                                 edge.information,
                                                                                 uncertain=edge.uncertain))

        print('Built pose graph with {} nodes and {} edges'.format(len(self.pose_graph.nodes), len(self.pose_graph.edges)))
        print(self.pose_graph)


    def run(self):
        voxel_size=0.007
        max_correspondence_distance_fine = voxel_size * 10.

        print("Optimizing PoseGraph ...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(self.pose_graph,
                                                           o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                                           o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                           option)



def main(args):
    relative_registration = RelativeRegistration(args.root_dir, args.scene)
    relative_registration.init()
    relative_registration.run()

    absolute_registration = AbsoluteRegistration()
    absolute_registration.init(relative_registration.nodes)
    absolute_registration.run()
    

if __name__ == '__main__':
    args = arg_parser()
    main(args)