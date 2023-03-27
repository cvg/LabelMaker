import os
import cv2
import gin
import json

import numpy as np
import open3d as o3d

from PIL import Image
from utils import Node, Edge
from pathlib import Path
from tqdm import tqdm

from hloc import (extract_features, match_features, reconstruction,
                  pairs_from_exhaustive, pairs_from_retrieval, match_dense)

@gin.configurable
class RelativeRegistration:

    def __init__(self, 
                 root_dir, 
                 scene,
                 icp_threshold_coarse=0.3,
                 icp_threshold_fine=0.04,
                 icp_max_iteration=1000,
                 downsample=10,
                 window_size=10,
                 voxel_size=0.02,
                 stop_frame=-1,
                 matching='sequential',
                 use_retrieval=False,
                 init_with_relative=False,
                 n_retrieval=10,
                 icp_type='point_to_point'):

        self.root_dir = root_dir
        self.scene = scene
        
        self.icp_threshold_coarse = icp_threshold_coarse
        self.icp_threshold_fine = icp_threshold_fine
        self.icp_max_iteration = icp_max_iteration

        self.downsample = downsample
        self.stop_frame = stop_frame
        self.window_size = window_size
        self.voxel_size = voxel_size

        self.matching = matching
        self.use_retrieval = use_retrieval
        self.n_retrieval = n_retrieval

        self.init_with_relative = init_with_relative
        self.icp_type = icp_type

    def init(self):

        self.frames = sorted(os.listdir(os.path.join(self.root_dir, self.scene, 'data', 'color')), key=lambda x: int(x.split('.')[0]))[::self.downsample]

        if self.stop_frame > 0:
            self.frames = self.frames[:self.stop_frame]

        self.intrinsics_file = os.path.join(self.root_dir, self.scene, 'data', 'intrinsic', 'intrinsic_depth.txt')
        self.intrinsics = np.loadtxt(self.intrinsics_file)[:3, :3]
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, intrinsic_matrix=self.intrinsics)

        self.nodes = []

        if self.use_retrieval:
            self.scene_path = Path(self.root_dir) / self.scene
            self.retrieval_conf = extract_features.confs['netvlad']
            self.tmp_dir = Path('/tmp/hloc')
            self.sfm_pairs = self.tmp_dir / 'sfm-pairs.txt'
            image_dir = self.scene_path / 'data/color'
            image_list = []
            image_paths = list(image_dir.iterdir())
            image_paths = [f for f in image_paths if f.name.split('.')[0] != 'query']
            image_paths = sorted(image_paths, key=lambda x: int(x.name.split('.')[0]))
            
            image_paths = image_paths[::self.downsample]
            if self.stop_frame > 0:
                image_paths = image_paths[:self.stop_frame]

            image_list_path = []
            indices = np.arange(len(image_paths))

            for index in indices:
                image_list.append(image_paths[index])
                image_list_path.append(
                    str(Path(image_paths[index]).relative_to(image_dir)))

            self.retrieval_path = extract_features.main(self.retrieval_conf,
                                                        image_dir,
                                                        self.tmp_dir,
                                                        image_list=image_list_path)
            
            pairs_from_retrieval.main(self.retrieval_path,
                                      self.sfm_pairs,
                                      num_matched=self.n_retrieval)
            
            # parse retrieval pairs
            self.retrieval_pairs_name = {}


            pairs = []
            with open(self.sfm_pairs, 'r') as f:
                for line in f:
                    line = line.strip().split()
                    pairs.append((line[0], line[1]))


            for (i, j) in pairs:
                if i not in self.retrieval_pairs_name:
                    self.retrieval_pairs_name[i] = [j]
                self.retrieval_pairs_name[i].append(j)
            
            self.retrieval_pairs_name_index_mapping = {}
            for idx, im_name in enumerate(sorted(self.retrieval_pairs_name.keys(), key=lambda x: int(x.split('.')[0]))):
                self.retrieval_pairs_name_index_mapping[im_name] = idx

            self.retrieval_pairs_idx = {}

            for idx, im_name in enumerate(sorted(self.retrieval_pairs_name.keys(), key=lambda x: int(x.split('.')[0]))):
                source_idx = self.retrieval_pairs_name_index_mapping[im_name]
                target_idx = [self.retrieval_pairs_name_index_mapping[j] for j in self.retrieval_pairs_name[im_name]]
                self.retrieval_pairs_idx[source_idx] = target_idx


        # init icp parameters
        if self.icp_type == 'point_to_plane':
            self.icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        elif self.icp_type == 'point_to_point':
            self.icp_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        else:
            raise ValueError('Invalid ICP type')

        self.icp_option_coarse = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_max_iteration, relative_fitness=1e-6, relative_rmse=1e-6)
        self.icp_option_fine = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_max_iteration, relative_fitness=1e-6, relative_rmse=1e-6)
            
    def _get_target_indices(self, source_idx, n_pcds):
        if self.use_retrieval:
            target_indices = self.retrieval_pairs_idx[source_idx]
        else:
            target_indices = []

        if self.matching == 'exhaustive':
            return list(range(n_pcds)) + target_indices
        if self.matching == 'sequential':
            return list(range(max(0, source_idx - self.window_size), min(n_pcds, source_idx + self.window_size + 1))) + target_indices
        if self.matching == 'sequential_forward':
            return list(range(source_idx + 1, min(n_pcds, source_idx + self.window_size + 1))) + target_indices

    def _relative_registration(self, source, target, init):

        trans_init = init

        reg_coarse = o3d.pipelines.registration.registration_icp(source, 
                                                                 target, 
                                                                 self.icp_threshold_coarse, 
                                                                 trans_init, 
                                                                 self.icp_method,
                                                                 self.icp_option_coarse)
        
        reg_fine = o3d.pipelines.registration.registration_icp(source,
                                                               target,
                                                               self.icp_threshold_fine,
                                                               reg_coarse.transformation,
                                                               self.icp_method,
                                                               self.icp_option_fine)


        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source,
                                                                                              target,
                                                                                              self.icp_threshold_fine, 
                                                                                              reg_fine.transformation)


        return reg_fine.transformation, information_icp
    
    def run(self):

        # load pcds
        pcds = []
        poses = []
            
        print('Loading pcds...')
        for idx, frame in tqdm(enumerate(self.frames), total=len(self.frames)):
                    
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
            
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics, np.eye(4))

            if self.icp_type == 'point_to_plane':
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))

            pcds.append(pcd.voxel_down_sample(voxel_size=self.voxel_size))
            poses.append(pose)

        # relative registration
        n_pcds = len(pcds)
        print('Running relative registration...')
        for source_idx in tqdm(range(n_pcds), total=n_pcds):
            if source_idx == 0:
                odometry = poses[source_idx] 

            edges = []
            for target_idx in self._get_target_indices(source_idx, n_pcds):

                if self.init_with_relative:
                    init = np.dot(poses[target_idx], np.linalg.inv(poses[source_idx]))
                else:
                    init = np.eye(4)

                transformation, information = self._relative_registration(pcds[source_idx], pcds[target_idx], init=init)
                if target_idx == source_idx + 1 and source_idx != 0:
                    odometry = np.dot(transformation, odometry)
                edges.append(Edge(idx=target_idx, transformation=transformation, information=information, uncertain=(not (target_idx == source_idx + 1))))
            
            # print('Adding node', source_idx, 'connected to edges', [edge.idx for edge in edges])
            node = Node(idx=source_idx, pcl=pcd, pose=pose, edges=edges, odometry=odometry)
            self.nodes.append(node)

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, node in enumerate(self.nodes):
            pose_file = os.path.join(save_dir, f'{str(i).zfill(5)}.txt')
            np.savetxt(pose_file, node.pose)

    def save_nodes(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pose_graph = []
        for i, node in enumerate(self.nodes):
            node_ = {'idx': node.idx, 'pose': node.pose, 'edges': node.edges, 'odometry': node.odometry}    
            pose_graph.append(node_)

        with open(os.path.join(save_dir, 'pose_graph.json'), 'w') as f:
            json.dump(pose_graph, f)