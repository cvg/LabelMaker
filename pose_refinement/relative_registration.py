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
from skimage.morphology import binary_dilation

from hloc import (extract_features, match_features, reconstruction,
                  pairs_from_exhaustive, pairs_from_retrieval, match_dense)

def project_points(points, extrinsics, intrinsics):
    
    # make homogenous coordinates
    points_h = np.concatenate((points, np.ones_like(points[:, 0:1])), axis=-1)
    points_c = (extrinsics[:3, :] @ points_h.T).T
    points_p = (intrinsics @ points_c.T).T
    
    points_p[:, 0] /= points_p[:, -1]
    points_p[:, 1] /= points_p[:, -1]
    
    return points_p

@gin.configurable
class RelativeRegistration:

    def __init__(self, 
                 root_dir, 
                 scene,
                 icp_threshold_coarse=0.05,
                 icp_threshold_fine=0.03,
                 depth_threshold=3.0,
                 icp_max_iteration=1000,
                 downsample=10,
                 window_size=10,
                 voxel_size=0.02,
                 stop_frame=-1,
                 matching='sequential',
                 use_retrieval=False,
                 init_with_relative=False,
                 n_retrieval=10,
                 filter_retrieval=True,
                 uncertain_threshold=1,
                 overlap_threshold=0.3,
                 icp_type='colored_icp',
                 no_icp=False,
                 output_dir=None):

        self.root_dir = root_dir
        self.output_dir = output_dir
        self.scene = scene
        
        self.icp_threshold_coarse = icp_threshold_coarse
        self.icp_threshold_fine = icp_threshold_fine
        self.icp_max_iteration = icp_max_iteration
        self.no_icp = no_icp

        self.depth_threshold = depth_threshold

        self.downsample = downsample
        self.stop_frame = stop_frame
        self.window_size = window_size
        self.voxel_size = voxel_size

        self.matching = matching
        self.use_retrieval = use_retrieval
        self.n_retrieval = n_retrieval
        self.filter_retrieval = filter_retrieval
        self.uncertain_threshold = uncertain_threshold

        self.init_with_relative = init_with_relative
        self.icp_type = icp_type

        self._h = 480
        self._w = 640
        self._overlap_threshold = overlap_threshold * self._w * self._h

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

            if self.filter_retrieval:
                for k, v in self.retrieval_pairs_idx.items():
                    old_matches = np.asarray(v)
                    new_matches = old_matches[np.abs(old_matches - k) > self.window_size]
                    self.retrieval_pairs_idx[k] = new_matches.tolist()

            for k, v in self.retrieval_pairs_idx.items():
                print(k, v)

        if self.matching == 'overlap':
            self.retrieval_pairs_idx = {}
            self.frame_to_idx = {}
            for i, frame in enumerate(self.frames):
                self.frame_to_idx[frame] = i
                self.retrieval_pairs_idx[i] = []

        # init icp parameters
        if self.icp_type == 'point_to_plane':
            self.icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        elif self.icp_type == 'point_to_point':
            self.icp_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        elif self.icp_type == 'colored_icp':
            self.icp_method = o3d.pipelines.registration.TransformationEstimationForColoredICP()
        else:
            raise ValueError('Invalid ICP type')

        self.icp_option_coarse = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_max_iteration, relative_fitness=1e-6, relative_rmse=1e-4)
        self.icp_option_fine = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_max_iteration, relative_fitness=1e-6, relative_rmse=1e-4)
            
    def _get_target_indices(self, source_idx, n_pcds):
        if self.use_retrieval:
            target_indices = self.retrieval_pairs_idx[source_idx]
        else:
            target_indices = []

        if self.matching == 'exhaustive':
            return list(range(n_pcds)) + target_indices
        elif self.matching == 'sequential':
            return list(range(max(0, source_idx - self.window_size), min(n_pcds, source_idx + self.window_size + 1))) + target_indices
        elif self.matching == 'sequential_forward':
            return list(range(source_idx + 1, min(n_pcds, source_idx + self.window_size + 1))) + target_indices
        elif self.matching == 'overlap':
            return self.retrieval_pairs_idx[source_idx]
        else:
            raise ValueError('Invalid matching type')

    def _relative_registration(self, source, target, init):

        trans_init = init

        if not self.no_icp:
            if self.icp_type == 'colored_icp':

                reg_coarse = o3d.pipelines.registration.registration_colored_icp(source, 
                                                                                target, 
                                                                                self.icp_threshold_coarse, 
                                                                                trans_init, 
                                                                                self.icp_method,
                                                                                self.icp_option_coarse)
                

                try:
                    reg_fine = o3d.pipelines.registration.registration_colored_icp(source,
                                                                                    target,
                                                                                    self.icp_threshold_fine,
                                                                                    reg_coarse.transformation,
                                                                                    self.icp_method,
                                                                                    self.icp_option_fine)
                except Exception as e:
                    print(e)
                    reg_fine = reg_coarse

            else:
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


            return reg_fine.transformation, information_icp, reg_fine
        
        else:
            information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source,
                                                                                                target,
                                                                                                self.icp_threshold_fine, 
                                                                                                trans_init)
            return trans_init, information_icp, None
    
    def run(self):

        # load pcds
        pcds = []
        pcds_world = []
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
                                                                      depth_trunc=self.depth_threshold, 
                                                                      convert_rgb_to_intensity=False)
            
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics, np.eye(4))
            pcd_world = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics, pose)

            if self.icp_type == 'point_to_plane' or self.icp_type == 'colored_icp':
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))


            pcds.append(pcd.voxel_down_sample(voxel_size=self.voxel_size))
            pcds_world.append(pcd_world)
            poses.append(pose)

        # get matches based on overlap
        if self.matching == 'overlap':
            # import matplotlib.pyplot as plt
            for source_idx, pcd in tqdm(enumerate(pcds_world), total=len(pcds_world)):
                pose = poses[source_idx]
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                for target_idx, pcd in enumerate(pcds[source_idx+1:]):
                    pose_target = poses[target_idx + source_idx + 1]
                    points_p = project_points(points, pose_target, self.intrinsics.intrinsic_matrix)
                    xx, yy = points_p[:, 0].astype(int), points_p[:, 1].astype(int)
                    mask = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (points_p[:, -1] > 0)
                    mask_projected = np.zeros((self._h, self._w))
                    mask_projected[yy[mask], xx[mask]] = 1
                    mask_projected = binary_dilation(mask_projected, footprint=np.ones((3, 3)))

                    image_projected = np.zeros((self._h, self._w, 3))
                    depth_projected = np.zeros((self._h, self._w))
                    image_projected[yy[mask], xx[mask]] = colors[mask]
                    depth_projected[yy[mask], xx[mask]] = points_p[mask, -1]
                    n_overlap = mask_projected.sum()

                    # fig, ax = plt.subplots(1, 3, figsize=(2*4, 3))
                    # ax[0].imshow(image_projected)
                    # ax[1].imshow(depth_projected)
                    # ax[2].imshow(mask_projected)

                    # for a in ax:
                    #     a.set_xticks([])
                    #     a.set_yticks([])
                    # plt.savefig('overlap/{}_{}_{}.png'.format(source_idx, target_idx + source_idx + 1, n_overlap / (640 * 480)))
                    # plt.close('all')

                    if n_overlap > self._overlap_threshold:
                        self.retrieval_pairs_idx[source_idx].append(target_idx + source_idx + 1)
                        self.retrieval_pairs_idx[target_idx + source_idx + 1].append(source_idx)
                    
                # enforce that previous index is in matching
                if source_idx > 0:
                    if source_idx - 1 not in self.retrieval_pairs_idx[source_idx]:
                        self.retrieval_pairs_idx[source_idx].append(source_idx - 1)
                        self.retrieval_pairs_idx[source_idx - 1].append(source_idx)

            if self.output_dir is not None:
                self.save_matches_to_hloc(self.output_dir)
                  
        # relative registration
        n_pcds = len(pcds)
        print('Running relative registration...')
        for source_idx in tqdm(range(n_pcds), total=n_pcds):
            
            updated_odometry = False
            
            if source_idx == 0:
                odometry = poses[source_idx] 
                updated_odometry = True


            edges = []
            for target_idx in self._get_target_indices(source_idx, n_pcds):
                


                if target_idx == source_idx:
                    continue

                if self.init_with_relative:
                    init = np.dot(poses[target_idx], np.linalg.inv(poses[source_idx]))
                else:
                    init = np.eye(4)
       
                transformation, information, result = self._relative_registration(pcds[source_idx], pcds[target_idx], init=init)
                if target_idx == source_idx - 1 and source_idx != 0:
                    odometry = np.dot(np.linalg.inv(transformation), odometry)
                    updated_odometry = True

                edges.append(Edge(idx=target_idx, transformation=transformation, information=information, uncertain=(abs(target_idx - source_idx) > self.uncertain_threshold)))

            if len(edges) == 0:
                continue

            assert updated_odometry

            # print('Adding node', source_idx, 'connected to edges', [edge.idx for edge in edges])
            node = Node(idx=source_idx, pcl=pcd, pose=poses[source_idx], edges=edges, odometry=odometry, name=self.frames[source_idx].replace('.jpg', ''))
            self.nodes.append(node)

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, node in enumerate(self.nodes):
            pose_file = os.path.join(save_dir, f'{node.name}.txt')
            np.savetxt(pose_file, node.pose)

    def save_matches_to_hloc(self, save_dir):
        """Function to save the matches obtained with retrieval or overlap matching to hloc format.
        I.e. generate the sfm-pairs.txt file from the matches.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        try:
            # get all lines
            lines = []
            for i_idx in self.retrieval_pairs_idx.keys():
                for j_idx in self.retrieval_pairs_idx[i_idx]:
                    frame_i = self.frames[i_idx]
                    frame_j = self.frames[j_idx]
                    lines.append((frame_i, frame_j))
        except AttributeError:
            raise AttributeError('No retrieval pairs found. Please run retrieval or overlap matching first.')

        # save to file
        with open(os.path.join(save_dir, 'sfm-pairs.txt'), 'w') as f:
            for line in lines:
                f.write(f'{line[0]} {line[1]}\n')
    
    def load_matches_from_hloc(self, matches_file):
        """Function to load the matches obtained with hloc.
        I.e. read the sfm-pairs.txt file and store the matches.
        """
        with open(matches_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            frame_i, frame_j = line.split()
            i_idx = self.frames.index(frame_i)
            j_idx = self.frames.index(frame_j)
            self.retrieval_pairs_idx[i_idx].append(j_idx)
            self.retrieval_pairs_idx[j_idx].append(i_idx)
            
    def save_nodes(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pose_graph = []
        for i, node in enumerate(self.nodes):
            node_ = {'idx': node.idx, 'pose': node.pose.tolist(), 
                     'name': node.name,
                     'edges': [{'transformation': e.transformation.tolist(), 
                                'information': e.information.tolist(), 
                                'uncertain': e.uncertain,
                                'idx': e.idx } for e in node.edges], 'odometry': node.odometry.tolist()}    
            pose_graph.append(node_)

        with open(os.path.join(save_dir, 'pose_graph.json'), 'w') as f:
            json.dump(pose_graph, f)

if __name__ == '__main__':
    root_dir = '/home/weders/scratch/scratch/03-PEOPLE/weders/datasets/scannet/scans'
    scene = 'scene0575_00'

    relative_registration = RelativeRegistration(root_dir, scene, matching='overlap')
    relative_registration.init()
    relative_registration.run()