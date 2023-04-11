import os
import pyquaternion
import cv2
import shutil

import numpy as np
import open3d as o3d

from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm

@dataclass
class Node:
    idx: int
    pcl: o3d.geometry.PointCloud
    pose: np.ndarray
    odometry: np.ndarray
    edges: []
    name: str
        
@dataclass
class Edge:
    idx: int
    information: np.ndarray
    transformation: np.ndarray
    uncertain: bool

class Images(object):
    
    def __init__(self):
        
        self._images = []
        
    def add(self, image_id, rotation, translation, camera_id, name):    
        self._images.append((image_id, rotation, translation, camera_id, name))
    
    def save(self, path):
        
        with open(path + '/images.txt', 'w') as file:
            
            for (image_id, rotation, translation, camera_id, name) in self._images:
                # everything is stored in scalar last but COLMAP uses scalar first convention
                file.write(f'{image_id} {rotation[0]} {rotation[1]} {rotation[2]} {rotation[3]} {translation[0]} {translation[1]} {translation[2]} {camera_id} {name} \n\n')

    def load(self, path):
        with open(path, 'r') as file:
            line_count = 0
            for line in file:
                if len(line) <= 1:
                    continue
                if line[0] == '#':
                    continue
                
                if line_count % 2 == 0:
                    self._images.append(line.rstrip().split(' '))
                line_count += 1

    def __getitem__(self, item):
        return self._images[item]

class Cameras(object):
    
    def __init__(self):
        
        self._cameras = []
        
    def add(self, camera_id, model, width, height, fx, fy, cx, cy):  
        self._cameras.append((camera_id, model, width, height, fx, fy, cx, cy))
    
    def save(self, path):
        
        with open(path + '/cameras.txt', 'w') as file:
            for (camera_id, camera_model, width, height, fx, fy, cx, cy) in self._cameras:
                file.write(f'{camera_id} {camera_model} {width} {height} {fx} {fy} {cx} {cy}\n')
                
class Points3D(object):
    
    def __init__(self):
        
        self._points = []
        
    def add(self):    
        self._cameras.append(())
    
    def save(self, path):
        # this is only a dummy so far to create an empty point3D file
        with open(path + '/points3D.txt', 'w') as file:
            for (_) in self._points:
                pass

def process_pose(pose, invert=True):

    if invert:
        pose = np.linalg.inv(pose)

    mat = pose[:3, :3]
    translation = pose[:3, 3]
    q = pyquaternion.Quaternion(matrix=mat, atol=1e-3, rtol=1e-3)
    rotation = (q.elements[0], q.elements[1], q.elements[2], q.elements[3])
    return translation, rotation

def save_to_colmap(pose_graph=None, nodes=None, intrinsics=None, save_dir=None, image_dir=None, invert_pose=False):
    if image_dir is not None:
        image_save_dir = os.path.join(save_dir, 'images')
        os.makedirs(image_save_dir, exist_ok=True)

    save_dir = os.path.join(save_dir, 'sparse')
    os.makedirs(save_dir, exist_ok=True)

    images = Images()
    cameras = Cameras()
    points3d = Points3D()
    

    cameras.add(0, 'PINHOLE', 640, 480, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])

    iter_nodes = pose_graph.nodes if pose_graph is not None else nodes

    for i, node in enumerate(iter_nodes):
        
        translation, rotation = process_pose(node.pose, invert=invert_pose)
        images.add(i, rotation, translation, 0, nodes[i].name + '.jpg')

        if image_dir is not None:
            image = np.asarray(Image.open(os.path.join(image_dir, nodes[i].name + '.jpg')))
            image = cv2.resize(image, (640, 480))
            Image.fromarray(image).save(os.path.join(image_save_dir, nodes[i].name + '.jpg'))

    images.save(save_dir)
    cameras.save(save_dir)
    points3d.save(save_dir)

def load_from_colmap(path, invert_pose=False):
    images = Images()
    images.load(os.path.join(path, 'images.txt'))

    nodes = []
    for i, image in enumerate(images):
        if i == 1:
            print(image)
        translation = np.array([float(image[5]), float(image[6]), float(image[7])])
        rotation = np.array([float(image[1]), float(image[2]), float(image[3]), float(image[4])])
        q = pyquaternion.Quaternion(rotation)
        mat = q.rotation_matrix
        pose = np.eye(4)
        pose[:3, :3] = mat
        pose[:3, 3] = translation
        if invert_pose:
            pose = np.linalg.inv(pose)
        node = Node(idx=i, pcl=None, pose=pose, odometry=pose, edges=[], name=image[9])
        nodes.append(node)

    return nodes

def save_to_scannet(nodes, path, pose_mode, scannet_path):
    
    # setup all paths
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/color', exist_ok=True)
    os.makedirs(path + '/depth', exist_ok=True)
    os.makedirs(path + '/intrinsic', exist_ok=True)

    os.makedirs(path + '/' + pose_mode, exist_ok=True)
    
    print('Saving output to ScanNet format...')
    for i, node in tqdm(enumerate(nodes), total=len(nodes)):
        
        # copy images and depth maps
        shutil.copyfile(scannet_path + '/color/' + node.name, path + '/color/' + node.name)
        shutil.copyfile(scannet_path + '/depth/' + node.name.replace('jpg', 'png'), path + '/depth/' + node.name.replace('jpg', 'png'))
        shutil.copyfile(scannet_path + '/intrinsic/intrinsic_color.txt', path + '/intrinsic/intrinsic_color.txt')
        shutil.copyfile(scannet_path + '/intrinsic/intrinsic_depth.txt', path + '/intrinsic/intrinsic_depth.txt')


        # save poses
        pose = node.pose
        pose = np.linalg.inv(pose)
        np.savetxt(path + '/' + pose_mode + '/' + node.name.replace('jpg', 'txt'), pose, fmt='%.6f')


if __name__ == '__main__':
    output_path = 'output/debug_600_5_loftr_sequential/scene0575_00'


    nodes = load_from_colmap(output_path + '/colmap/triangulation_ba')
    save_to_scannet(nodes=nodes, path=output_path + '/scannet', pose_mode='pose_ba', scannet_path='/home/weders/scratch/scratch/03-PEOPLE/weders/datasets/scannet/scans/scene0575_00/data')