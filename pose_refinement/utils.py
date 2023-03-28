import os
import pyquaternion
import cv2

import numpy as np
import open3d as o3d

from dataclasses import dataclass
from PIL import Image

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
            for line in file:
                if len(line) <= 1:
                    continue
                self._images.append(line.rstrip().split(' '))

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
            cv2.imwrite(os.path.join(image_save_dir, nodes[i].name + '.jpg'), image)

    images.save(save_dir)
    cameras.save(save_dir)
    points3d.save(save_dir)