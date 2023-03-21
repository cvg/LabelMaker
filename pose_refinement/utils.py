import numpy as np
import open3d as o3d

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