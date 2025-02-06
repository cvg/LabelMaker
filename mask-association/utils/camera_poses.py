from typing import Tuple

import numpy as np
import torch
from torchtyping import TensorType as TorchTensor
from kornia.geometry import conversions
from kornia.geometry.conversions import QuaternionCoeffOrder


HomogeneousTransform = TorchTensor["batch":..., 4, 4]
Qvec = TorchTensor["batch":..., 4]
Tvec = TorchTensor["batch":..., 3]


def spherical_to_cartesian(theta: TorchTensor, phi: TorchTensor) -> Tuple[TorchTensor, ...]:
    """ 
    Convert spherical coordinates to cartesian coordinates 
    """
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return x, y, z


def cartesian_to_spherical(x: TorchTensor, y: TorchTensor, z: TorchTensor) -> Tuple[TorchTensor, ...]:
    """ 
    Convert cartesian coordinates to spherical coordinates 
    """
    r     = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(y, x)
    phi   = torch.acos(z / r)
    return theta, phi


def spherical_mean(thetas: TorchTensor, phis: TorchTensor) -> Tuple[float, float]:
    """ 
    Compute the average of angles in spherical coordinates 
    """
    x, y, z = spherical_to_cartesian(thetas, phis)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    mean_z = torch.mean(z)
    return cartesian_to_spherical(mean_x, mean_y, mean_z)


def batch_repeat(x1: TorchTensor, x2: TorchTensor) -> TorchTensor:
    """ 
    Repeat `x1` along the batch dimension to match the batch dimension of `x2`.
    """
    return x1.unsqueeze(0).repeat(x2.shape[0], *([1] * x1.ndim))


def Rt_dists(poses: HomogeneousTransform, deg=True) -> Tuple[TorchTensor, TorchTensor]:
    """ 
    Compute the distance between the rotation and translation components of a 4x4 transformation matrix
    using the principal axes and Euclidean distance, respectively.
    """
    Rs = poses[:, :3, -1] # principal axes
    Rs = Rs / torch.linalg.norm(Rs, axis=1, keepdims=True)
    ts = poses[:, :3,  3]
    Tdmat = torch.linalg.norm(ts[:, None] - ts[None], axis=-1)
    Rdmat = torch.einsum('mi,ni->mn', Rs, Rs)
    Rdmat = torch.arccos(torch.clip(Rdmat, -1.0, 1.0))
    if deg: Rdmat = torch.rad2deg(Rdmat)
    return Rdmat, Tdmat


def qt_to_matrix4x4(q: Qvec, t: Tvec, order=QuaternionCoeffOrder.WXYZ) -> HomogeneousTransform:
    """ 
    Convert a quaternion and translation vector to a 4x4 transformation matrix.
    """
    R = conversions.quaternion_to_rotation_matrix(q, order=order)
    return conversions.Rt_to_matrix4x4(R, t.unsqueeze(-1))


def matrix4x4_to_qt(poses: HomogeneousTransform, order=QuaternionCoeffOrder.WXYZ) -> Tuple[Qvec, Tvec]:
    """
    Convert a 4x4 transformation matrix to a quaternion and translation vector.
    """
    R, t = conversions.matrix4x4_to_Rt(poses)
    q = conversions.rotation_matrix_to_quaternion(R.contiguous(), order=order)
    return q, t.squeeze(-1)


def opengl2opencv_coordinate_convention(poses: HomogeneousTransform) -> HomogeneousTransform:
    """
    Convert between OpenGL and OpenCV camera coordinate conventions. 
    """
    trans = torch.eye(4)
    trans[1, 1] = trans[2, 2] = -1
    return trans @ poses


def nerfstudio2colmap_poses(poses: HomogeneousTransform) -> HomogeneousTransform:
    """
    Convert poses in nerfstudio camera coordinate convention (cam2world OpenGL) to colmap (world2cam OpenCV). 
    """
    return torch.inverse(opengl2opencv_coordinate_convention(poses))


def colmap2nerfstudio_poses(poses: HomogeneousTransform) -> HomogeneousTransform:
    """ 
    Convert poses in colmap camera coordinate convention (world2cam OpenCV) to nerfstudio (cam2world OpenGL).
    """
    return opengl2opencv_coordinate_convention(torch.inverse(poses))
    
