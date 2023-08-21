import os
import cv2
import argparse

import open3d as o3d
import numpy as np

from tqdm import tqdm
from PIL import Image



def arg_parser():

    parser = argparse.ArgumentParser(description='Project 3D points to 2D image plane and aggregate labels and save label txt')
    parser.add_argument('--data_dir', type=str, default='/home/weders/scratch/scratch/scannetter', help='Path where the data is stored')
    parser.add_argument('--label_key', default='label-filt')
    parser.add_argument('--subsampling', type=int, default=1, help='Subsampling of the frames')
    parser.add_argument('--scene')
    parser.add_argument('--max_label', type=int, default=2000, help='Max label value')
    parser.add_argument('--dataset', type=str, default='scannet', help='Dataset name')
    parser.add_argument('--use_sdfstudio_mesh', action='store_true', help='Use SDFStudio mesh instead of the original one')
    parser.add_argument('--output_key', type=str, required=True, help='Output key for the label txt file')
    return parser.parse_args()


def project_pointcloud(points, pose, intrinsics):
    
    points_h = np.hstack((points, np.ones_like(points[:, 0:1])))
    points_c = np.linalg.inv(pose) @ points_h.T
    points_c = points_c.T

    if intrinsics.shape[-1] == 3:
        intrinsics = np.hstack((intrinsics, np.zeros((3, 1))))
        intrinsics = np.vstack((intrinsics, np.zeros((1, 4))))
        intrinsics[-1, -1] = 1.

    points_p = intrinsics @ points_c.T
    points_p = points_p.T[:, :3]
    
    points_p[:, 0] /=  (points_p[:, -1] + 1.e-6)
    points_p[:, 1] /=  (points_p[:, -1] + 1.e-6)
    
    return points_p

def main(args):

    print('Procesing {} using label key {} and subsampling {}'.format(args.scene, args.label_key, args.subsampling))

    sc = args.scene


    if args.dataset == 'scannet' or args.dataset == 'replica':
        scene_path = os.path.join(args.data_dir, sc)
    elif args.dataset == 'arkitscenes':
        scene_path = os.path.join(args.data_dir, 'arkit/raw/Validation', sc)
    
    if args.dataset == 'scannet':
        image_path = os.path.join(scene_path, 'color')
        depth_path = os.path.join(scene_path, 'depth')
        intrinsics_path = os.path.join(scene_path, 'intrinsic')
        pose_path = os.path.join(scene_path, 'pose')
        label_path = os.path.join(scene_path, args.label_key)
    elif args.dataset == 'arkitscenes':
        image_path = os.path.join(scene_path, 'color_old')
        depth_path = os.path.join(scene_path, 'depth_old')
        intrinsics_path = os.path.join(scene_path, 'intrinsic')
        pose_path = os.path.join(scene_path, 'pose_old')
        label_path = os.path.join(scene_path, args.label_key)
    
    if args.dataset == 'scannet':
        mesh_path = os.path.join(scene_path, f'{sc}_vh_clean.ply')

        if args.use_sdfstudio_mesh:
            run_id = args.label_key.replace('pred_sdfstudio_', '')
            mesh_path = os.path.join(scene_path, f'../{run_id}/mesh_visible_scaled.ply')

    elif args.dataset == 'arkitscenes':
        mesh_path = os.path.join(scene_path, f'{sc}_3dod_mesh.ply')

    # load mesh and extract colors
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices)

    # init label container
    labels_3d = np.zeros((vertices.shape[0], args.max_label + 1))

    files = [f for f in os.listdir(label_path) if f.endswith('png')]
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    resize_image = False
    subsampling = args.subsampling
    intrinsics_loaded = False
    for idx, file in  tqdm(enumerate(files), total=len(files)):

        frame_key = int(file.split('.')[0]) * subsampling
            
        image = np.asarray(Image.open(os.path.join(image_path, f'{frame_key}.jpg'))).astype(np.uint8)
        depth = np.asarray(Image.open(os.path.join(depth_path, f'{frame_key}.png'))).astype(np.float32) / 1000.
        labels = np.asarray(Image.open(os.path.join(label_path, file)))

        max_label = np.max(labels)
        if max_label > labels_3d.shape[-1] - 1:
            raise ValueError(f'Label {max_label} is not in the label range of {labels_3d.shape[-1]}')
       
        if resize_image:
            h, w = depth.shape
            image = cv2.resize(image, (w, h))
            labels = cv2.resize(labels, (w, h))
        else:
            h, w, _ = image.shape
            depth = cv2.resize(depth, (w, h))
        
        if not intrinsics_loaded:
            intrinsics = np.loadtxt(intrinsics_path + '/intrinsic_color.txt')
            intrinsics_loaded = False
        
        pose_file = os.path.join(pose_path, f'{frame_key}.txt')
        pose = np.loadtxt(pose_file)
        
        points_p = project_pointcloud(vertices, pose, intrinsics)
        
        xx = points_p[:, 0].astype(int)
        yy = points_p[:, 1].astype(int)
        zz = points_p[:, 2]
        
        valid_mask = (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)
        
        d = depth[yy[valid_mask], xx[valid_mask]]
        
        valid_mask[valid_mask] = (zz[valid_mask] > 0) & (np.abs(zz[valid_mask] - d) <= 0.1)

        labels_2d = labels[yy[valid_mask], xx[valid_mask]]
        labels_3d[valid_mask, labels_2d] += 1


    labels_3d = np.argmax(labels_3d, axis=-1)

    
    # map labels to original mesh
    if args.use_sdfstudio_mesh:
        original_mesh = o3d.io.read_triangle_mesh(os.path.join(scene_path, f'{sc}_vh_clean.ply'))

        kdtree = o3d.geometry.KDTreeFlann(mesh)
        vertices = np.asarray(original_mesh.vertices)
        labels_3d_mapped = np.zeros((vertices.shape[0], ))

        for idx, point in tqdm(enumerate(vertices), total=len(vertices)):
            [_, old_idx, _] = kdtree.search_knn_vector_3d(point, 1)
            labels_3d_mapped[idx] = labels_3d[old_idx]
        
        labels_3d_mapped = labels_3d_mapped.astype(int)
        np.savetxt(os.path.join(scene_path, f'{args.output_key}_labels_3d_sdfstudio.txt'), labels_3d_mapped, fmt='%i')

    else:
        np.savetxt(os.path.join(scene_path, f'{args.output_key}_labels_3d.txt'), labels_3d, fmt='%i')
            
if __name__ == '__main__':
    args = arg_parser()
    main(args)