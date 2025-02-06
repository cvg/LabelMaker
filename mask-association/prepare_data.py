import argparse
import os
import json
from glob import glob
from pathlib import Path
from typing import List

import torch
import deeplake
import numpy as np
from torchvision.ops import masks_to_boxes
from tqdm import tqdm
from PIL import Image



def load_poses(filename: Path):
    """
    Load poses for given scan.
    """
    poses = []
    with open(filename) as f:
        for line in f:
            poses.append([float(x) for x in line.split()])
    poses = np.array(poses)
    return poses.reshape(-1, 4, 4)


def load_pose(filename: Path):
    """
    Load pose from pose.txt.
    """
    lines = []
    with open(filename) as f:
        for line in f:
            lines.extend([float(x) for x in line.split()])
    pose = np.array(lines).reshape(4, 4)
    return pose.astype(np.float32)


def load_image(filename: Path, downsample=1):
    """
    Load image for given scan.
    """
    image = Image.open(filename)
    image = image.resize((image.width // downsample, image.height // downsample))
    return np.array(image)


def load_mask(filename: Path, downsample=1):
    """
    Load mask for given scan.
    """
    mask = Image.open(filename)
    mask = mask.resize((mask.width // downsample, mask.height // downsample))
    return np.array(mask)


def decompose_mask(mask: np.ndarray, return_labels=False) -> List[np.ndarray]:
    """
    Decompose mask into individual bmasks.
    """
    labels = np.unique(mask[mask > 0])
    bmasks = []
    for label in labels:
        bmasks.append(mask == label)
    return (bmasks, labels) if return_labels else bmasks


def processed2deeplake(output_panopli: Path, output_dataset: Path, downsample=1, stride=1, include_sam=True):
    """
    Convert Panopli preprocessed dataset into Deeplake dataset for evaluation.
    """
    # Convert Panopli dataset into Deeplake dataset
    image_filenames = sorted(glob(str(output_panopli / 'color/*.jpg')))
    poses_filenames = sorted(glob(str(output_panopli / 'pose/*.txt')))
    depth_filenames = sorted(glob(str(output_panopli / 'depth/*.png')))
    smask_filenames = sorted(glob(str(output_panopli / 'resized_manual_label/*.png')))   
    imask_filenames = sorted(glob(str(output_panopli / 'intermediate' / 'instance_groundedsam/*.png')))
    image_filenames = image_filenames[::stride]
    # print('image file names',image_filenames[:5])
    depth_filenames = depth_filenames[::stride]
    # print('depth file names',depth_filenames[:5])
    poses_filenames = poses_filenames[::stride]
    # print('pose file names',poses_filenames[:5])
    smask_filenames = smask_filenames[::stride]
    # print('semantic file names',smask_filenames[:5])
    imask_filenames = imask_filenames[::stride]
    # print('instance file names',imask_filenames[:5])
    imask_filenames_sam = sorted(glob(str(output_panopli / 'intermediate' / 'instance_groundedsam/*.png')))    
    smask_filenames_consensus = sorted(glob(str(output_panopli / 'intermediate' /'consensus/*.png')))    
    imask_filenames_sam = imask_filenames_sam[::stride]
    # print('instance file names',imask_filenames_sam[:5])
    smask_filenames_consensus = smask_filenames_consensus[::stride]
    # print('consensus file names',smask_filenames_consensus[:5])

    ds = deeplake.empty(output_dataset, overwrite=True)
    ds.create_tensor('image'         , htype='image'                        , sample_compression='png')
    # ds.create_tensor('depth'         , htype='image'                        , sample_compression='png')
    ds.create_tensor('instances/mask', htype='instance_label', dtype='int32'  , sample_compression='lz4')
    # ds.create_tensor('semantics/mask', htype='class_label', dtype='int32'  , sample_compression='lz4')
    ds.create_tensor('pose'          , htype='generic'     , dtype='float32')
    ds.create_tensor('metadata'      , htype='json')
    if include_sam:
        # ds.create_tensor('consensus/semantics/mask'  , htype='segment_mask', dtype='int32', sample_compression='lz4')
        ds.create_tensor('sam/instances/mask'  , htype='segment_mask', dtype='int32', sample_compression='lz4')
        ds.create_tensor('sam/instances/bmasks', htype='binary_mask'                , sample_compression='lz4')
        ds.create_tensor('sam/instances/bboxes', htype='bbox'        , dtype='int32', coords={'type': 'pixel', 'mode': 'LTRB'})

    # Sample image used downstream for camera parameters
    image0 = load_image(image_filenames[0])
    assert image0.shape[0] % downsample == 0 and \
           image0.shape[1] % downsample == 0
    image0 = load_image(image_filenames[0], downsample=downsample)

    def process_imask(imask):
        bmasks = decompose_mask(imask)
        if len(bmasks):
            bmasks = np.stack(bmasks)
            bboxes = masks_to_boxes(torch.from_numpy(bmasks)).numpy().astype(np.int32)
        else:
            bmasks = np.zeros((1, image0.shape[0], image0.shape[1]), dtype=bool)
            bboxes = np.zeros((1, 4), dtype=np.int32)
        return bmasks, bboxes

    for filename, pfilename, dfilename, sfilename, ifilename, sfilename_consensus, ifilename_sam in tqdm(zip(
        image_filenames,
        poses_filenames,
        depth_filenames,
        smask_filenames,
        imask_filenames,
        smask_filenames_consensus,
        imask_filenames_sam,
    )):
        pose = load_pose(pfilename) @ np.array([ # Convert back to Nerfstudio (Instant-NGP) convention from panopli format
            [1,  0,  0,  0],
            [0, -1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1],
        ], dtype=np.float32)
        image = load_image(filename, downsample=downsample)
        # depth = load_image(dfilename, downsample=downsample)
        # smask = load_mask(sfilename, downsample=downsample)
        imask = load_mask(ifilename, downsample=downsample)
        if include_sam:
            # smask_consensus = load_mask(sfilename_consensus, downsample=downsample)
            imask_sam = load_mask(ifilename_sam, downsample=downsample)
            bmasks, bboxes = process_imask(imask_sam)
        appended = {
            'image'         : image,
            # 'depth'         : depth,
            # 'semantics/mask': smask,
            'instances/mask': imask,
            'pose'          : pose,
            'metadata'      : {},
        }
        if include_sam:
            appended.update({
                # 'consensus/semantics/mask'  : smask_consensus,
                'sam/instances/mask'  : imask_sam,
                'sam/instances/bmasks': bmasks.transpose(1, 2, 0), # deeplake convention HWN
                'sam/instances/bboxes': bboxes,
            })
        ds.append(appended)

    # camera_params = load_pose(output_panopli / 'intrinsic/intrinsic_color.txt')
    camera_params = np.loadtxt(output_panopli / 'intrinsic/000000.txt')
    ds.info['camera_params'] = {
        'fx'    : float(camera_params[0, 0]) / downsample,
        'fy'    : float(camera_params[1, 1]) / downsample,
        'cx'    : float(camera_params[0, 2]) / downsample,
        'cy'    : float(camera_params[1, 2]) / downsample,
        'height': image0.shape[0], # image0 already downsampled
        'width' : image0.shape[1],
    }
    if include_sam:
        # with open('resources/scannet_reduced_categories.json') as f:
        #     categories = json.load(f)
        # ds.info['sam/categories'] = categories
        # ds.info['consensus/semantics/num_classes'] = max(categories['things'] + categories['stuff']) + 1
        ds.info['consensus/semantics/num_classes'] = 188
        # ds.info['sam/instances/num_classes'] = max(categories['things']                      ) + 1
        ds.info['sam/instances/num_instances'] = 188
    
    print(ds.summary())
    print(ds.info)



def process_scannet(output_panopli: Path, output_dataset: Path):
    """
    Convert Scannet scan into deeplake dataset for evaluation.
    """
    print('Processing Scannet ', '->', output_panopli, output_dataset)

    # os.makedirs(output_panopli, exist_ok=True)
    os.makedirs(output_dataset, exist_ok=True)

    # downsample = 1 if scan.stem in ['scene0144_01', 'scene0354_00'] else 2
    processed2deeplake(output_panopli, output_dataset, downsample=1, stride=5, include_sam=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process Scannet scans.'
    )
    parser.add_argument(
        '-i', '--input', required=False, type=str, help='Input scan name'
    )
    parser.add_argument(
        '-id', '--input_dataset', required=False, type=str, help='Input directory for dataset.' , 
        default='/home/user/zhang2/LabelMaker/data/scannet/scene0164_02'
    )
    parser.add_argument(
        '-od', '--output_dataset', required=False, type=str, help='Output directory for deeplake dataset.' ,
        default='/home/user/zhang2/IML/data/scannet/scene0164_02'
    )
    parser.add_argument(
        '-t', '--type', required=False, type=str, choices=['scannet', 'arkit'], help='Type of the scan.',
        default='scannet'
    )
    args = parser.parse_args()
    print(args)

    # process = {
    #     'scannet'     : process_scannet,
    # }[args.type]

    process_scannet(Path(args.input_dataset), Path(args.output_dataset))

