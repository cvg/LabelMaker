
from email.mime import base
import json
import argparse
import os
import numpy as np
import cv2
import copy
import csv
from utils import nyu40_colour_code


def load_scannet_nyu40_mapping(path):
    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, nyu40id = int(line[0]), int(line[4])
            mapping[scannet_id] = nyu40id
    return mapping

def load_scannet_nyu13_mapping(path):
    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, nyu40id = int(line[0]), int(line[5])
            mapping[scannet_id] = nyu40id
    return mapping


parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

parser.add_argument("--scene_folder", type=str, default="")
parser.add_argument("--scaled_image", action="store_true")
parser.add_argument("--semantics", action="store_true")
parser.add_argument("--pallete", type=str, default="")
args = parser.parse_args()
basedir = args.scene_folder

print(f"processing folder: {basedir}")

# Step for generating training images
step = 1

frame_ids = os.listdir(os.path.join(basedir, 'color'))
frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
frame_ids =  sorted(frame_ids)

intrinsic_file = os.path.join(basedir,"intrinsic/intrinsic_color.txt")
intrinsic = np.loadtxt(intrinsic_file)
print("intrinsic parameters:")
print(intrinsic)

imgs = []
poses = []

K_unscaled = copy.deepcopy(intrinsic)

W_unscaled = 1296
H_unscaled = 968

W = 320
H = 240
K = copy.deepcopy(intrinsic)
scale_x = 320. / 1296.
scale_y = 240. / 968.


K[0, 0] = K[0, 0]*scale_x # fx
K[1, 1] = K[1, 1]*scale_y # fy
K[0, 2] = K[0, 2]*scale_x  # cx
K[1, 2] = K[1, 2]*scale_y  # cy

if args.semantics:
    #with open(args.pallete, "r") as f:
        #pallete = json.load(f)
    label_mapping_nyu = load_scannet_nyu40_mapping(basedir)
    os.makedirs(os.path.join(basedir, 'label_40'), exist_ok=True)
    os.makedirs(os.path.join(basedir, 'label_40_scaled'), exist_ok=True)


train_ids = frame_ids

print(f"total number of training frames: {len(train_ids)}")

os.makedirs(os.path.join(basedir, 'color_scaled'), exist_ok=True)

for ids in (train_ids,):
    transform_json = {}
    transform_json["fl_x"] = K[0, 0]
    transform_json["fl_y"] = K[1, 1]
    transform_json["cx"] = K[0, 2]
    transform_json["cy"] = K[1, 2]
    transform_json["w"] = W
    transform_json["h"] = H
    transform_json["camera_angle_x"] = np.arctan2(W/2,K[0, 0]) * 2
    transform_json["camera_angle_y"] = np.arctan2(H/2,K[1, 1]) * 2
    transform_json["aabb_scale"] = 16
    transform_json["frames"] = []

    transform_json_unscaled = {}
    transform_json_unscaled["fl_x"] = K_unscaled[0, 0]
    transform_json_unscaled["fl_y"] = K_unscaled[1, 1]
    transform_json_unscaled["cx"] = K_unscaled[0, 2]
    transform_json_unscaled["cy"] = K_unscaled[1, 2]
    transform_json_unscaled["w"] = W_unscaled
    transform_json_unscaled["h"] = H_unscaled
    transform_json_unscaled["camera_angle_x"] = np.arctan2(W_unscaled/2,K_unscaled[0, 0]) * 2
    transform_json_unscaled["camera_angle_y"] = np.arctan2(H_unscaled/2,K_unscaled[1, 1]) * 2
    transform_json_unscaled["aabb_scale"] = 16
    transform_json_unscaled["frames"] = []

    for frame_id in ids:
        pose = np.loadtxt(os.path.join(basedir, 'pose', '%d.txt' % frame_id))
        pose = pose.reshape((4, 4))
        if np.any(np.isinf(pose)):
            continue
        if args.scaled_image:
            file_name_image = os.path.join(basedir, 'color', '%d.jpg'% frame_id)
            image = cv2.imread(file_name_image)[:,:,::-1] # change from BGR uinit 8 to RGB float
            #image = cv2.copyMakeBorder(src=image, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0,0,0]) # pad 4 pixels to height so that images have aspect ratio of 4:3
            #assert image.shape[0] * 4==3 * image.shape[1]
            image = image/255.0
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            image_save = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)
            image_save = image_save * 255.0
            cv2.imwrite(os.path.join(basedir, 'color_scaled', '%d.jpg'% frame_id), image_save)

            if args.semantics:
                file_name_label = os.path.join(basedir, 'label-filt', '%d.png'% frame_id)
                semantic = cv2.imread(file_name_label, cv2.IMREAD_UNCHANGED)
                semantic_copy = copy.deepcopy(semantic)
                for scan_id, nyu_id in label_mapping_nyu.items():
                    semantic[semantic_copy == scan_id] = nyu_id
                # semantic_scaled = cv2.copyMakeBorder(src=semantic, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=0)
                semantic_scaled = cv2.resize(semantic, (W, H), interpolation=cv2.INTER_NEAREST)
                semantic = semantic.astype(np.uint8)
                semantic_scaled = semantic_scaled.astype(np.uint8)
                cv2.imwrite(os.path.join(basedir, 'label_40_scaled', '%d.png'% frame_id), semantic_scaled)
                cv2.imwrite(os.path.join(basedir, 'label_40', '%d.png'% frame_id), semantic)


        json_image_dict ={}
        json_image_dict["file_path"] = os.path.join('color_scaled', '%d.jpg'% frame_id)
        if args.semantics:
            json_image_dict["label_path"] = os.path.join('label_40_scaled', '%d.png'% frame_id)
        json_image_dict["transform_matrix"] = pose.tolist()
        transform_json["frames"].append(json_image_dict)

        json_image_dict_unscaled ={}
        json_image_dict_unscaled["file_path"] = os.path.join('color', '%d.jpg'% frame_id)
        if args.semantics:
            json_image_dict_unscaled["label_path"] = os.path.join('label_40', '%d.png'% frame_id)
        json_image_dict_unscaled["transform_matrix"] = pose.tolist()
        transform_json_unscaled["frames"].append(json_image_dict_unscaled)


    if ids == train_ids:
        file_name =  'transforms_train_scaled'
    else:
        file_name = 'transforms_test_scaled'

    if args.semantics:
        file_name += "_semantics_40"
    file_name += ".json"
    out_file = open(os.path.join(basedir, file_name), "w")
    json.dump(transform_json, out_file, indent = 4)
    out_file.close()

    if ids == train_ids:
        file_name =  'transforms_train'
    else:
        file_name = 'transforms_test'
    if args.semantics:
        file_name += "_semantics_40"
    file_name += ".json"
    out_file = open(os.path.join(basedir, file_name), "w")
    json.dump(transform_json_unscaled, out_file, indent = 4)
    out_file.close()


