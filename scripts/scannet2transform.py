
from email.mime import base
import json
import argparse
import os
import numpy as np
import cv2
import copy
import csv
from pathlib import Path
from PIL import Image
# from utils import nyu40_colour_code


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


def scale_data(scene_dir, keys, template):
    scene_dir = Path(scene_dir)
    for k in keys:
        img = Image.open(scene_dir / template.format(k=k))
        img = img.resize((320, 240), Image.NEAREST)
        scaled_dir = template.split('/')
        scaled_dir[-2] = scaled_dir[-2] + '_scaled'
        scaled_dir = '/'.join(scaled_dir)
        (scene_dir / scaled_dir.format(k=k)).parent.mkdir(parents=True, exist_ok=True)
        img.save(scene_dir / scaled_dir.format(k=k))


def process_scene(scene_dir, keys, img_template, pose_template='pose/{k}.txt', semantics=False, scaled_images=True):
    print(f"processing folder: {scene_dir}")
    # Step for generating training images
    step = 1

    scene_dir = Path(scene_dir)
    basedir = scene_dir
    frame_ids =  sorted(keys)

    intrinsic_file = os.path.join(basedir,"intrinsic/intrinsic_color.txt")
    intrinsic = np.loadtxt(intrinsic_file)
    print("intrinsic parameters:")
    print(intrinsic)

    imgs = []
    poses = []

    K_unscaled = copy.deepcopy(intrinsic)

    sample_img = cv2.imread(str(scene_dir / img_template.format(k=frame_ids[0])))
    #W_unscaled = 1296
    #H_unscaled = 968
    W_unscaled = sample_img.shape[1]
    H_unscaled = sample_img.shape[0]

    W = 320
    H = 240
    K = copy.deepcopy(intrinsic)
    scale_x = 320. / float(W_unscaled)
    scale_y = 240. / float(H_unscaled)


    K[0, 0] = K[0, 0]*scale_x # fx
    K[1, 1] = K[1, 1]*scale_y # fy
    K[0, 2] = K[0, 2]*scale_x  # cx
    K[1, 2] = K[1, 2]*scale_y  # cy

    if semantics:
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
            pose = np.loadtxt(os.path.join(basedir, pose_template.format(k=frame_id)))
            pose = pose.reshape((4, 4))
            if np.any(np.isinf(pose)):
                continue
            if scaled_images:
                #file_name_image = os.path.join(basedir, 'color', '%d.jpg'% frame_id)
                file_name_image = str(scene_dir / img_template.format(k=frame_id))
                image = cv2.imread(file_name_image)[:,:,::-1] # change from BGR uinit 8 to RGB float
                #image = cv2.copyMakeBorder(src=image, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0,0,0]) # pad 4 pixels to height so that images have aspect ratio of 4:3
                #assert image.shape[0] * 4==3 * image.shape[1]
                image = image/255.0
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                image_save = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)
                image_save = image_save * 255.0
                cv2.imwrite(os.path.join(basedir, 'color_scaled', '%d.jpg'% frame_id), image_save)

                if semantics:
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
            json_image_dict["frame_id"] = frame_id
            if semantics:
                json_image_dict["label_path"] = os.path.join('label_40_scaled', '%d.png'% frame_id)
            json_image_dict["transform_matrix"] = pose.tolist()
            transform_json["frames"].append(json_image_dict)

            json_image_dict_unscaled ={}
            json_image_dict_unscaled["file_path"] = img_template.format(k=frame_id)
            json_image_dict_unscaled["frame_id"] = frame_id
            if semantics:
                json_image_dict_unscaled["label_path"] = os.path.join('label_40', '%d.png'% frame_id)
            json_image_dict_unscaled["transform_matrix"] = pose.tolist()
            transform_json_unscaled["frames"].append(json_image_dict_unscaled)


        if ids == train_ids:
            file_name =  'transforms_train_scaled'
        else:
            file_name = 'transforms_test_scaled'

        if semantics:
            file_name += "_semantics_40"
        file_name += ".json"
        out_file = open(os.path.join(basedir, file_name), "w")
        json.dump(transform_json, out_file, indent = 4)
        out_file.close()

        if ids == train_ids:
            file_name =  'transforms_train'
        else:
            file_name = 'transforms_test'
        if semantics:
            file_name += "_semantics_40"
        file_name += ".json"
        out_file = open(os.path.join(basedir, file_name), "w")
        json.dump(transform_json_unscaled, out_file, indent = 4)
        out_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

    parser.add_argument("scene", type=str)
    parser.add_argument('--replica', default=False)
    parser.add_argument("--semantics", default=False)
    parser.add_argument("--pallete", type=str, default="")
    parser.add_argument('--pose_mode', type=str, default='pose_ba', help='pose folder to be used from pose refinement')

    flags = parser.parse_args()
    scene_dir = Path(flags.scene)
    assert scene_dir.exists() and scene_dir.is_dir()
    if flags.replica:
        keys = sorted(
            int(x.name.split('.')[0].split('_')[1])
            for x in (scene_dir / 'rgb').iterdir())
        img_template = 'rgb/rgb_{k}.png'
    else:
        keys = sorted(
            int(x.name.split('.')[0]) for x in (scene_dir / 'color').iterdir())
        img_template = 'color/{k}.jpg'

    process_scene(scene_dir, keys, img_template, pose_template=f'{flags.pose_mode}' + '/{k}.txt')
    
    
    # scale_data(scene_dir, keys, 'pred_consensus/{k}.png')



