from ast import Store
import copy
import glob
import json
import argparse
from tkinter import Pack
import numpy as np
# import open3d as o3d
import os
from pathlib import Path


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def closest_point_2_lines(
    oa, da, ob, db
):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


parser = argparse.ArgumentParser(
    description=
    "Run neural graphics primitives testbed with additional configuration & output options"
)

parser.add_argument("--scene_folder", type=str, default="")
parser.add_argument(
    "--transform_file",
    type=str,
    default="",
)

parser.add_argument("--interval",
                    default=10,
                    type=int,
                    help="Sample Interval.")

args = parser.parse_args()

scannet_folder = args.scene_folder
transform_json = args.transform_file
interval = args.interval
json_train_base = Path(transform_file).stem
# Select the frames from the json. This are the frames of which we want to find
# the actual transform.
c2ws = []
frame_names = []
with open(json_train, "r") as f:
    transforms = json.load(f)
# - Get filenames and concurrently load the c2w.
for frame_idx, frame in enumerate(transforms['frames']):
    if (frame_idx % interval == 0):
        frame_name = os.path.basename(frame['file_path']).split('.jpg')[0]
        pose_name = os.path.join(scannet_folder, f"pose/{frame_name}.txt")
        c2w = np.loadtxt(pose_name)
        if np.any(np.isinf(c2w)):
            continue
        frame_names.append(frame_name)
        c2ws.append(c2w)
selected_transforms = copy.deepcopy(transforms_train)
selected_transforms.pop('frames')
selected_transforms['frames'] = []

if args.transform_test:
    json_test = args.transform_test
    json_test_base = Path(json_test).stem
    c2ws_test = []
    frame_names_test = []
    with open(json_test, "r") as f:
        transforms_test = json.load(f)
    # - Get filenames and concurrently load the c2w.
    for frame_idx, frame in enumerate(transforms_test['frames']):
        if (frame_idx % interval == 0):
            frame_name = os.path.basename(frame['file_path']).split('.jpg')[0]
            pose_name = os.path.join(scannet_folder, f"pose/{frame_name}.txt")
            c2w = np.loadtxt(pose_name)
            if np.any(np.isinf(c2w)):
                continue
            frame_names_test.append(frame_name)
            c2ws_test.append(c2w)
    selected_transforms_test = copy.deepcopy(transforms_test)
    selected_transforms_test.pop('frames')
    selected_transforms_test['frames'] = []

# Open the mesh file to retrieve the scene center.
if args.room_center:
    mesh_files = glob.glob(os.path.join(scannet_folder, "*_vh_clean.ply"))
    assert (len(mesh_files) == 1), (
        "Found no/more than 1 'vh_clean' mesh files in "
        f"{scannet_folder}.")

    mesh = o3d.io.read_triangle_mesh(mesh_files[0])
    max_coord_mesh = np.max(mesh.vertices, axis=0)
    min_coord_mesh = np.min(mesh.vertices, axis=0)
    room_center = (max_coord_mesh + min_coord_mesh) / 2.
else:
    room_center = np.zeros(3)

up = np.zeros(3)
print(f"length of c2ws: {len(c2ws)}")
for c2w_idx in range(len(c2ws)):
    c2ws[c2w_idx][:3, 3] -= room_center
    c2ws[c2w_idx][0:3, 2] *= -1  # flip the y and z axis
    c2ws[c2w_idx][0:3, 1] *= -1
    c2ws[c2w_idx] = c2ws[c2w_idx][[1, 0, 2, 3], :]  # swap y and z
    c2ws[c2w_idx][2, :] *= -1  # flip whole world upside down
    up += c2ws[c2w_idx][0:3, 1]

if args.transform_test:
    for c2w_idx in range(len(c2ws_test)):
        c2ws_test[c2w_idx][:3, 3] -= room_center
        c2ws_test[c2w_idx][0:3, 2] *= -1  # flip the y and z axis
        c2ws_test[c2w_idx][0:3, 1] *= -1
        c2ws_test[c2w_idx] = c2ws_test[c2w_idx][[1, 0, 2,
                                                 3], :]  # swap y and z
        c2ws_test[c2w_idx][2, :] *= -1  # flip whole world upside down

print(f"up vector: {up}")

nframes = len(c2ws)
up = up / np.linalg.norm(up)
print("up vector was", up)
R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
R = np.pad(R, [0, 1])
R[-1, -1] = 1

for c2w_idx in range(len(c2ws)):
    c2ws[c2w_idx] = np.matmul(R, c2ws[c2w_idx])  # rotate up to be the z axis

if args.transform_test:
    for c2w_idx in range(len(c2ws_test)):
        c2ws_test[c2w_idx] = np.matmul(
            R, c2ws_test[c2w_idx])  # rotate up to be the z axis

# find a central point they are all looking at
if not args.room_center:
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for c2w_idx_1 in range(len(c2ws)):
        mf = c2ws[c2w_idx_1][0:3, :]
        for c2w_idx_2 in range(len(c2ws)):
            mg = c2ws[c2w_idx_2][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:,
                                                                          2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print("room center was:")
    print(totp)  # the cameras are looking at totp
    for c2w_idx in range(len(c2ws)):
        c2ws[c2w_idx][0:3, 3] -= totp

    if args.transform_test:
        for c2w_idx in range(len(c2ws_test)):
            c2ws_test[c2w_idx][0:3, 3] -= totp

avglen = 0.
for c2w_idx in range(len(c2ws)):
    avglen += np.linalg.norm(c2ws[c2w_idx][0:3, 3])
print(f"avglen:{avglen}")
print(nframes)
avglen /= nframes

# This factor converts one meter to the unit of measure of the scene.
# NOTE: This incorporates both the scaling previously done in this script
# and the one previously done in `nerf_matrix_to_ngp`, which now no longer
# scales the poses.
one_m_to_scene_uom = 4.0 / avglen * 0.33

print("avg camera distance from origin", avglen)
for c2w_idx in range(len(c2ws)):
    c2ws[c2w_idx][0:3, 3] *= one_m_to_scene_uom  # scale to "nerf sized"
if args.transform_test:
    for c2w_idx in range(len(c2ws_test)):
        c2ws_test[c2w_idx][0:3,
                           3] *= one_m_to_scene_uom  # scale to "nerf sized"

store_dict = {}
store_dict["avglen"] = avglen
store_dict["up"] = up
store_dict["totp"] = totp
store_dict["totw"] = totw
scene_name = os.path.basename(os.path.dirname(scannet_folder))
np.savez(f"transform_info_{scene_name}.npz", store_dict)

curr_frame_name_idx = 0
for frame_idx in range(len(transforms_train['frames'])):
    if (curr_frame_name_idx == len(frame_names)):
        break
    frame = transforms_train['frames'][frame_idx]
    frame_name = os.path.basename(frame['file_path']).split('.jpg')[0]
    if (frame_name == frame_names[curr_frame_name_idx]):
        c2w = c2ws[curr_frame_name_idx]
        transforms_train['frames'][frame_idx]['transform_matrix'] = c2w.tolist(
        )
        selected_transforms['frames'].append(
            transforms_train['frames'][frame_idx])
        curr_frame_name_idx += 1
selected_transforms['one_m_to_scene_uom'] = one_m_to_scene_uom

out_path = os.path.join(scannet_folder,
                        f"{json_train_base}_{interval}_modified.json")
with open(out_path, "w") as f:
    json.dump(selected_transforms, f, indent=4)

if args.transform_test:
    curr_frame_name_idx = 0
    for frame_idx in range(len(transforms_test['frames'])):
        if (curr_frame_name_idx == len(frame_names_test)):
            break
        frame = transforms_test['frames'][frame_idx]
        frame_name = os.path.basename(frame['file_path']).split('.jpg')[0]
        if (frame_name == frame_names_test[curr_frame_name_idx]):
            c2w = c2ws_test[curr_frame_name_idx]
            transforms_test['frames'][frame_idx][
                'transform_matrix'] = c2w.tolist()
            selected_transforms_test['frames'].append(
                transforms_test['frames'][frame_idx])
            curr_frame_name_idx += 1
    selected_transforms_test['one_m_to_scene_uom'] = one_m_to_scene_uom

    out_path = os.path.join(scannet_folder,
                            f"{json_test_base}_{interval}_modified.json")
    with open(out_path, "w") as f:
        json.dump(selected_transforms_test, f, indent=4)
