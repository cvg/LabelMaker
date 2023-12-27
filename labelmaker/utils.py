import json
import shutil
from pathlib import Path
from typing import Any, Callable, List, Union

import cv2
import numpy as np
import open3d as o3d


def get_unprocessed_keys(
    keys: List[Any],
    target_dir: Union[str, Path],
    target_file_template: str,
    validity_fn: Callable = lambda x: True,
):
  """
  This is a universal funciton used in pipeline control.
  Given a list of keys (stem of file names), we want the corresponding files exists in the target folder. This function returns the unprocessed keys that needs to be process.

  This function is helpful when sometimes the task crushes in the half way and need a partial rerun.

  keys: List of string denoting the stem of the file
  target_file_template: a template string that can convert from key to the file name, for example: "{k}.png", this template should only take one argument, with keyword k
  validity_fn: a function to test if the file is readable and in desired format.
  """
  if len(keys) == 0:
    return []

  target_dir = Path(target_dir)

  try:
    target_file_template.format(k=keys[0])
  except:
    raise ValueError(
        "The target_file_template you passed fails to be called. It has to accept 'k' as argument, and only takes one argument. '{k}.png' is an example."
    )

  unproc_keys = []
  for key in keys:
    target_file = target_dir / target_file_template.format(k=key)
    if not target_file.exists() or not validity_fn(target_file):
      unproc_keys.append(key)

  return unproc_keys


def remove_files_by_keys(
    keys: List[Any],
    target_dir: Union[str, Path],
    target_file_template: str,
):
  """Used for removing unprocessed/failed files"""
  if len(keys) == 0:
    return

  target_dir = Path(target_dir)

  try:
    target_file_template.format(k=keys[0])
  except:
    raise ValueError(
        "The target_file_template you passed fails to be called. It has to accept 'k' as argument, and only takes one argument. '{k}.png' is an example."
    )

  for key in keys:
    target_file = target_dir / target_file_template.format(k=key)
    shutil.rmtree(target_file, ignore_errors=True)


"""Here are some validity checking funcitons"""


def is_file(x: Path):
  return x.is_file()


def is_uint8_img(x: Path):
  if x.is_file():
    try:
      img = cv2.imread(str(x), cv2.IMREAD_UNCHANGED)
      assert img.dtype == np.uint8
      return True

    except:
      return False
  else:
    return False


def is_uint16_img(x: Path):
  if x.is_file():
    try:
      img = cv2.imread(str(x), cv2.IMREAD_UNCHANGED)
      assert img.dtype == np.uint16
      return True

    except:
      return False
  else:
    return False


def is_rgb_img(x: Path):
  if x.is_file():
    try:
      img = cv2.imread(str(x), cv2.IMREAD_UNCHANGED)
      assert img.shape[2] == 3
      assert img.dtype == np.uint8
      return True

    except:
      return False
  else:
    return False


def numpy_dtype_shape(x: Path, dtype: np.dtype, shape: List[int]):
  """need to use with lambda expression"""
  if x.is_file():
    try:
      a = np.load(str(x))
      assert isinstance(a, np.ndarray)
      assert list(a.shape) == list(shape)
      assert a.dtype == dtype
      return True

    except:
      return False
  else:
    return False


def np_txt_dtype_shape(x: Path, dtype: np.dtype, shape: List[int]):
  if x.is_file():
    try:
      a = np.loadtxt(str(x))
      assert isinstance(a, np.ndarray)
      assert list(a.shape) == list(shape)
      assert a.dtype == dtype
      return True

    except:
      return False
  else:
    return False


def get_keys(scene_dir: Union[str, Path]):
  """get the keys (int) of this scene"""
  scene_dir = Path(scene_dir)
  assert scene_dir.exists() and scene_dir.is_dir()
  corres_path = scene_dir / 'correspondence.json'
  assert corres_path.exists() and corres_path.is_file()

  with open(str(corres_path), 'r') as f:
    data = json.load(f)

  keys = sorted([int(item['frame_id']) for item in data])

  return keys


def check_scene_in_labelmaker_format(scene_dir: Union[str, Path]):
  """
  This function checks if the scene directory satisfies the format of labelmaker format. If not, it will throw error.

  For a typical scene with 1785 keys, I use 5.4s for checking.
  """
  scene_dir = Path(scene_dir)
  keys = get_keys(scene_dir)  # this step already checks corres file

  # check color folder, 991 it/s
  color_dir = scene_dir / 'color'
  assert color_dir.exists() and color_dir.is_dir()
  unproc_keys = get_unprocessed_keys(
      keys=keys,
      target_dir=color_dir,
      target_file_template='{k:06d}.jpg',
      validity_fn=is_rgb_img,
  )
  assert len(unproc_keys) == 0

  # check depth folder, 575 it/s
  depth_dir = scene_dir / 'depth'
  assert depth_dir.exists() and depth_dir.is_dir()
  unproc_keys = get_unprocessed_keys(
      keys=keys,
      target_dir=depth_dir,
      target_file_template='{k:06d}.png',
      validity_fn=is_uint16_img,
  )
  assert len(unproc_keys) == 0

  # check intrinsic folder, 0.0s
  intrinsic_dir = scene_dir / 'intrinsic'
  assert intrinsic_dir.exists() and intrinsic_dir.is_dir()
  unproc_keys = get_unprocessed_keys(
      keys=keys,
      target_dir=intrinsic_dir,
      target_file_template='{k:06d}.txt',
      validity_fn=lambda x: np_txt_dtype_shape(
          x, dtype=np.float64, shape=[3, 3]),
  )
  assert len(unproc_keys) == 0

  # check pose folder, 8925.0 it/s
  pose_dir = scene_dir / 'pose'
  assert pose_dir.exists() and pose_dir.is_dir()
  unproc_keys = get_unprocessed_keys(
      keys=keys,
      target_dir=pose_dir,
      target_file_template='{k:06d}.txt',
      validity_fn=lambda x: np_txt_dtype_shape(
          x, dtype=np.float64, shape=[4, 4]),
  )
  assert len(unproc_keys) == 0

  mesh_path = scene_dir / 'mesh.ply'
  assert mesh_path.exists() and mesh_path.is_file()
  try:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    assert mesh.has_vertices() and mesh.has_triangles(
    ) and mesh.has_vertex_colors()
  except:
    assert False
