import logging
import os
import sys
from pathlib import Path
from typing import Callable, List, Union
from dask import delayed

import cv2
import numpy as np
from PIL import Image
from prefect import flow, task
from prefect_dask.task_runners import DaskTaskRunner

from labelmaker.label_data import get_ade150, get_nyu40, get_replica, get_scannet200, get_scannet_all, get_wordnet
from labelmaker.utils import get_unprocessed_keys, is_rgb_img, remove_files_by_keys

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from video_vizualization import visualize_image, viz2video

logging.basicConfig(level="INFO")
log = logging.getLogger('Semantic Segmentation Video Visualization')


def get_taskrunner_viz():
  """
    get the task runner, paramters like image resolution and size passed by environment variables.
  """
  mode = os.environ.get('LABELMAKER_SERVER_TYPE', 'local')
  assert mode in ['local', 'slurm']

  taskrunner = None
  if mode == 'local':
    taskrunner = DaskTaskRunner(
        cluster_class="distributed.LocalCluster",
        cluster_kwargs={
            "n_workers": 1,
            "processes": False,
            "threads_per_worker": 8,
            "memory_limit": "3GiB",
        },
    )
  elif mode == 'slurm':
    taskrunner = DaskTaskRunner(
        cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "n_workers":
                1,
            "memory":
                "3GiB",
            "job_cpu":
                8,
            "cores":
                8,
            "interface":
                "access",  # possibles are access, lo, eth0, eth1, eth3
            "walltime":
                "1:00:00",
            "job_extra_directives": [
                "--mem-per-cpu=500M",
                "--output=/cluster/home/guanji/LabelMaker/render_video_%j.out",
            ],
            "job_directives_skip": ["--mem"],
            "job_script_prologue": [
                "module load eth_proxy",
                'export PATH="/cluster/project/cvg/labelmaker/miniconda3/bin:${PATH}"',
                'env_name=labelmaker',
                'eval "$(conda shell.bash hook)"',
                'conda activate $env_name',
                'conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"',
                'conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"',
                'export PYTHONPATH=/cluster/home/guanji/LabelMaker:${PYTHONPATH}',
                'export PYTHONPATH=/cluster/home/guanji/LabelMaker/scripts:${PYTHONPATH}',
            ],
        },
    )
  else:
    raise ValueError

  return taskrunner


@task(
    name='Visualize Single Semantic Segmentation',
    task_run_name=
    'Visualize Single Image name:{name}-labelspace:{label_space}-key:{key:06d}',
    retries=5,
    retry_delay_seconds=0.5,
)
def wrap_visualize_single_image(
    *args,
    name: str,
    label_space: str,
    key: int,
    rgb_path: Path,
    label_path: Path,
    temp_save_dir: Path,
    label_color: np.ndarray,
    label_names: List[str],
    alpha: float,
    resize: float,
    font_size: int,
    skip: bool,
):
  if skip:
    return

  rgb_img = np.asarray(Image.open(str(rgb_path)).convert("RGB"))
  label_img = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
  viz_img = visualize_image(
      rgb=rgb_img,
      label=label_img,
      color_mapping=label_color,
      label_mapping=label_names,
      alpha=alpha,
      resize=resize,
      font_size=font_size,
  )
  Image.fromarray(viz_img).save(str(temp_save_dir / '{:06d}.png'.format(key)))


@task(name='Visualization Check')
def wrap_check(
    *args,
    keys,
    target_dir,
    target_file_template,
):
  unproc_keys = get_unprocessed_keys(
      keys=keys,
      target_dir=target_dir,
      target_file_template=target_file_template,
      validity_fn=is_rgb_img,
  )
  assert len(unproc_keys) == 0


@task(
    name='Render Video',
    task_run_name='Render Video name:{name}-labelspace:{label_space}',
    retries=5,
    retry_delay_seconds=0.5,
)
def wrap_render_video(
    *args,
    name: str,
    label_space: str,
    scene_dir: Union[str, Path],
    image_folder: Union[str, Path],
    output_name: Union[str, Path],
    fps: int = 30,
    image_glob_expression="*.png",
    image_name_to_key_fn=lambda x: int(x.stem),
    verbose: bool = False,
    delete_image_folder: bool = False,
):
  viz2video(
      scene_dir=scene_dir,
      image_folder=image_folder,
      output_name=output_name,
      fps=fps,
      image_glob_expression=image_glob_expression,
      image_name_to_key_fn=image_name_to_key_fn,
      verbose=verbose,
      delete_image_folder=delete_image_folder,
  )


@flow(
    task_runner=get_taskrunner_viz(),
    name='Visualize Semantic Segmentation and Render Video',
    flow_run_name='Visualize and Render name:{name}-labelspace:{label_space}',
    retries=5,
    retry_delay_seconds=1,
)
def render_semantic_segmentation_video(
    *args,
    name: str,
    label_space: str,
    scene_dir: Union[str, Path],
    rgb_folder: Union[str, Path],
    label_folder: Union[str, Path],
    temp_save_folder: Union[str, Path],
    output_video_name: Union[str, Path],
    rgb_glob_expression: str = '*.jpg',
    path_to_key_fn: Callable = lambda x: int(x.stem),
    label_file_template: str = '{k:06d}.png',
    label_glob_expression: str = '*.png',
    fps: int = 30,
    alpha: float = 0.6,
    resize: float = 1.0,
    font_size: int = 13,
    delete_temp_image_folder: bool = True,
):
  # convert str to Path object
  scene_dir = Path(scene_dir)
  rgb_folder = Path(rgb_folder)
  label_folder = Path(label_folder)
  temp_save_folder = Path(temp_save_folder)
  output_video_name = Path(output_video_name)

  # convert folder to path
  rgb_dir = scene_dir / rgb_folder
  label_dir = scene_dir / label_folder
  temp_save_dir = scene_dir / temp_save_folder

  # check
  assert scene_dir.exists() and scene_dir.is_dir()
  assert rgb_dir.exists() and rgb_dir.is_dir()
  assert label_dir.exists() and label_dir.is_dir()

  # check if all rgb files have their corresponding labels
  rgb_files = rgb_dir.glob(rgb_glob_expression)
  rgb_files = sorted(rgb_files, key=path_to_key_fn)
  keys = [path_to_key_fn(path) for path in rgb_files]
  label_files = [label_dir / label_file_template.format(k=key) for key in keys]
  assert all([path.exists() and path.is_file() for path in label_files
             ]), "Not all rgb files has their label, check your folder!"

  # get label set
  get_label_info_fn = {
      "ade20k": get_ade150,
      "nyu40": get_nyu40,
      "scannet200": get_scannet200,
      'wordnet': get_wordnet,
      "scannet": get_scannet_all,
      "replica": get_replica,
  }
  assert label_space in get_label_info_fn.keys()
  label_info = get_label_info_fn[label_space]()

  id2name, id2color = {0: "unknown"}, {0: [0, 0, 0]}
  for item in label_info:
    id2name[item['id']] = item['name'].split('.')[0]
    id2color[item['id']] = item['color']

  id_range = np.array(list(id2name.keys())).max() + 1

  label_names = [""] * id_range
  label_color = np.zeros(shape=(id_range, 3), dtype=np.uint8)

  for idx in id2name.keys():
    label_names[idx] = id2name[idx]
    label_color[idx] = id2color[idx]

  # check unprocessed keys
  unproc_keys = get_unprocessed_keys(
      keys=keys,
      target_dir=temp_save_dir,
      target_file_template=label_file_template,
      validity_fn=is_rgb_img,
  )
  remove_files_by_keys(
      keys=unproc_keys,
      target_dir=temp_save_dir,
      target_file_template=label_file_template,
  )
  log.info('[visualization] All folders and files checked!')

  os.makedirs(str(temp_save_dir), exist_ok=True)

  # now begin processing
  log.info('[visualization] labelling image starts!')
  results = []
  for key, rgb_path, label_path in zip(keys, rgb_files, label_files):
    results.append(
        wrap_visualize_single_image.submit(
            name=name,
            label_space=label_space,
            key=key,
            rgb_path=rgb_path,
            label_path=label_path,
            temp_save_dir=temp_save_dir,
            label_color=label_color,
            label_names=label_names,
            alpha=alpha,
            resize=resize,
            font_size=font_size,
            skip=key not in unproc_keys,
        ))

  # check if all files are intact
  checked_flag = wrap_check.submit(
      *results,
      keys=keys,
      target_dir=temp_save_dir,
      target_file_template=label_file_template,
  )

  # render video
  wrap_render_video(
      checked_flag,
      name=name,
      label_space=label_space,
      scene_dir=scene_dir,
      image_folder=temp_save_folder,
      output_name=output_video_name,
      fps=fps,
      image_glob_expression=label_glob_expression,
      image_name_to_key_fn=path_to_key_fn,
      verbose=False,
      delete_image_folder=delete_temp_image_folder,
  )


if __name__ == "__main__":
  # this is only test code
  mode = os.environ.get("LABELMAKER_SERVER_TYPE", 'local')
  if mode == 'slurm':
    pass

  elif mode == 'local':
    render_semantic_segmentation_video(
        name='TestRender',
        label_space='ade20k',
        scene_dir="/scratch/quanta/Experiments/LabelMaker/replica/room_0_1",
        rgb_folder='color',
        label_folder='intermediate/ade20k_internimage_1',
        temp_save_folder='temp_viz_ade20k_internimage_1',
        output_video_name='intermediate/viz_ade20k_internimage_1.mp4',
        rgb_glob_expression='*.jpg',
        path_to_key_fn=lambda x: int(x.stem),
        label_file_template='{k:06d}.png',
        label_glob_expression='*.png',
        fps=30,
        alpha=0.6,
        resize=1.0,
        font_size=13,
        delete_temp_image_folder=True,
    )

  else:
    raise ValueError
