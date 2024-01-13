import json
from pathlib import Path
from typing import Dict, List

MODELS = {
    "gsam",
    "ovseg",
    "cmx",
    "internimage",
    "mask3d",
}
LABEL_SPACE = {
    "scannet200",
    "wordnet",
    "ade20k",
    "nyu40",
}
pipeline_config = {
    "intermediate_models": [
        {
            'model': "gsam",
            "flip": False,
            "vote": 1,
        },
        {
            'model': "gsam",
            "flip": True,
            "vote": 1,
        },
        {
            'model': "ovseg",
            "flip": False,
            "vote": 1,
        },
        {
            'model': "ovseg",
            "flip": True,
            "vote": 1,
        },
        {
            'model': "cmx",
            "flip": False,
            "vote": 1,
        },
        {
            'model': "cmx",
            "flip": True,
            "vote": 1,
        },
        {
            'model': "internimage",
            "flip": False,
            "vote": 1,
        },
        {
            'model': "internimage",
            "flip": True,
            "vote": 1,
        },
        {
            'model': "mask3d",
            "seed": 42,
            "flip": False,
            "vote": 1,
        },
        {
            'model': "mask3d",
            "seed": 43,
            "flip": False,
            "vote": 1,
        },
    ],
    "3D_lifting": True,
}
TASK_TYPE = {
    'download_preprocessing',
    'render',
    'gsam',
    'internimage',
    'mask3d',
    'cmx',
    'ovseg',
    'hha',
    'omnidepth',
    'consensus',
    'point_lifting',
    'sdfstudio_train',
    'sdfstudio_extract',
    'sdfstudio_render',
    'sdfstudio_post',
    'stats',
}


def config_to_tasks(pipeline_config: Dict):
  tasks = []

  # first download and preprocessing
  tasks.append({
      "type": "download_preprocessing",
      "name": "download_preprocessing",
      "flag": "download_preprocessing_flag",
      "dependency": [],  # dependency is a list to store flag name
  })

  # then run all intermediate models

  gsam_counts = 0
  gsam_flip_counts = 0
  ovseg_counts = 0
  ovseg_flip_counts = 0
  internimage_counts = 0
  internimage_flip_counts = 0
  cmx_counts = 0
  cmx_flip_counts = 0
  mask3d_counts = 0

  # first check if cmx is needed
  if 'cmx' in [
      item['model'] for item in pipeline_config['intermediate_models']
  ]:
    tasks.append({
        "type": "omnidepth",
        "name": "omnidepth",
        "flag": "omnidepth_flag",
        "dependency": ["download_preprocessing_flag",],
    })

    tasks.append({
        "type": "hha",
        "name": "hha",  # appear in squeue
        "flag": "hha_flag",
        "dependency": ["omnidepth_flag",],
    })

  consensus_dependencies = []

  for task in pipeline_config['intermediate_models']:

    if task['model'] == "gsam":
      if not task['flip']:
        gsam_counts += 1
        name = f"gsam_{gsam_counts}"
        output_folder_args = f"wordnet_groundedsam_{task['vote']}_{gsam_counts}"
        real_output_folder = output_folder_args

      else:
        gsam_flip_counts += 1
        name = f"gsam_flip_{gsam_flip_counts}"
        output_folder_args = f"wordnet_groundedsam_{task['vote']}_{gsam_flip_counts}"
        real_output_folder = output_folder_args + "_flip"

      flag = name + "_flag"

      tasks.append({
          "type": "gsam",
          "name": name,
          "flag": flag,
          "flip": task['flip'],
          "output_folder_args": output_folder_args,
          "real_output_folder": real_output_folder,
          "dependency": ["download_preprocessing_flag",],
      })

      # rendering
      video_render_name = real_output_folder + '_viz.mp4'
      tasks.append({
          "type": "render",
          'name': "render_" + name,
          "label_space": "wordnet",
          "video_render_name": video_render_name,
          "rel_path": f"intermediate/{real_output_folder}",
          "dependency": [flag],
      })

    elif task['model'] == "ovseg":
      if not task['flip']:
        ovseg_counts += 1
        name = f"ovseg_{ovseg_counts}"
        output_folder_args = f"wordnet_ovseg_{task['vote']}_{ovseg_counts}"
        real_output_folder = output_folder_args

      else:
        ovseg_flip_counts += 1
        name = f"ovseg_flip_{ovseg_flip_counts}"
        output_folder_args = f"wordnet_ovseg_{task['vote']}_{ovseg_flip_counts}"
        real_output_folder = output_folder_args + "_flip"

      flag = name + "_flag"

      tasks.append({
          "type": "ovseg",
          "name": name,
          "flag": flag,
          "flip": task['flip'],
          "output_folder_args": output_folder_args,
          "real_output_folder": real_output_folder,
          "dependency": ["download_preprocessing_flag",],
      })

      # rendering
      video_render_name = real_output_folder + '_viz.mp4'
      tasks.append({
          "type": "render",
          'name': "render_" + name,
          "label_space": "wordnet",
          "video_render_name": video_render_name,
          "rel_path": f"intermediate/{real_output_folder}",
          "dependency": [flag],
      })

    elif task['model'] == "internimage":
      if not task['flip']:
        internimage_counts += 1
        name = f"internimage_{internimage_counts}"
        output_folder_args = f"ade20k_internimage_{task['vote']}_{internimage_counts}"
        real_output_folder = output_folder_args

      else:
        internimage_flip_counts += 1
        name = f"internimage_flip_{internimage_flip_counts}"
        output_folder_args = f"ade20k_internimage_{task['vote']}_{internimage_flip_counts}"
        real_output_folder = output_folder_args + "_flip"

      flag = name + "_flag"

      tasks.append({
          "type": "internimage",
          "name": name,
          "flag": flag,
          "flip": task['flip'],
          "output_folder_args": output_folder_args,
          "real_output_folder": real_output_folder,
          "dependency": ["download_preprocessing_flag",],
      })

      # rendering
      video_render_name = real_output_folder + '_viz.mp4'
      tasks.append({
          "type": "render",
          'name': "render_" + name,
          "label_space": "ade20k",
          "video_render_name": video_render_name,
          "rel_path": f"intermediate/{real_output_folder}",
          "dependency": [flag],
      })

    elif task['model'] == "mask3d":
      mask3d_counts += 1
      name = f"mask3d_{mask3d_counts}"
      output_folder_args = f"scannet200_mask3d_{task['vote']}_{mask3d_counts}"
      real_output_folder = output_folder_args
      flag = name + "_flag"

      tasks.append({
          "type": "mask3d",
          "name": name,
          "flag": flag,
          "flip": task['flip'],
          "seed": task['seed'],
          "output_folder_args": output_folder_args,
          "real_output_folder": real_output_folder,
          "dependency": ["download_preprocessing_flag",],
      })

      # rendering
      video_render_name = real_output_folder + '_viz.mp4'
      tasks.append({
          "type": "render",
          'name': "render_" + name,
          "label_space": "scannet",
          "video_render_name": video_render_name,
          "rel_path": f"intermediate/{real_output_folder}",
          "dependency": [flag],
      })

    elif task['model'] == "cmx":
      if not task['flip']:
        cmx_counts += 1
        name = f"cmx_{cmx_counts}"
        output_folder_args = f"nyu40_cmx_{task['vote']}_{cmx_counts}"
        real_output_folder = output_folder_args

      else:
        cmx_flip_counts += 1
        name = f"cmx_flip_{cmx_flip_counts}"
        output_folder_args = f"nyu40_cmx_{task['vote']}_{cmx_flip_counts}"
        real_output_folder = output_folder_args + "_flip"

      flag = name + "_flag"

      tasks.append({
          "type": "cmx",
          "name": name,
          "flag": flag,
          "flip": task['flip'],
          "output_folder_args": output_folder_args,
          "real_output_folder": real_output_folder,
          "dependency": ["hha_flag",],
      })

      # rendering
      video_render_name = real_output_folder + '_viz.mp4'
      tasks.append({
          "type": "render",
          'name': "render_" + name,
          "label_space": "nyu40",
          "video_render_name": video_render_name,
          "rel_path": f"intermediate/{real_output_folder}",
          "dependency": [flag],
      })

    else:
      raise NotImplementedError

    consensus_dependencies.append(flag)

  # consensus
  tasks.append({
      "type": "consensus",
      'name': "consensus",
      "flag": "consensus_flag",
      "dependency": consensus_dependencies,
  })

  # consensus rendering
  tasks.append({
      "type": "render",
      'name': "render_consensus",
      "label_space": "wordnet",
      "video_render_name": "consensus_viz.mp4",
      "rel_path": f"intermediate/consensus",
      "dependency": ["consensus_flag"],
  })

  # point_lifting
  tasks.append({
      "type": "point_lifting",
      "name": "point_lifting",
      "flag": "point_lifting_flag",
      "dependency": ['consensus_flag'],
  })

  if pipeline_config["3D_lifting"]:
    # sdfstudio training
    tasks.append({
        "type": "sdfstudio_train",
        "name": "sdfstudio_train",
        "flag": "sdfstudio_train_flag",
        "dependency": ['consensus_flag'],
    })

    # sdfstudio extraction
    tasks.append({
        "type": "sdfstudio_extract",
        "name": "sdfstudio_extract",
        "flag": "sdfstudio_extract_flag",
        "dependency": ['sdfstudio_train_flag'],
    })

    # sdfstudio render
    tasks.append({
        "type": "sdfstudio_render",
        "name": "sdfstudio_render",
        "flag": "sdfstudio_render_flag",
        "dependency": ['sdfstudio_train_flag'],
    })

    # render visualization
    tasks.append({
        "type": "render",
        'name': "render_neus",
        "label_space": "wordnet",
        "video_render_name": "neus_lifted_viz.mp4",
        "rel_path": f"neus_lifted",
        "dependency": ["sdfstudio_render_flag"],
    })

    # render visualization
    tasks.append({
        "type": "sdfstudio_post",
        'name': "sdfstudio_post",
        "dependency": [
            'sdfstudio_render_flag',
            "sdfstudio_extract_flag",
        ],
    })

  return tasks


def check_folder(
    workspace: str,
    keys: List[int] = [],
    files_templates: List[str] = [],
    auxilary_files: List[str] = [],
) -> bool:

  workspace = Path(workspace)

  # check all templates
  for template in files_templates:
    for k in keys:
      file_path: Path = workspace / template.format(key=k)
      if not (file_path.exists() and file_path.is_file()):
        return False

  # check all auxilary_files
  for file_rel_path in auxilary_files:
    file_path: Path = workspace / file_rel_path
    if not (file_path.exists() and file_path.is_file()):
      return False

  return True


def check_progress_given_tasks(workspace: str, tasks: List[Dict]):
  workspace = Path(workspace)

  for task in tasks:
    task['finished'] = False

  # first check if correspondence exists and get keys
  keys = None
  corres_file: Path = workspace / "correspondence.json"
  if corres_file.exists() and corres_file.is_file():
    with open(str(corres_file)) as f:
      keys = [int(item['frame_id']) for item in json.load(f)]
  else:
    return tasks

  for task in tasks:
    if task['type'] == "download_preprocessing":
      task['finished'] = check_folder(
          workspace=workspace,
          keys=keys,
          files_templates=[
              "color/{key:06d}.jpg",
              "depth/{key:06d}.png",
              "intrinsic/{key:06d}.txt",
              "pose/{key:06d}.txt",
          ],
          auxilary_files=[
              "correspondence.json",
              "mesh.ply",
          ],
      )

    elif task["type"] == "render":
      task['finished'] = check_folder(
          workspace=workspace,
          auxilary_files=[
              f"video/{task['video_render_name']}",
          ],
      )

    elif task['type'] == "omnidepth":
      task['finished'] = check_folder(
          workspace=workspace,
          keys=keys,
          files_templates=[
              "intermediate/depth_omnidata_1/{key:06d}.png",
          ],
      )

    elif task['type'] == "hha":
      task['finished'] = check_folder(
          workspace=workspace,
          keys=keys,
          files_templates=[
              "intermediate/hha/{key:06d}.png",
          ],
      )

    elif task['type'] in ["gsam", "cmx", "ovseg", "internimage", "mask3d"]:
      task['finished'] = check_folder(
          workspace=workspace,
          keys=keys,
          files_templates=[
              f"intermediate/{task['real_output_folder']}" + "/{key:06d}.png",
          ],
      )

    elif task['type'] == "consensus":
      task['finished'] = check_folder(
          workspace=workspace,
          keys=keys,
          files_templates=[
              "intermediate/consensus/{key:06d}.png",
              "intermediate/consensus/{key:06d}_aux.png",
          ],
      )

    elif task['type'] == "point_lifting":
      task['finished'] = check_folder(
          workspace=workspace,
          auxilary_files=[
              "labels.txt",
              "point_lifted_mesh.ply",
          ],
      )

    elif task['type'] == "sdfstudio_train":
      for p in (workspace / 'intermediate/sdfstudio_train').glob("**/*"):
        if ".ckpt" in str(p):
          task["finished"] = True
          break

    elif task['type'] == "sdfstudio_extract":
      for p in (workspace / 'intermediate/sdfstudio_train').glob("**/*"):
        if "mesh_visible_scaled.ply" in str(p):
          task["finished"] = True
          break

    elif task['type'] == "sdfstudio_render":
      task['finished'] = check_folder(
          workspace=workspace,
          keys=keys,
          files_templates=[
              "neus_lifted/{key:05d}.png",
          ],
      )

    elif task['type'] in ["sdfstudio_post", "stats"]:
      pass

    else:
      raise NotImplementedError

  return tasks
