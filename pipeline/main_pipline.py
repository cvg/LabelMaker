import logging
import os
import shutil
import sys
from typing import Dict, List

from prefect import flow, task, serve
from prefect_dask.task_runners import DaskTaskRunner

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline.arkitscene_download_preprocess_pipeline import (
    arkitscene_download_and_preprocess,
    get_fold,
    get_scene_download_dir,
)
from pipeline.grounded_sam_pipeline import grounded_sam_inference
from pipeline.video_viz_pipeline import render_semantic_segmentation_video

MODEL_LABELSPACE = {
    'groundedsam': 'wordnet',
    'cmx': 'nyu40',
    'internimage': 'ade20k',
    'ovseg': 'wordnet',
    'mask3d': "scannet200",
}


@flow(
    name='Single Scene Pipeline',
    flow_run_name='scene:{video_id}',
)
def process_single_scene(
    video_id: str,
    download_dir: str,
    target_dir: str,
    models_list: List[Dict],
    preprocessing_sdf_trunc: float = 0.04,
    preprocessing_voxel_length: float = 0.008,
    preprocessing_depth_trunc: float = 3.0,
    gsam_box_threshold: float = 0.25,
    gsam_text_threshold: float = 0.2,
    gsam_iou_threshold: float = 0.5,
    gsam_sam_defect_threshold: int = 30,
):
  """
  An example of models_dict is
  [
    {
      "model": 'groundedsam',
      "flip": True,
      "seed": seed,
      "id": '1',
      "vote": '3',
    }
  ]
  the naming of each folder is {labelspace}_{model}_{vote}_{id}[_flip]
  """
  download_preprocess_flag = arkitscene_download_and_preprocess(
      download_dir=download_dir,
      video_id=video_id,
      target_dir=target_dir,
      sdf_trunc=preprocessing_sdf_trunc,
      voxel_length=preprocessing_voxel_length,
      depth_trunc=preprocessing_depth_trunc,
  )

  intermediate_flags = []
  for item in models_list:
    model = str(item['model'])
    labelspace = str(MODEL_LABELSPACE[model])
    vote = str(item['vote'])
    idx = str(item['id'])
    flip = item['flip']
    seed = int(item['seed'])

    model_folder = f'intermediate/{labelspace}_{model}_{vote}_{idx}'

    if model == 'groundedsam':
      gsam_flag = grounded_sam_inference(
          wait_for=[download_preprocess_flag],
          video_id=video_id,
          scene_dir=target_dir,
          output_folder=model_folder,
          box_threshold=gsam_box_threshold,
          text_threshold=gsam_text_threshold,
          iou_threshold=gsam_iou_threshold,
          sam_defect_threshold=gsam_sam_defect_threshold,
          flip=flip,
          clean_run=False,
          seed=seed,
      )
      intermediate_flags.append(gsam_flag)

    elif model == 'ovseg':
      pass

    else:
      raise NotImplementedError

  # consensus
  # run_consensus(wait_for=intermediate_flags)


if __name__ == "__main__":
  deploys = []
  for video_id in [
      # "41126559",
      # "41126754",
      "41159519",
      # "41159856",
  ]:
    target_dir = f'/scratch/quanta/Experiments/LabelMaker/test_{video_id}'
    # set concurrency of tag by
    # prefect concurrency-limit create {tag} {max_limits}
    # e.g.
    # prefect concurrency-limit create scene 2
    deploy = process_single_scene.to_deployment(
        name=f'Single Scene Process',
        tags=['scene'],  # use this tag to control concurrency
        parameters={
            "video_id":
                video_id,
            "download_dir":
                "/scratch/quanta/Datasets/ARKitScenes",
            'target_dir':
                target_dir,
            "models_list": [
                {
                    "model": 'groundedsam',
                    "flip": False,
                    "seed": 42,
                    "id": '1',
                    "vote": '2',
                },
                {
                    "model": 'groundedsam',
                    "flip": True,
                    "seed": 42,
                    "id": '1',
                    "vote": '2',
                },
            ],
        },
    )
    deploys.append(deploy)

  serve(*deploys)
