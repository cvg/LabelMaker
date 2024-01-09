import json
from pathlib import Path
from typing import Union
import argparse


def fast_check(scene_dir: Union[str, Path]):
  scene_dir = Path(scene_dir)

  if not (scene_dir.exists() and scene_dir.is_dir()):
    print('    ', f'{str(scene_dir)} does not exists or is not a folder')
    return

  # read keys
  with open(str(scene_dir / 'correspondence.json')) as f:
    corres = json.load(f)

  keys = [int(item['frame_id']) for item in corres]
  scene_size = len(corres)

  # check consensus
  consensus_dir = scene_dir / 'intermediate/consensus'
  if not (consensus_dir.exists() and consensus_dir.is_dir()):
    print('    ', 'consensus does not exists')
    return
  else:
    consensus_files = consensus_dir.glob('*.png')

    pred_files = [f for f in consensus_files if f.stem.isnumeric()]
    aux_files = list(consensus_dir.glob('*_aux.png'))

    if not (len(pred_files) == scene_size and len(aux_files) == scene_size):
      print('    ', 'The consensus files are not complete!')
      pred_missing_keys = [
          f.stem for f in pred_files if int(f.stem) not in keys
      ]
      aux_missing_keys = [
          f.stem[:6] for f in aux_files if int(f.stem[:6]) not in keys
      ]
      if len(pred_missing_keys) > 0:
        print('    ', pred_missing_keys, ' are not processed!')
      if len(aux_missing_keys) > 0:
        print('    ', aux_missing_keys, ' are not processed!')

      return

  # check point lifting
  point_lifting_label = scene_dir / 'labels.txt'
  point_lifting_mesh = scene_dir / 'point_lifted_mesh.ply'
  if not (point_lifting_label.exists() and point_lifting_label.is_file() and
          point_lifting_mesh.exists() and point_lifting_mesh.is_file()):
    print('    ', 'Point lifting fails!')

  # # check sdfstudio training
  # sdfstudio_train_flag = True
  # sdfstudio_train_dir = scene_dir / 'intermediate' / 'sdfstudio_train' / 'neus-facto'
  # if not (sdfstudio_train_dir.exists() and sdfstudio_train_dir.is_dir()):
  #   sdfstudio_train_flag = False
  # else:
  #   possible_dirs = list(sdfstudio_train_dir.glob('*'))
  #   if len(possible_dirs) == 0:
  #     sdfstudio_train_flag = False
  #   sdfstudio_train_dir = possible_dirs[0]

  #   checkpoint_path: Path = sdfstudio_train_dir / 'sdfstudio_models' / 'step-000020000.ckpt'
  #   if not (checkpoint_path.exists() and checkpoint_path.is_file()):
  #     sdfstudio_train_flag = False

  # if sdfstudio_train_flag == False:
  #   print('    ', 'sdfstudio training fails!')
  #   return

  # # check extract mesh
  # sdfstudio_mesh = sdfstudio_train_dir / 'mesh_visible_scaled.ply'
  # if not (sdfstudio_mesh.exists() and sdfstudio_mesh.is_file()):
  #   print('    ', 'sdfstudio extract fails!')

  # # check render
  # sdfstudio_render_dir = scene_dir / 'neus_lifted'
  # if not (sdfstudio_render_dir.exists() and sdfstudio_render_dir.is_dir()):
  #   print('    ', "sdfstudio render folder does not exists")
  # else:
  #   render_files = sdfstudio_render_dir.glob('*.png')

  #   pred_files = [f for f in render_files if f.stem.isnumeric()]

  #   if not (len(pred_files) == scene_size and len(aux_files) == scene_size):
  #     print('    ', 'sdfstudio render files are not complete!')
  #     return


def arg_parser():
  parser = argparse.ArgumentParser(description='Run consensus segmentation')
  parser.add_argument(
      '--workspace',
      type=str,
      required=True,
      help='Path to workspace directory. There should be a "color" folder.',
  )
  return parser.parse_args()


if __name__ == '__main__':
  args = arg_parser()
  fast_check(scene_dir=args.workspace)
