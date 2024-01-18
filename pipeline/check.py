import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline.pipeline_utils import check_progress_given_tasks, config_to_tasks, pipeline_config


def check_progress(
    root_dir: str,
    fold: str,
    video_id: str,
    verbose=True,
):
  workspace = Path(root_dir) / fold / video_id


  if not workspace.exists():
    return

  task_results = check_progress_given_tasks(
      workspace=workspace,
      tasks=config_to_tasks(pipeline_config=pipeline_config),
  )

  unfinished_tasks = []
  for task in task_results:
    if task["type"] in ["sdfstudio_post", "stats"]:
      continue

    if not task['finished']:
      unfinished_tasks.append(task['name'])

  if verbose:
    if len(unfinished_tasks) == 0:
      print(f'Scene {video_id} of {fold} split is fully finished!')
    else:
      print(
          f'Scene {video_id} of {fold} split is NOT finished, unfinished task contains:'
      )
      for t in unfinished_tasks:
        print(f'   {t}')

  else:
    print(len(unfinished_tasks) == 0)

    # print(
    #     f'   use the following command: " python ./pipeline/submit.py --root_dir {root_dir} --fold {fold} --video_id {video_id} " to submit the unfinished part of the pipeline!'
    # )


def arg_parser():
  parser = argparse.ArgumentParser(description='Check pipeline.')
  parser.add_argument(
      '--root_dir',
      type=str,
      required=True,
  )
  parser.add_argument(
      '--fold',
      type=str,
      required=True,
  )
  parser.add_argument(
      '--video_id',
      type=str,
      required=True,
  )
  parser.add_argument(
      '--verbose', action="store_true"
  )
  parser.add_argument(
      '--no-verbose', dest='verbose', action="store_false",
  )
  parser.set_defaults(verbose=True)
  return parser.parse_args()


if __name__ == '__main__':
  args = arg_parser()
  check_progress(
      root_dir=args.root_dir,
      fold=args.fold,
      video_id=args.video_id,
      verbose=args.verbose,
  )
