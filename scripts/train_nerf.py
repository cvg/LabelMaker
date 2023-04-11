from nr4seg import ROOT_DIR
from nr4seg.lightning import LightningNerf, DataModuleNerf
from nr4seg.utils import load_yaml, flatten_dict, get_wandb_logger

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import argparse
import os
from pathlib import Path
import datetime
import shutil
import coloredlogs

coloredlogs.install()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="cfg/exp/debug.yml",
        help=
        ("Experiment yaml file path relative to template_project_name/cfg/exp "
         "directory."),
    )
    parser.add_argument(
        "--exp_name",
        default="debug",
        help="overall experiment of this continual learning experiment.",
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--project_name", default="test_one_by_one")
    parser.add_argument("--nerf_train_epoch", default=10, type=int)

    args = parser.parse_args()
    return args


def train(exp, env, exp_cfg_path, env_cfg_path, args) -> float:
    seed_everything(args.seed)
    exp["exp_name"] = args.exp_name

    # Create experiment folder.
    if exp["general"]["timestamp"]:
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
        model_path = os.path.join(env["results"], exp["general"]["name"])
        p = model_path.rfind("/") + 1
        model_path = model_path[:p] + str(timestamp) + "_" + model_path[p:]
    else:
        model_path = os.path.join(env["results"], exp["general"]["name"])
        if exp["general"]["clean_up_folder_if_exists"]:
            shutil.rmtree(model_path, ignore_errors=True)

    # Create the directory
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # Copy config files
    exp_cfg_fn = os.path.split(exp_cfg_path)[-1]
    env_cfg_fn = os.path.split(env_cfg_path)[-1]
    print(f"Copy {env_cfg_path} to {model_path}/{exp_cfg_fn}")
    shutil.copy(exp_cfg_path, f"{model_path}/{exp_cfg_fn}")
    shutil.copy(env_cfg_path, f"{model_path}/{env_cfg_fn}")
    exp["general"]["name"] = model_path

    # Create logger.
    logger = get_wandb_logger(exp=exp,
                              env=env,
                              exp_p=exp_cfg_path,
                              env_p=env_cfg_path,
                              project_name='pose_refinement',
                              save_dir=model_path)
    ex = flatten_dict(exp)
    # logger.log_hyperparams(ex)

    # Create network and dataset.
    model = LightningNerf(exp, env)
    datamodule = DataModuleNerf(exp, env)
    datamodule.setup()

    # Trainer setup.
    # - Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    cb_ls = [lr_monitor]

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        save_last=True,
        save_top_k=1,
    )
    cb_ls.append(checkpoint_callback)
    # - Set GPUs.
    if (exp["trainer"]).get("gpus", -1) == -1:
        nr = torch.cuda.device_count()
        print(f"Set GPU Count for Trainer to {nr}!")
        for i in range(nr):
            print(f"Device {i}: ", torch.cuda.get_device_name(i))
        exp["trainer"]["gpus"] = nr

    # - Check whether to restore checkpoint.
    if exp["trainer"]["resume_from_checkpoint"] is True:
        exp["trainer"]["resume_from_checkpoint"] = exp["general"][
            "checkpoint_load"]
    else:
        del exp["trainer"]["resume_from_checkpoint"]

    # if exp["trainer"]["load_from_checkpoint"] is True:
    #     if exp["general"]["load_pretrain"]:
    #         checkpoint = torch.load(exp["general"]["checkpoint_load"])
    #         checkpoint = checkpoint["state_dict"]
    #         # remove any aux classifier stuff
    #         removekeys = [
    #             key for key in checkpoint.keys()
    #             if key.startswith('_model._model.aux_classifier')
    #         ]
    #         for key in removekeys:
    #             del checkpoint[key]

    # del exp["trainer"]["load_from_checkpoint"]

    # - Add distributed plugin.
    if exp["trainer"]["gpus"] > 1:
        if exp["trainer"]["accelerator"] == "ddp" or exp["trainer"][
                "accelerator"] is None:
            ddp_plugin = DDPPlugin(find_unused_parameters=exp["trainer"].get(
                "find_unused_parameters", False))
        exp["trainer"]["plugins"] = [ddp_plugin]

    #exp["trainer"]["max_epochs"] = args.nerf_train_epoch

    trainer_nerf = Trainer(**exp["trainer"],
                           default_root_dir=model_path,
                           logger=logger,
                           callbacks=cb_ls)
    exp["trainer"]["check_val_every_n_epoch"] = 10
    trainer_nerf.fit(model,
                     train_dataloaders=datamodule.train_dataloader_nerf())
    trainer_nerf.test(model, dataloaders=datamodule.test_dataloader_nerf())
    # save checkpoint of the deeplab model


if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    args = parse_args()
    exp_cfg_path = os.path.join(ROOT_DIR, args.exp)
    exp = load_yaml(exp_cfg_path)
    exp['data_module']['root'] = args.root
    exp["general"]["load_pretrain"] = True
    env_cfg_path = os.path.join(ROOT_DIR, "cfg/env",
                                os.environ["ENV_WORKSTATION_NAME"] + ".yml")
    env = load_yaml(env_cfg_path)
    train(exp, env, exp_cfg_path, env_cfg_path, args)
