source environments:

first 
source euler_env

if preprocessing segmentations:
source segenv/bin/activate


# Data Locations

all stored in

on euler:
/cluster/project/cvg/blumh/{replica/scannet}

on scratch:
scratch/scanneter/<scene>

required additional inputs
- omnidata_depth
- omnidata_normal
- hha

predictions
- pred_cmx
- pred_cmx_flip
- pred_ovseg_wn_nodef
- pred_ovseg_wn_nodef_flip
- pred_internimage
- pred_internimage_flip
- pred_sam
- pred_consensus
- pred_consensus_noscannet


# Data Preprocessing
! careful: activate (segenv), not (env)
runs omnidata depth + normal, SAM, HHA depth preprocessing, InternImage, OVSEG, CMX, Consensus voting

bash scripts/eulerpreprocessing.bash

# SDFStudio
! careful: active (env), not (segenv)

to put it into SDFStudio, we have another preprocessing script that we always run as part of the sdfstudio job
scritps/sdfstudio_scannnet_preprocessing.py

all you actually need to run is something like
sbatch < scripts/scene0458_00sdf.bash

if you want to check how the job performed:
wandb sync ./outputs/<timestamp of experiment>/wandb/offline_run_<id>

the mesh will be in ./outputs/<timestamp of experiment>/{mesh,mesh_scaled}.ply
the renderings are in /cluster/project/cvg/blumh/{replica/scannet}/pred_sdfstudio_<timestamp>/

# Evaluation
code is in segmentation_tools/label_mappings.py and scripts/segmentation_eval.py (for 2D eval) and in scripts/mesh_eval.py (for 3D eval)
