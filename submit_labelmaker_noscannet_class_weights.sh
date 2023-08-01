#!/bin/bash

source euler_env

sbatch < scripts/labelmaker3d_noscannet/scene0000_00sdf_class_weights.bash
sbatch < scripts/labelmaker3d_noscannet/scene0164_02sdf_class_weights.bash
sbatch < scripts/labelmaker3d_noscannet/scene0458_00sdf_class_weights.bash
sbatch < scripts/labelmaker3d_noscannet/scene0474_01sdf_class_weights.bash
sbatch < scripts/labelmaker3d_noscannet/scene0518_00sdf_class_weights.bash