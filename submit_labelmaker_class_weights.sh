#!/bin/bash

source euler_env

sbatch < scripts/scene0000_00sdf_class_weights.bash
sbatch < scripts/scene0164_02sdf_class_weights.bash
sbatch < scripts/scene0458_00sdf_class_weights.bash
sbatch < scripts/scene0474_01sdf_class_weights.bash
sbatch < scripts/scene0518_00sdf_class_weights.bash