#!/bin/bash

source euler_env

sbatch < scripts/semantic_nerf/scene0000_00sdf_class_weights.bash
sbatch < scripts/semantic_nerf/scene0164_02sdf_class_weights.bash
sbatch < scripts/semantic_nerf/scene0458_00sdf_class_weights.bash
sbatch < scripts/semantic_nerf/scene0474_01sdf_class_weights.bash
sbatch < scripts/semantic_nerf/scene0518_00sdf_class_weights.bash