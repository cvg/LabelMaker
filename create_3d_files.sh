#!/bin/bash


# scannet mappings
# python scripts/project_3d.py --label_key label-filt --scene scene0000_00
# python scripts/project_3d.py --label_key label-filt --scene scene0164_02
# python scripts/project_3d.py --label_key label-filt --scene scene0458_00
# python scripts/project_3d.py --label_key label-filt --scene scene0474_01 --max_label 1500
# python scripts/project_3d.py --label_key label-filt --scene scene0518_00


# agile3d groundtruth mappings
# python scripts/project_3d.py --label_key label_agile3d --scene scene0000_00
# python scripts/project_3d.py --label_key label_agile3d --scene scene0164_02
# python scripts/project_3d.py --label_key label_agile3d --scene scene0458_00
# python scripts/project_3d.py --label_key label_agile3d --scene scene0474_01 --max_label 1500
# python scripts/project_3d.py --label_key label_agile3d --scene scene0518_00

# labelmaker3d mappings
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-07-30_112430 --scene scene0000_00 --subsampling 2
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-07-30_104700 --scene scene0164_02 --subsampling 2
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-07-30_104953 --scene scene0458_00 --subsampling 2
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-07-30_105014 --scene scene0474_01 --subsampling 2 --max_label 1500
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-07-30_104735 --scene scene0518_00 --subsampling 2


# # semantic nerf mappings
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-08-02_003056 --scene scene0000_00 --subsampling 2
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-08-02_000339 --scene scene0164_02 --subsampling 2
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-08-02_000600 --scene scene0458_00 --subsampling 2
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-08-02_000611 --scene scene0474_01 --subsampling 2 --max_label 1500
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-08-02_000401 --scene scene0518_00 --subsampling 2

# noscannet mappings
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-08-02_015512 --scene scene0000_00 --subsampling 2
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-08-02_002042 --scene scene0164_02 --subsampling 2
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-08-02_002220 --scene scene0458_00 --subsampling 2
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-08-02_083537 --scene scene0474_01 --subsampling 2 --max_label 1500
# python scripts/project_3d.py --label_key pred_sdfstudio_2023-08-02_083237 --scene scene0518_00 --subsampling 2

# # pred_ovseg mappings
# python scripts/project_3d.py --label_key pred_ovseg_wn_nodef --scene scene0000_00 --subsampling 1
# python scripts/project_3d.py --label_key pred_ovseg_wn_nodef --scene scene0164_02 --subsampling 1
# python scripts/project_3d.py --label_key pred_ovseg_wn_nodef --scene scene0458_00 --subsampling 1
# python scripts/project_3d.py --label_key pred_ovseg_wn_nodef --scene scene0474_01 --subsampling 1 --max_label 1500
# python scripts/project_3d.py --label_key pred_ovseg_wn_nodef --scene scene0518_00 --subsampling 1

# # pred_intern mappings
# python scripts/project_3d.py --label_key pred_internimage --scene scene0000_00 --subsampling 1
# python scripts/project_3d.py --label_key pred_internimage --scene scene0164_02 --subsampling 1
# python scripts/project_3d.py --label_key pred_internimage --scene scene0458_00 --subsampling 1
# python scripts/project_3d.py --label_key pred_internimage --scene scene0474_01 --subsampling 1 --max_label 1500
# python scripts/project_3d.py --label_key pred_internimage --scene scene0518_00 --subsampling 1

# # pred_cmx mappings
# python scripts/project_3d.py --label_key pred_cmx --scene scene0000_00 --subsampling 1
# python scripts/project_3d.py --label_key pred_cmx --scene scene0164_02 --subsampling 1
# python scripts/project_3d.py --label_key pred_cmx --scene scene0458_00 --subsampling 1
# python scripts/project_3d.py --label_key pred_cmx --scene scene0474_01 --subsampling 1 --max_label 1500
# python scripts/project_3d.py --label_key pred_cmx --scene scene0518_00 --subsampling 1

# # pred_mask3d mappings
# python scripts/project_3d.py --label_key pred_mask3d_rendered --scene scene0000_00 --subsampling 1
# python scripts/project_3d.py --label_key pred_mask3d_rendered --scene scene0164_02 --subsampling 1
# python scripts/project_3d.py --label_key pred_mask3d_rendered --scene scene0458_00 --subsampling 1
# python scripts/project_3d.py --label_key pred_mask3d_rendered --scene scene0474_01 --subsampling 1 --max_label 1500
# python scripts/project_3d.py --label_key pred_mask3d_rendered --scene scene0518_00 --subsampling 1

# # pred_consensus mappings
# python scripts/project_3d.py --label_key pred_consensus_5_scannet --scene scene0000_00 --subsampling 1
# python scripts/project_3d.py --label_key pred_consensus_5_scannet --scene scene0164_02 --subsampling 1
# python scripts/project_3d.py --label_key pred_consensus_5_scannet --scene scene0458_00 --subsampling 1
# python scripts/project_3d.py --label_key pred_consensus_5_scannet --scene scene0474_01 --subsampling 1 --max_label 1500
# python scripts/project_3d.py --label_key pred_consensus_5_scannet --scene scene0518_00 --subsampling 1

python scripts/project_3d.py --dataset arkitscenes  --scene 42445991 --subsampling 1 --label_key pred_sdfstudio_2023-08-02_230529
python scripts/project_3d.py --dataset arkitscenes  --scene 42446517 --subsampling 1 --label_key pred_sdfstudio_2023-08-02_230530
python scripts/project_3d.py --dataset arkitscenes  --scene 42446527 --subsampling 1 --label_key pred_sdfstudio_2023-08-02_230521
python scripts/project_3d.py --dataset arkitscenes  --scene 42897688 --subsampling 1 --label_key pred_sdfstudio_2023-08-02_230607
