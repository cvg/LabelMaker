#!/bin/bash


# scannet mappings
python scripts/project_3d.py --label_key label-filt --scene scene0000_00
python scripts/project_3d.py --label_key label-filt --scene scene0164_02
python scripts/project_3d.py --label_key label-filt --scene scene0458_00
python scripts/project_3d.py --label_key label-filt --scene scene0474_01
python scripts/project_3d.py --label_key label-filt --scene scene0518_00


# agile3d groundtruth mappings
python scripts/project_3d.py --label_key label_agile3d --scene scene0000_00
python scripts/project_3d.py --label_key label_agile3d --scene scene0164_02
python scripts/project_3d.py --label_key label_agile3d --scene scene0458_00
python scripts/project_3d.py --label_key label_agile3d --scene scene0474_01
python scripts/project_3d.py --label_key label_agile3d --scene scene0518_00

# labelmaker3d mappings
python scripts/project_3d.py --label_key pred_sdfstudio_2023-07-30_112430 --scene scene0000_00 --subsampling 2
python scripts/project_3d.py --label_key pred_sdfstudio_2023-07-30_104700 --scene scene0164_02 --subsampling 2
python scripts/project_3d.py --label_key pred_sdfstudio_2023-07-30_104953 --scene scene0458_00 --subsampling 2
python scripts/project_3d.py --label_key pred_sdfstudio_2023-07-30_105014 --scene scene0474_01 --subsampling 2
python scripts/project_3d.py --label_key pred_sdfstudio_2023-07-30_104735 --scene scene0518_00 --subsampling 2


