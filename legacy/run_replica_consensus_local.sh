#!/bin/bash

for sc in room_0_1 room_0_2 room_1_1 room_1_2 room_2_1 room_2_2 office_0_1 office_0_2 office_1_1 office_1_2 office_2_1 office_2_2 office_3_1 office_3_2 office_4_1 office_4_2; do
    echo $sc
    python scripts/segmentation_consensus.py --replica True --votes 4 /home/weders/scratch/scratch/scannetter/$sc
done