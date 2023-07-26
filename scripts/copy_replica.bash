#!/bin/bash

echo $1
mkdir $TMPDIR/$1
for SUBDIR in rgb depth intrinsic pose traj_w_c.txt intrinsic_color.txt
do
    echo $SUBDIR
    cp -r /cluster/project/cvg/blumh/replica/$1/$SUBDIR $TMPDIR/$1/
done

