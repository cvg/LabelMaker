#!/bin/bash

echo $1
mkdir $TMPDIR/$1
for SUBDIR in color depth intrinsic pose 
do
    echo $SUBDIR
    cp -r /cluster/project/cvg/blumh/arkitscenes_new/$1/$SUBDIR $TMPDIR/$1/
    ls $TMPDIR/$1/$SUBDIR | head
done

