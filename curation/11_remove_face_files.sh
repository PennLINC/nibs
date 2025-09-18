#!/bin/bash
module load afni/2022_05_03

face_files=$(find /cbica/projects/nibs/dset/sub-*/ses-*/anat/*.face.nii.gz)
for face_file in $face_files
do
    echo "$face_file"
    rm -f $face_file
done
