#!/bin/bash
unset LD_LIBRARY_PATH

subjects="24053_13187"
token=$(</cbica/projects/nibs/tokens/flywheel.txt)
~/bin/glibc-2.34/lib/ld-linux-x86-64.so.2 ~/bin/linux_amd64/fw login "$token"
cd "/cbica/projects/nibs/sourcedata/imaging" || exit

for subject in $subjects; do
    ~/bin/glibc-2.34/lib/ld-linux-x86-64.so.2 ~/bin/linux_amd64/fw download --yes --zip "fw://bbl/NIBS_857664/${subject}"
done
