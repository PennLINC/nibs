#!/usr/bin/env bash
# Add .heudiconv/ and sourcedata/ to the .gitignore file
echo ".heudiconv/" >> /cbica/projects/nibs/dset/.gitignore
echo "sourcedata/" >> /cbica/projects/nibs/dset/.gitignore

# Create the datalad dataset after anonymizing anatomical images and metadata
datalad create --force -c text2git /cbica/projects/nibs/dset

# Save the datalad dataset
datalad save -d /cbica/projects/nibs/dset -m "Initial commit"
