"""Expand dicom zip files in order to heudiconv."""

import os
import zipfile
from glob import glob

if __name__ == "__main__":
    zip_files = sorted(glob("/cbica/projects/nibs/sourcedata/*_*.zip"))
    for zip_file in zip_files:
        print(f"Processing {os.path.basename(zip_file)}")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(zip_file))
