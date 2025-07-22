#!/bin/bash
unset LD_LIBRARY_PATH

subjects=(
    "153-327"
    "24037_13298"
    "24037_13302"
    "60501_13333"
    "60501_13334"
    "60505_13327"
    "60505_13329"
    "60514_13330"
    "60514_13331"
    "60516_13324"
    "60516_13325"
    "60518_13313"
    "60518_13315"
    "60519_13320"
    "60519_13321"
    "60520_13322"
    "60520_13323"
    "60522_13307"
    "60522_13309"
    "60526_13316"
    "60526_13317"
    "techdev_dt_myelin"
    "techdev_human_myelin_02"
    "techdev_ihmt"
)

token=$(</cbica/projects/nibs/tokens/flywheel.txt)
#~/bin/glibc-2.34/lib/ld-linux-x86-64.so.2 ~/bin/linux_amd64/fw login "$token"
~/bin/linux_amd64/fw login "$token"
cd "/cbica/projects/nibs/sourcedata" || exit

# Initialize download status file if it doesn't exist
download_status_file="/cbica/projects/nibs/code/curation/status_download.txt"
touch "$download_status_file"

# Read already downloaded subjects into an array
if [[ -f "$download_status_file" ]]; then
    mapfile -t downloaded_subjects < "$download_status_file"
else
    downloaded_subjects=()
fi

for subject in "${subjects[@]}"; do
    # Check if subject is already downloaded
    if [[ " ${downloaded_subjects[*]} " =~ " ${subject} " ]]; then
        echo "Subject ${subject} already downloaded, skipping..."
        continue
    fi

    echo "Downloading subject ${subject}..."
    #~/bin/glibc-2.34/lib/ld-linux-x86-64.so.2 ~/bin/linux_amd64/fw download --yes --zip "fw://bbl/NIBS_857664/${subject}"
    if ~/bin/linux_amd64/fw download --yes --zip "fw://bbl/NIBS_857664/${subject}"; then
        echo "Successfully downloaded ${subject}, adding to status file..."
        echo "${subject}\n" >> "$download_status_file"
    else
        echo "Failed to download ${subject}"
    fi
done
