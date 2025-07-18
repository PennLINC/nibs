#!/bin/bash
unset LD_LIBRARY_PATH

subjects=(
    "techdev_ihmt"
    "techdev_dt_myelin"
    "techdev_human_myelin_02"
    "153-327"
    "60522_13307"
    "60522_13309"
    "60518_13313"
    "60518_13315"
    "60526_13316"
    "60526_13317"
    "60519_13320"
    "60519_13321"
    "60516_13324"
    "60516_13325"
)

token=$(</cbica/projects/nibs/tokens/flywheel.txt)
#~/bin/glibc-2.34/lib/ld-linux-x86-64.so.2 ~/bin/linux_amd64/fw login "$token"
~/bin/linux_amd64/fw login "$token"
cd "/cbica/projects/nibs/sourcedata" || exit

# Initialize download status file if it doesn't exist
download_status_file="download_status.txt"
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
