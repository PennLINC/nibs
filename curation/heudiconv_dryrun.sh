heudiconv \
    -d "/Users/taylor/Downloads/flywheel/bbl/NIBS_857664/techdev_human_myelin_{subject}/*/*/*/*.dcm" \
    -o /Users/taylor/Downloads/flywheel/bbl/dset \
    -f heuristic.py \
    -s "02" \
    --ses "01" \
    -g all \
    -c dcm2niix \
    --bids
