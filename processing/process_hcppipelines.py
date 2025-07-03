"""Run HCPPipelines through QuNex.

python qunex_container \
    --container=/cbica/projects/nibs/apptainer/qunex_suite_1_2_2.sif \
    import_bids \
    --sessionsfolder=/cbica/projects/nibs/hcp_test \
    --inbox=/cbica/projects/nibs/dset \
    --action='copy' \
    --archive='leave' \
    --overwrite=no
"""
...
