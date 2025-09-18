#!/usr/bin/env bash
# Run the BIDS validator at this stage (pre-CuBIDS)
deno run -ERWN jsr:@bids/validator /cbica/projects/nibs/dset
