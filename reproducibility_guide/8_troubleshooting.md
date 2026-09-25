# Troubleshooting

## The wrong profile is being loaded

```bash
echo "${MIRROR_CONFIG}"
python configuration/resolve_paths.py
```

`MIRROR_CONFIG` may be a profile name such as `hpc` or an absolute YAML path.
For replication, an absolute path to a copied profile is least ambiguous.

## The repository was renamed or moved

Keep `code_dir: auto`. The loader resolves code and configuration assets from
the current checkout, so no profile edit is needed after moving or renaming it.

## A Slurm array skipped a participant

Array index 1 corresponds to the first data row after the header in
`participants.tsv`. Verify the selected `bids_dir`, array range, and participant
table before resubmitting a missing index.

## A downstream job started too early

Use `--dependency=afterok:<job-id>`. In particular:

- atlas and bundle warping require T1w registration;
- DKT statistics require atlas warping;
- bundle statistics require bundle warping; and
- regional correlation, ICC, and discriminability require both regional
  summary branches.

## A figure reports a missing input

Run the corresponding analysis first and confirm that it used the same profile.
The artifact-to-analysis table in the figures chapter identifies the required
producer. Avoid copying results between run roots unless their provenance is
recorded.

## Figure 2 is using the checked table

Figure 2 prefers:

```text
<output_derivatives_dir>/missingness/missingness_list.tsv
```

and falls back to `data/qc/missingness_list.tsv`. Run
`analysis/01_build_missingness_list.py` to reproduce the table
from the configured BIDS dataset.

## Plotting fonts differ

The scripts request Arial. If it is unavailable, Matplotlib may substitute a
font and change text placement. Install Arial or document the substitution;
compare numeric tables rather than raster pixels.

## Python imports or command-line tools are missing

Activate the `processing` environment. Registration and warping additionally
require the external neuroimaging commands invoked by their launchers, such as
ANTs, FreeSurfer tools, and tractogram conversion utilities.

## Temporary files consume too much project storage

Set `work_dir` to cluster scratch in the replication profile. It should be
unique to the run and may be cleaned after successful completion and review.
