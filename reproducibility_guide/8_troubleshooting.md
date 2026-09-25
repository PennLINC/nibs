# Troubleshooting

## The wrong profile is being loaded

```bash
echo "${MIRROR_CONFIG}"
python configuration/resolve_paths.py
```

`MIRROR_CONFIG` may be a profile name such as `hpc` or an absolute YAML path.
For replication, point it to the edited profile in the current checkout:

```bash
repo_root="$(git rev-parse --show-toplevel)"
export MIRROR_CONFIG="${repo_root}/configuration/profiles/replication.example.yml"
```

## The repository was renamed or moved

Keep `code_dir: auto`. The loader resolves code and configuration assets from
the current checkout, so no profile edit is needed after moving or renaming it.

## A job looks for `/var/spool/configuration/load_profile.sh`

Slurm executes a temporary copy of an SBATCH script under `/var/spool`.
Current launchers therefore recover the checkout with `SLURM_SUBMIT_DIR` and
Git rather than treating that temporary copy as the repository. Submit from
any directory inside the checkout. If submitting while standing elsewhere,
export the checkout explicitly:

```bash
export MIRROR_CODE_ROOT=/cbica/projects/nibs/code_replication
```

Seeing the `/var/spool/configuration` error means the checkout contains an
older launcher; update the branch before resubmitting.

## Slurm output appears beside the submitted script

Without an `--output` option, Slurm writes `slurm-<job-id>.out` in the
submission directory. It chooses this path before the job starts and before
the replication profile can be loaded. The configured `logs_dir` is used for
logs written by the running workflow, not automatically for Slurm's own
stdout/stderr file.

To collect scheduler output there as well:

```bash
mkdir -p /cbica/projects/nibs/logs/replication/slurm
sbatch \
  --output=/cbica/projects/nibs/logs/replication/slurm/%x-%A_%a.out \
  processing/03_registration_and_warping/01_submit_t1w_registration.sbatch
```

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
