# Clone, environment, and configuration

## Clone the repository

```bash
git clone git@github.com:PennLINC/nibs.git /path/to/your/checkout
cd /path/to/your/checkout
```

The checkout does not have to be named `MIRROR` or placed directly below the
project root.

## Create the processing environment

The processing environment also supplies the packages used by the analyses and
figure scripts:

```bash
micromamba env create \
  -f environment_processing.yml \
  --channel-priority=flexible
micromamba activate processing
```

The curation environment is unnecessary unless the raw BIDS dataset is being
rebuilt.

## Create a replication profile

Copy the example outside the checkout so local machine paths do not become
version-controlled changes:

```bash
mkdir -p /path/to/replication/config
cp configuration/profiles/replication.example.yml \
  /path/to/replication/config/mirror_replication.yml
```

Edit the copy. JSON syntax is valid YAML and can be loaded even when PyYAML is
not installed. A cluster profile may look like:

```json
{
  "project_root": "/cbica/projects/nibs",
  "bids_dir": "dset",
  "code_dir": "auto",
  "work_dir": "/cbica/comp_space/USERNAME/mirror_data_descriptor_replication",
  "source_derivatives_dir": "derivatives",
  "run_name": "mirror_data_descriptor_replication",
  "output_derivatives_dir": "derivatives/mirror_data_descriptor_replication",
  "logs_dir": "logs/mirror_data_descriptor_replication",
  "derivatives": {
    "smriprep": "smriprep",
    "qsiprep": "qsiprep",
    "qsirecon_dipydki": "qsirecon/derivatives/qsirecon-DIPYDKI",
    "qsirecon_noddi": "qsirecon/derivatives/qsirecon-NODDI",
    "qsirecon_dsistudio": "qsirecon/derivatives/qsirecon-DSIStudio",
    "pymp2rage": "pymp2rage",
    "ihmt": "ihmt",
    "mese": "mese",
    "megre": "megre",
    "qsm": "qsm",
    "t1wt2w_ratio": "t1wt2w_ratio",
    "q_ratio": "q_ratio",
    "g_ratio": "g_ratio"
  }
}
```

Retain the `apptainer`, `freesurfer`, and `software` sections from the example
when they are needed by the target cluster.

```{warning}
Use a new, preferably nonexistent, `output_derivatives_dir` for the replication.
Rerunning within that same directory may replace outputs from that replication.
It will not affect the manuscript run when the two profiles use different paths.
```

## Select and inspect the profile

```bash
export MIRROR_CONFIG=/path/to/replication/config/mirror_replication.yml
python configuration/resolve_paths.py
```

Confirm that:

1. `MIRROR_CODE_ROOT` is the checkout you just cloned;
2. `MIRROR_SOURCE_DERIVATIVES` points to the reusable inputs; and
3. `MIRROR_OUTPUT_DERIVATIVES`, `MIRROR_LOGS_DIR`, and `MIRROR_WORK_DIR`
   point to replication-specific locations.

All SBATCH launchers load this same profile through
`configuration/load_profile.sh`.

## Optional repository checks

```bash
python -m compileall -q analysis configuration curation figures processing utils
find analysis configuration curation processing figures -type f \
  \( -name '*.sh' -o -name '*.sbatch' \) -print0 \
  | xargs -0 -n1 bash -n
```
