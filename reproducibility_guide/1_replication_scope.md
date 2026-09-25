# Replication scope and data boundary

## Reusable inputs

Replication starts after preprocessing and source-scalar generation. The
following derivative datasets are treated as inputs:

- `smriprep`
- `qsiprep`
- `qsirecon`
- `pymp2rage`
- `ihmt`
- `mese`
- `megre`
- `qsm`
- `t1wt2w_ratio`
- `q_ratio`
- `g_ratio`

These directories normally live immediately below `source_derivatives_dir`.
The exact QSIRecon subdirectories are declared in the selected profile.

## Steps not repeated

The replication described here does not run:

- `curation/`, which documents DICOM-to-BIDS curation;
- `processing/01_smri_dmri/`, which runs sMRIPrep, QSIPrep, and QSIRecon; or
- `processing/02_other_metric_processing/`, which generates source scalar
  derivatives.

Those workflows remain in the repository for provenance and for users who need
to regenerate the released source data.

## First replicated step

Begin with:

```text
processing/03_registration_and_warping/
```

All later computational outputs are written beneath the configured
`output_derivatives_dir`. Generated figures and tables default to its
`figures/` subdirectory. Logs and temporary files use the separately
configured `logs_dir` and `work_dir`.

## Expected project layout

One possible cluster layout is:

```text
<project_root>/
├── apptainer/
├── code_replication/             # this repository; any name/location is valid
├── derivatives/
│   ├── smriprep/                 # reusable inputs
│   ├── qsiprep/
│   ├── qsirecon/
│   ├── pymp2rage/
│   ├── ihmt/
│   ├── mese/
│   ├── megre/
│   ├── qsm/
│   ├── t1wt2w_ratio/
│   ├── q_ratio/
│   ├── g_ratio/
│   └── replication/              # new outputs
├── dset/                         # raw BIDS dataset
├── logs/
└── work/
```

The source and output trees may live elsewhere, including on different
filesystems, as long as the profile contains their absolute paths.
