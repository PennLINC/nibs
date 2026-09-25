# Processing workflow

Directory numbers encode dependency order. Parallel modality branches have
their own folders and SBATCH entry points.

## Reproduction boundary

A manuscript reproduction begins at `03_registration_and_warping`, after all
source scalar maps exist. The following top-level derivative datasets are
reusable, read-only inputs:

- `smriprep`, `qsiprep`, and `qsirecon`
- `pymp2rage`, `ihmt`, `mese`, `megre`, and `qsm`
- `t1wt2w_ratio`, `q_ratio`, and `g_ratio`

Later products are isolated below the selected profile's
`output_derivatives_dir`. Changing that one setting protects the original
manuscript results during replication.

## Run order

1. `00_setup/pull_apptainer_images.sh` downloads containers to the sibling
   `<project_root>/apptainer` directory.
2. `01_smri_dmri/01_submit_smriprep.sbatch` creates anatomical derivatives.
3. `01_smri_dmri/02_submit_qsiprep.sbatch` preprocesses diffusion MRI.
4. `01_smri_dmri/03_submit_qsirecon.sbatch` generates diffusion scalar maps.
5. `02_other_metric_processing/` generates the other reusable source scalars.
   Each modality folder is independently SBATCHable. Within `08_g_ratio`, run
   `01_submit_scaling_factors.sbatch`, `02_aggregate_scaling_factors.sbatch`,
   and `03_submit_g_ratio.sbatch` in that order.
6. `03_registration_and_warping/01_t1w_registration/submit.sbatch` creates the
   ACPC↔T1w transforms in the run output namespace.
7. After step 6 succeeds, the DKT atlas and bundle branches may run in parallel:
   `02_dkt_atlas_warping/submit.sbatch` and
   `03_bundle_warping/submit.sbatch`.
8. `04_quality_control/` contains coregistration, scalar, and consolidated
   analysis-input QC reports. QC is a processing concern and is not duplicated
   under `analysis/`.

Every launcher loads `MIRROR_CONFIG` through
`configuration/load_profile.sh`. Cluster-specific Python executables and
modules can be selected with environment variables such as `PYTHON_BIN` and
`MATLAB_MODULE` without editing tracked scripts.
