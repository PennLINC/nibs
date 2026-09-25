# Registration, warping, and quality control

Run commands from the repository root with `MIRROR_CONFIG` exported.

## T1w registration

The first replicated processing step estimates the ACPC↔T1w transforms:

```bash
registration_job=$(sbatch --parsable \
  processing/03_registration_and_warping/01_t1w_registration/submit.sbatch)
echo "T1w registration job: ${registration_job}"
```

Outputs are written to:

```text
<output_derivatives_dir>/t1w_registration/
```

## Atlas and bundle warping

Both branches depend on successful T1w registration and may then run in
parallel:

```bash
dkt_warp_job=$(sbatch --parsable --dependency="afterok:${registration_job}" \
  processing/03_registration_and_warping/02_dkt_atlas_warping/submit.sbatch)

bundle_warp_job=$(sbatch --parsable --dependency="afterok:${registration_job}" \
  processing/03_registration_and_warping/03_bundle_warping/submit.sbatch)

echo "DKT warp job: ${dkt_warp_job}"
echo "Bundle warp job: ${bundle_warp_job}"
```

The DKT segmentation is stored with the registration derivatives; warped
AutoTrack bundles are stored under:

```text
<output_derivatives_dir>/warped_bundles/
```

## Quality-control reports

Coregistration and scalar reports inspect reusable inputs but write reportlets
under the replication output root:

```bash
coreg_qc_job=$(sbatch --parsable \
  processing/04_quality_control/01_coregistration_reports/submit.sbatch)

scalar_qc_job=$(sbatch --parsable \
  processing/04_quality_control/02_scalar_reports/submit.sbatch)
```

After registration, warping, and the analyses are available, generate the
consolidated spatial QC report:

```bash
sbatch processing/04_quality_control/03_analysis_qc/submit.sbatch
```

QC outputs are collected below:

```text
<output_derivatives_dir>/quality_control/
<output_derivatives_dir>/qc_report/
```

```{note}
Successful job completion is not a substitute for visual QC. Review missing
input warnings, registration overlays, scalar reportlets, and the consolidated
report before interpreting replicated statistics.
```

