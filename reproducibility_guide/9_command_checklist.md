# End-to-end command checklist

This condensed checklist assumes the reusable source derivatives already exist.
Run it from the repository root after reviewing the preceding chapters.

## 1. Select the environment and profile

```bash
micromamba activate processing
export MIRROR_CONFIG=/absolute/path/to/mirror_replication.yml
python configuration/resolve_paths.py
```

## 2. Registration and warping

```bash
registration_job=$(sbatch --parsable \
  processing/03_registration_and_warping/01_t1w_registration/submit.sbatch)

dkt_warp_job=$(sbatch --parsable --dependency="afterok:${registration_job}" \
  processing/03_registration_and_warping/02_dkt_atlas_warping/submit.sbatch)

bundle_warp_job=$(sbatch --parsable --dependency="afterok:${registration_job}" \
  processing/03_registration_and_warping/03_bundle_warping/submit.sbatch)
```

## 3. Shared analysis inputs

```bash
ribbon_job=$(sbatch --parsable \
  analysis/00_prepare_inputs/01_mni_ribbon_masks/submit.sbatch)

dkt_stats_job=$(sbatch --parsable --dependency="afterok:${dkt_warp_job}" \
  analysis/00_prepare_inputs/02_dkt_parcel_stats/submit.sbatch)

bundle_stats_job=$(sbatch --parsable --dependency="afterok:${bundle_warp_job}" \
  analysis/00_prepare_inputs/03_bundle_myelin_stats/submit.sbatch)

python analysis/01_missingness/01_build_missingness_list.py
```

## 4. Analyses

```bash
sbatch --dependency="afterok:${ribbon_job}" \
  analysis/02_gm_wm_differentiation/01_mni_effect_sizes/submit.sbatch

sbatch --dependency="afterok:${ribbon_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/02_gm_wm_differentiation/01_mni_effect_sizes/submit.sbatch

sbatch --dependency="afterok:${ribbon_job}" \
  analysis/03_correlations/01_mni_voxelwise/submit.sbatch

sbatch --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  analysis/03_correlations/02_parcel_bundle/submit.sbatch

sbatch --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  analysis/04_icc/01_parcel_bundle/submit.sbatch

sbatch analysis/04_icc/02_mni_voxelwise/submit.sbatch

sbatch analysis/05_discriminability/01_mni_voxelwise/submit.sbatch

sbatch --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  analysis/05_discriminability/02_parcel_bundle/submit.sbatch
```

Wait for every required job and array task to finish successfully.

## 5. Quality control

```bash
sbatch processing/04_quality_control/01_coregistration_reports/submit.sbatch
sbatch processing/04_quality_control/02_scalar_reports/submit.sbatch
sbatch processing/04_quality_control/03_analysis_qc/submit.sbatch
```

Review the reports before accepting the results.

## 6. Figures and tables

```bash
python figures/figure_01_primary_maps/plot_figure_1_primary_maps.py
python figures/figure_02_missingness/plot_figure_2_missingness.py
python figures/figure_03_gm_wm_effect_sizes/plot_figure_3_gm_wm_effect_sizes.py
python figures/figure_04_correlations/plot_figure_4_correlations.py
python figures/figure_05_icc/plot_figure_5_icc.py

python figures/figure_s01_workflow/plot_figure_s1_workflow.py
python figures/figure_s02_gm_wm_effect_sizes/plot_figure_s2_gm_wm_effect_sizes.py
python figures/figure_s03-s06_correlation_matrices/plot_figure_s3_s6_correlation_matrices.py
python figures/figure_s07_variability_ratio/plot_figure_s7_variability_ratio.py
python figures/figure_s08-s11_icc/plot_figure_s8_s11_icc.py
python figures/figure_s12_discriminability/plot_figure_s12_discriminability.py

python figures/table_04_discriminability/make_table_4_discriminability.py
python figures/table_s03_metrics/make_table_s3_metrics.py
```

## 7. Record provenance

```bash
git rev-parse HEAD
python configuration/resolve_paths.py
```

Archive the commit hash, selected profile, successful job IDs, QC reports, and
environment record with the replication output.
