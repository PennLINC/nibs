# Manuscript analyses

The numbered analysis directories encode the intended order. Once all shared
inputs exist, independent branches can run concurrently.

## GM/WM differentiation

Figure 3 uses the primary metric set, while Figure S2 uses the full metric
set. Submit both variants; their filenames are distinct within the same output
directory:

```bash
primary_effect_size_job=$(sbatch --parsable \
  --dependency="afterok:${ribbon_job}" \
  analysis/02_gm_wm_differentiation/01_mni_effect_sizes/submit.sbatch)

full_effect_size_job=$(sbatch --parsable \
  --dependency="afterok:${ribbon_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/02_gm_wm_differentiation/01_mni_effect_sizes/submit.sbatch)
```

## Correlations

Voxelwise correlations use MNI-space scalar maps and ribbon masks:

```bash
mni_correlation_job=$(sbatch --parsable --dependency="afterok:${ribbon_job}" \
  analysis/03_correlations/01_mni_voxelwise/submit.sbatch)
```

Parcel/bundle correlations use both regional summary branches:

```bash
regional_correlation_job=$(sbatch --parsable \
  --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  analysis/03_correlations/02_parcel_bundle/submit.sbatch)
```

These results support Figure 4 and Figures S3–S6.

## Intraclass correlation

Regional ICC uses the DKT and bundle tables:

```bash
regional_icc_job=$(sbatch --parsable \
  --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  analysis/04_icc/01_parcel_bundle/submit.sbatch)
```

Voxelwise ICC may run independently once the configured source derivatives are
available:

```bash
mni_icc_job=$(sbatch --parsable \
  analysis/04_icc/02_mni_voxelwise/submit.sbatch)
```

The regional ICC output includes `within_subject_sd` and
`between_subject_sd`. Figure S7 calculates their ratio directly from these
tables. ICC results also support Figure 5 and Figures S8–S11.

## Discriminability

```bash
mni_discriminability_job=$(sbatch --parsable \
  analysis/05_discriminability/01_mni_voxelwise/submit.sbatch)

regional_discriminability_job=$(sbatch --parsable \
  --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  analysis/05_discriminability/02_parcel_bundle/submit.sbatch)
```

These results support Table 4 and Figure S12.

## Monitor jobs

```bash
squeue -u "${USER}"
```

Job logs are written below the profile's `logs_dir`. A failed array task should
be diagnosed and rerun before figures are generated; otherwise plotting scripts
may skip missing inputs or stop in strict mode.
