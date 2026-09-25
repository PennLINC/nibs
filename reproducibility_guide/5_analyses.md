# Manuscript analyses

The numbered analysis directories encode the intended order. Once all shared
inputs exist, independent branches can run concurrently.

## Select the metric set

All analysis launchers use the same `ANALYSIS_SET` environment variable:

| Value | Purpose |
| --- | --- |
| `primary` | Main analyses and main-text artifacts; this is the default |
| `full` | Process all metrics and write both primary and supplemental result views |

The analytic replicator normally runs the default `primary` workflow. No
additional flag is required. For an explicit primary-only session, run:

```bash
export ANALYSIS_SET=primary
```

All subsequent `sbatch` submissions inherit that value. To run a launcher with
the supplemental metric set, add `--export=ALL,ANALYSIS_SET=full` to its
`sbatch` command. Output tables identify the selected analysis set, and most
set-specific filenames include `primary` or `full`.

```{important}
`full` includes every primary metric. It also refreshes the primary result view
before writing the expanded supplemental view. Therefore, running `full` after
`primary` in the same output root is safe: primary summaries may be replaced by
equivalent recomputed versions, while supplemental results are added.
```

## GM/WM differentiation

The default job produces the primary results for Figure 3:

```bash
effect_size_job=$(sbatch --parsable \
  --dependency="afterok:${ribbon_job}" \
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

The primary results support Figure 4; the corresponding full results support
Figures S3–S6.

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
tables. Primary ICC results support Figure 5, while full ICC results support
Figures S8–S11.

## Discriminability

```bash
mni_discriminability_job=$(sbatch --parsable \
  analysis/05_discriminability/01_mni_voxelwise/submit.sbatch)

regional_discriminability_job=$(sbatch --parsable \
  --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  analysis/05_discriminability/02_parcel_bundle/submit.sbatch)
```

Primary discriminability results support Table 4; full regional results support
Figure S12.

## Full workflow with supplemental metrics

The primary commands above are sufficient for the main analyses. In a
complete reproduction, add the expanded supplemental results by submitting the
relevant launchers with `ANALYSIS_SET=full`:

```bash
sbatch --dependency="afterok:${ribbon_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/02_gm_wm_differentiation/01_mni_effect_sizes/submit.sbatch

sbatch --dependency="afterok:${ribbon_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/03_correlations/01_mni_voxelwise/submit.sbatch

sbatch --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/03_correlations/02_parcel_bundle/submit.sbatch

sbatch --export=ALL,ANALYSIS_SET=full \
  analysis/04_icc/02_mni_voxelwise/submit.sbatch

sbatch --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/04_icc/01_parcel_bundle/submit.sbatch

sbatch --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/05_discriminability/02_parcel_bundle/submit.sbatch
```

These jobs provide Figure S2, Figures S3–S6, Figures S8–S11, and Figure S12.
Figure S7 uses primary regional ICC results. Full voxelwise discriminability
can also be requested with the same flag, although no supplemental plotting
script requires it.

Because `full` also writes the primary result view, these commands may be used
from the outset instead of first running the primary analysis commands.

## Monitor jobs

```bash
squeue -u "${USER}"
```

Job logs are written below the profile's `logs_dir`. A failed array task should
be diagnosed and rerun before figures are generated; otherwise plotting scripts
may skip missing inputs or stop in strict mode.
