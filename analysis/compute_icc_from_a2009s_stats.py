#!/usr/bin/env python3
"""Compute parcel-wise ICC from a2009s per-run scalar summary CSVs."""

from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pingouin as pg

    HAVE_PINGOUIN = True
except Exception:
    HAVE_PINGOUIN = False


FILE_RE = re.compile(r"sub-(?P<sub>[^_]+)_(?P<ses>ses-[^_]+)_(?P<run>run-[^_]+)_")


def compute_icc2_fallback(
    values: np.ndarray, subjects: np.ndarray, sessions: np.ndarray
) -> float:
    """Compute ICC(2,1) using a two-way random effects fallback."""
    subs_unique, sub_idx = np.unique(subjects, return_inverse=True)
    sessions_unique, sess_idx = np.unique(sessions, return_inverse=True)
    n_sub, n_ses = len(subs_unique), len(sessions_unique)
    if n_sub < 2 or n_ses < 2:
        return np.nan

    matrix = np.full((n_sub, n_ses), np.nan, dtype=float)
    for val, i_sub, i_ses in zip(values, sub_idx, sess_idx):
        matrix[i_sub, i_ses] = val

    # Complete-case subjects only.
    matrix = matrix[~np.any(np.isnan(matrix), axis=1)]
    if matrix.shape[0] < 2:
        return np.nan

    n_sub = matrix.shape[0]
    grand_mean = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)

    ssr = n_ses * np.sum((row_means - grand_mean) ** 2)
    ssc = n_sub * np.sum((col_means - grand_mean) ** 2)
    sse = np.sum((matrix - grand_mean) ** 2) - ssr - ssc

    msr = ssr / (n_sub - 1)
    msc = ssc / (n_ses - 1)
    mse = sse / ((n_sub - 1) * (n_ses - 1))

    denom = msr + (n_ses - 1) * mse + n_ses * (msc - mse) / n_sub
    if denom == 0:
        return np.nan
    return float((msr - mse) / denom)


def parse_filename(path: Path) -> tuple[str, str, str]:
    match = FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse subject/session/run from: {path}")
    return match.group("sub"), match.group("ses"), match.group("run")


def collect_rows(input_glob: str) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for file_str in sorted(glob(input_glob)):
        path = Path(file_str)
        subject, session, run = parse_filename(path)
        df = pd.read_csv(path)
        df["subject"] = subject
        df["session"] = session
        df["run"] = run
        records.append(df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def build_value_table(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    base_cols = [
        "subject",
        "session",
        "run",
        "parcel_intensity",
        "parcel_name",
        "parcel_hemi",
        "parcel_count_t1w",
        "parcel_count_acpc",
    ]
    metric_cols = [col for col in df.columns if col.endswith(f"_{stat}")]
    if not metric_cols:
        raise RuntimeError(f"No columns found ending with _{stat}")

    value_df = df[base_cols + metric_cols].melt(
        id_vars=base_cols,
        value_vars=metric_cols,
        var_name="metric_stat",
        value_name="value",
    )
    value_df["metric"] = value_df["metric_stat"].str[: -(len(stat) + 1)]
    value_df["parcel"] = (
        value_df["parcel_hemi"].astype(str) + "_" + value_df["parcel_name"].astype(str)
    )
    return value_df.drop(columns=["metric_stat"])


def compute_icc_table(value_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = value_df.groupby(["metric", "parcel", "parcel_intensity", "parcel_hemi"], sort=True)
    for (metric, parcel, parcel_intensity, parcel_hemi), dfg in grouped:
        dfg = dfg[np.isfinite(dfg["value"].to_numpy())].copy()
        if dfg.empty:
            continue

        session_counts = dfg.groupby("subject")["session"].nunique()
        valid_subjects = session_counts[session_counts >= 2].index
        dfg = dfg[dfg["subject"].isin(valid_subjects)]
        if dfg["subject"].nunique() < 2 or dfg["session"].nunique() < 2:
            continue

        subjects = dfg["subject"].to_numpy()
        sessions = dfg["session"].to_numpy()
        values = dfg["value"].to_numpy(dtype=float)

        icc_val = np.nan
        ci95 = None
        f_val = np.nan
        df1 = np.nan
        df2 = np.nan
        pval = np.nan

        if HAVE_PINGOUIN:
            try:
                tab = pd.DataFrame(
                    {
                        "targets": subjects,
                        "raters": sessions,
                        "scores": values,
                    }
                )
                icc_tab = pg.intraclass_corr(
                    data=tab,
                    targets="targets",
                    raters="raters",
                    ratings="scores",
                )
                icc_row = icc_tab.query("Type == 'ICC2'").iloc[0]
                icc_val = float(icc_row["ICC"])
                ci95 = str(icc_row.get("CI95%", ""))
                f_val = float(icc_row.get("F", np.nan))
                df1 = float(icc_row.get("df1", np.nan))
                df2 = float(icc_row.get("df2", np.nan))
                pval = float(icc_row.get("pval", np.nan))
            except Exception:
                icc_val = compute_icc2_fallback(values, subjects, sessions)
        else:
            icc_val = compute_icc2_fallback(values, subjects, sessions)

        rows.append(
            {
                "metric": metric,
                "parcel": parcel,
                "parcel_intensity": int(parcel_intensity),
                "parcel_hemi": parcel_hemi,
                "ICC2_1": icc_val,
                "CI95": ci95,
                "F": f_val,
                "df1": df1,
                "df2": df2,
                "pval": pval,
                "n_subjects": int(dfg["subject"].nunique()),
                "n_sessions": int(dfg["session"].nunique()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["metric", "parcel"]).reset_index(drop=True)


def plot_heatmap(df_icc: pd.DataFrame, out_file: Path) -> None:
    pivot = df_icc.pivot(index="metric", columns="parcel", values="ICC2_1")
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    col_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot.loc[row_order, col_order]

    fig_width = max(12, 0.24 * len(pivot.columns))
    fig_height = max(6, 0.3 * len(pivot.index))
    plt.figure(figsize=(fig_width, fig_height))
    im = plt.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, label="ICC(2,1)")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.title("Parcel-wise ICC Heatmap (rows/columns ordered by mean ICC)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default="/cbica/projects/nibs/derivatives/parcel_bundle_stats/sub-*/sub-*_ses-*_run-*_desc-a2009s_scalarstats.csv",
        help="Glob pattern to per-run parcel summary CSVs.",
    )
    parser.add_argument(
        "--outdir",
        default="/cbica/projects/nibs/derivatives/ICC",
        help="Output directory for ICC CSV + heatmap.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = collect_rows(args.input_glob)
    if all_rows.empty:
        raise RuntimeError(f"No input files found with glob: {args.input_glob}")

    for stat in ("mean", "median"):
        value_df = build_value_table(all_rows, stat=stat)
        icc_df = compute_icc_table(value_df)
        if icc_df.empty:
            raise RuntimeError(
                f"No valid ICC results for stat={stat}. Check run coverage and missing data."
            )

        icc_csv = outdir / f"icc_summary_a2009s_{stat}.csv"
        icc_df.to_csv(icc_csv, index=False)

        heatmap_png = outdir / f"icc_heatmap_a2009s_{stat}.png"
        plot_heatmap(icc_df, heatmap_png)

        print(f"Wrote: {icc_csv}", flush=True)
        print(f"Wrote: {heatmap_png}", flush=True)


if __name__ == "__main__":
    main()
