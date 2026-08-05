#!/usr/bin/env python3
"""Create simplified ICC matrix outputs for gray and white matter.

Rules:
1) Filter to selected metric names.
2) Drop columns (parcels/bundles) containing any NaN after filtering.
3) Order rows and columns by row-/column-wise average ICC.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# User-selected scalars mapped to this repo's canonical metric names.
SELECTED_METRICS = {
    "FA",
    "MD",
    "RD",
    "ICVF",
    "MKT",
    "RK",
    "RTOP",
    "RTAP",
    "NG",
    "GFA",
    "QSM-X-R2p-E5-Dia",
    "QSM-X-R2p-E5-Para",
    "QSM-X-R2p-E5-X",
    "QSM-SEPIA-E5",
    "MPRAGE-MyelinW",
    "SPACE-MyelinW",
    "ihMTR",
    "ihMTsat-B1c",
    "R1-B1c",
    "R1",
}


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


ALIAS_TO_CANONICAL = {
    # DWI selected
    "dkimd": "MD",
    "md": "MD",
    "tortoiseinnershellmd": "MD",
    "tortoiseinnershellrd": "RD",
    "rd": "RD",
    "dkitensorfa": "FA",
    "tortoiseinnershellfa": "FA",
    "fa": "FA",
    "dkimkt": "MKT",
    "mkt": "MKT",
    "dkirk": "RK",
    "rk": "RK",
    "noddiicvf": "ICVF",
    "icvf": "ICVF",
    "tortoisemapmrirtop": "RTOP",
    "mapmrirtop": "RTOP",
    "rtop": "RTOP",
    "mapmrirtap": "RTAP",
    "rtap": "RTAP",
    "mapmring": "NG",
    "ng": "NG",
    "dsistudiogqigfa": "GFA",
    "gfa": "GFA",
    # myelin selected
    "ihmtr": "ihMTR",
    "ihmtsatb1c": "ihMTsat-B1c",
    "r1": "R1",
    "r1b1c": "R1-B1c",
    "mpragemyelinw": "MPRAGE-MyelinW",
    "spacemyelinw": "SPACE-MyelinW",
    "qsmsepiae5": "QSM-SEPIA-E5",
    "qsmxr2pe5x": "QSM-X-R2p-E5-X",
    "qsmxr2pe5para": "QSM-X-R2p-E5-Para",
    "qsmxr2pe5dia": "QSM-X-R2p-E5-Dia",
    # apostrophe variants
    "qsmxr2e5x": "QSM-X-R2p-E5-X",
    "qsmxr2e5para": "QSM-X-R2p-E5-Para",
    "qsmxr2e5dia": "QSM-X-R2p-E5-Dia",
}


def _canonicalize_metric_name(metric: str) -> str:
    # If source included prefixes, keep only right-most metric token.
    metric = str(metric).strip()
    if "__" in metric:
        metric = metric.split("__")[-1]
    key = _norm_token(metric)
    return ALIAS_TO_CANONICAL.get(key, metric)


def _filter_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only selected canonical metrics."""
    out = df.copy()
    out["metric"] = out["metric"].astype(str).map(_canonicalize_metric_name)
    return out[out["metric"].isin(SELECTED_METRICS)].copy()


def _prepare_icc_table(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to selected metrics; leave ICC values otherwise unchanged."""
    out = df.copy()
    out = _filter_metrics(out)
    return out.sort_values(["metric"]).reset_index(drop=True)


def _retain_common_metrics(
    gm_df: pd.DataFrame, wm_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_metrics = SELECTED_METRICS.intersection(gm_df["metric"]).intersection(
        wm_df["metric"]
    )
    if not common_metrics:
        raise RuntimeError("No selected metrics are shared by GM and WM ICC tables.")
    dropped = sorted((set(gm_df["metric"]) | set(wm_df["metric"])) - common_metrics)
    if dropped:
        print(
            "[WARN] Excluding selected metrics absent from either GM or WM: "
            + ", ".join(dropped),
            flush=True,
        )
    return (
        gm_df[gm_df["metric"].isin(common_metrics)].copy(),
        wm_df[wm_df["metric"].isin(common_metrics)].copy(),
    )


def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required ICC file: {path}")
    return pd.read_csv(path)


def _plot_matrix(df: pd.DataFrame, region_col: str, title: str, out_png: Path) -> None:
    if df.empty:
        print(f"[WARN] No rows to plot for {title}")
        return
    pivot = df.pivot(index="metric", columns=region_col, values="ICC2_1")
    # Per user request: remove columns with NaNs after metric filtering.
    pivot = pivot.dropna(axis=1, how="any")
    # Drop rows that are fully NaN after column pruning.
    pivot = pivot.dropna(axis=0, how="all")
    if pivot.empty:
        print(f"[WARN] Empty matrix after dropping NaN columns for {title}")
        return
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    col_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot.loc[row_order, col_order]

    n_rows = len(pivot.index)
    n_cols = len(pivot.columns)
    fig_w = max(12, 0.22 * n_cols)
    # Keep enough vertical space for row labels plus dense x labels.
    fig_h = max(8, 0.45 * n_rows + 3.0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, label="ICC(2,1)")
    cbar.ax.tick_params(labelsize=8)

    y_fontsize = max(8, min(12, int(120 / max(n_rows, 1))))
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=y_fontsize)

    # For very wide matrices, thin x labels to avoid crushing the heatmap.
    if n_cols > 160:
        step = 4
    elif n_cols > 100:
        step = 3
    elif n_cols > 60:
        step = 2
    else:
        step = 1
    xticks = list(range(0, n_cols, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([pivot.columns[i] for i in xticks], rotation=90, fontsize=7)

    ax.set_title(title)
    fig.subplots_adjust(left=0.25, right=0.96, top=0.9, bottom=0.38)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"Wrote: {out_png}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--icc-dir",
        default="/cbica/projects/nibs/derivatives/ICC",
        help="Directory containing ICC summary CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="/cbica/projects/nibs/derivatives/ICC",
        help="Directory for cleaned outputs.",
    )
    parser.add_argument(
        "--qc-mode",
        choices=("metricqc", "completeqc"),
        default="metricqc",
        help="QC-filtered ICC version to clean and plot.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    icc_dir = Path(args.icc_dir)
    out_dir = Path(args.out_dir)

    # Inputs from existing pipelines
    gm_mean = _load_required(icc_dir / f"icc_summary_DKTatlas_mean_{args.qc_mode}.csv")
    gm_median = _load_required(icc_dir / f"icc_summary_DKTatlas_median_{args.qc_mode}.csv")
    wm_mean = _load_required(icc_dir / f"icc_summary_wm_bundles_masked_mean_{args.qc_mode}.csv")
    wm_median = _load_required(icc_dir / f"icc_summary_wm_bundles_masked_median_{args.qc_mode}.csv")

    gm_mean_clean = _prepare_icc_table(gm_mean)
    gm_median_clean = _prepare_icc_table(gm_median)
    wm_mean_clean = _prepare_icc_table(wm_mean)
    wm_median_clean = _prepare_icc_table(wm_median)
    gm_mean_clean, wm_mean_clean = _retain_common_metrics(gm_mean_clean, wm_mean_clean)
    gm_median_clean, wm_median_clean = _retain_common_metrics(
        gm_median_clean, wm_median_clean
    )

    print(
        f"[INFO] Rows after metric filtering | GM mean: {len(gm_mean_clean)}, "
        f"GM median: {len(gm_median_clean)}, "
        f"WM mean: {len(wm_mean_clean)}, WM median: {len(wm_median_clean)}"
    )

    _plot_matrix(
        gm_mean_clean,
        region_col="parcel",
        title="Gray Matter ICC Matrix (mean, clean)",
        out_png=out_dir / f"icc_matrix_DKTatlas_mean_{args.qc_mode}_clean.png",
    )
    _plot_matrix(
        gm_median_clean,
        region_col="parcel",
        title="Gray Matter ICC Matrix (median, clean)",
        out_png=out_dir / f"icc_matrix_DKTatlas_median_{args.qc_mode}_clean.png",
    )
    _plot_matrix(
        wm_mean_clean,
        region_col="bundle",
        title="White Matter ICC Matrix (mean, clean)",
        out_png=out_dir / f"icc_matrix_wm_bundles_masked_mean_{args.qc_mode}_clean.png",
    )
    _plot_matrix(
        wm_median_clean,
        region_col="bundle",
        title="White Matter ICC Matrix (median, clean)",
        out_png=out_dir / f"icc_matrix_wm_bundles_masked_median_{args.qc_mode}_clean.png",
    )

    print(f"Done. selected_metrics={sorted(SELECTED_METRICS)}")


if __name__ == "__main__":
    main()
