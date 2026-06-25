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
    "MD",  # DKI MD
    "MKT",  # DKI MKT
    "FA",  # DKI Tensor FA
    "ICVF",  # NODDI ICVF
    "ICVF-Modulated",  # NODDI ICVF Modulated
    "RTOP",  # TORTOISE MAPMRI RTOP
    "ihMTR",
    "ihMTsat-B1c",
    "R1",
    "R1-B1c",
    "MPRAGE-MyelinW",
    "SPACE-MyelinW",
    "G-ihMTsat",
    "G-ihMTR",
    "QSM-SEPIA-E5",
    "QSM-X-R2p-E5-X",  # QSM-X-R2'-E5-X
    "QSM-X-R2p-E5-Para",  # QSM-X-R2'-E5-Para
    "QSM-X-R2p-E5-Dia",  # QSM-X-R2'-E5-Dia
}


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


ALIAS_TO_CANONICAL = {
    # DWI selected
    "dkimd": "MD",
    "md": "MD",
    "dkimicromd": "MD-Micro",
    "mdmicro": "MD-Micro",
    "dkitensorfa": "FA",
    "fa": "FA",
    "dkimkt": "MKT",
    "mkt": "MKT",
    "noddiicvf": "ICVF",
    "icvf": "ICVF",
    "noddiicvfmodulated": "ICVF-Modulated",
    "icvfmodulated": "ICVF-Modulated",
    "tortoisemapmrirtop": "RTOP",
    "rtop": "RTOP",
    # myelin selected
    "ihmtr": "ihMTR",
    "ihmtsatb1c": "ihMTsat-B1c",
    "r1": "R1",
    "r1b1c": "R1-B1c",
    "mpragemyelinw": "MPRAGE-MyelinW",
    "spacemyelinw": "SPACE-MyelinW",
    "gihmtsat": "G-ihMTsat",
    "gihmtr": "G-ihMTR",
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    icc_dir = Path(args.icc_dir)
    out_dir = Path(args.out_dir)

    # Inputs from existing pipelines
    gm_mean = _load_required(icc_dir / "icc_summary_a2009s_mean.csv")
    gm_median = _load_required(icc_dir / "icc_summary_a2009s_median.csv")
    wm_mean = _load_required(icc_dir / "icc_summary_wm_bundles_mean.csv")
    wm_median = _load_required(icc_dir / "icc_summary_wm_bundles_median.csv")

    gm_mean_clean = _prepare_icc_table(gm_mean)
    gm_median_clean = _prepare_icc_table(gm_median)
    wm_mean_clean = _prepare_icc_table(wm_mean)
    wm_median_clean = _prepare_icc_table(wm_median)

    print(
        f"[INFO] Rows after metric filtering | GM mean: {len(gm_mean_clean)}, "
        f"GM median: {len(gm_median_clean)}, "
        f"WM mean: {len(wm_mean_clean)}, WM median: {len(wm_median_clean)}"
    )

    _plot_matrix(
        gm_mean_clean,
        region_col="parcel",
        title="Gray Matter ICC Matrix (mean, clean)",
        out_png=out_dir / "icc_matrix_a2009s_mean_clean.png",
    )
    _plot_matrix(
        gm_median_clean,
        region_col="parcel",
        title="Gray Matter ICC Matrix (median, clean)",
        out_png=out_dir / "icc_matrix_a2009s_median_clean.png",
    )
    _plot_matrix(
        wm_mean_clean,
        region_col="bundle",
        title="White Matter ICC Matrix (mean, clean)",
        out_png=out_dir / "icc_matrix_wm_bundles_mean_clean.png",
    )
    _plot_matrix(
        wm_median_clean,
        region_col="bundle",
        title="White Matter ICC Matrix (median, clean)",
        out_png=out_dir / "icc_matrix_wm_bundles_median_clean.png",
    )

    print(f"Done. selected_metrics={sorted(SELECTED_METRICS)}")


if __name__ == "__main__":
    main()
