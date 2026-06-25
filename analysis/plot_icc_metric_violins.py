#!/usr/bin/env python3
"""Plot median ICC violins for selected metrics in GM and WM."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# User-requested metric list (adapted to canonical names used in CSVs).
SELECTED_METRICS = [
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
]


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


ALIAS_TO_CANONICAL = {
    "md": "MD",
    "mkt": "MKT",
    "fa": "FA",
    "icvf": "ICVF",
    "icvfmodulated": "ICVF-Modulated",
    "rtop": "RTOP",
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
    "qsmxr2e5x": "QSM-X-R2p-E5-X",
    "qsmxr2e5para": "QSM-X-R2p-E5-Para",
    "qsmxr2e5dia": "QSM-X-R2p-E5-Dia",
}


def canonicalize_metric(metric: str) -> str:
    metric = str(metric).strip()
    if "__" in metric:
        metric = metric.split("__")[-1]
    return ALIAS_TO_CANONICAL.get(_norm_token(metric), metric)


def wm_category(bundle: str) -> str:
    b = str(bundle)
    for prefix in (
        "Association",
        "Cerebellum",
        "Commissure",
        "CranialNerve",
        "ProjectionBasalGanglia",
        "ProjectionBrainstem",
    ):
        if b.startswith(prefix + "_"):
            return prefix
    return "Other"


def gm_category(parcel: str) -> str:
    p = str(parcel)
    # parcel format often includes hemi prefix, e.g. lh_ctx_lh_G_front_middle
    if "_" in p:
        p = p.split("_", 1)[1]
    l = p.lower()

    rules = [
        ("insula_opercular", ("insula", "_ins_", "opercular", "circular_insula")),
        ("cingulate_limbic", ("cingul", "pericallosal", "subcallosal", "parahip")),
        ("frontal", ("front", "precentral", "rectus", "orbital", "suborbital")),
        (
            "parietal",
            ("pariet", "postcentral", "precuneus", "intrapariet", "supramar", "angular", "subparietal"),
        ),
        ("temporal", ("temp", "temporal", "fusifor", "plan_tempo", "plan_polar", "pole_temporal")),
        ("occipital", ("occip", "cuneus", "calcarine", "lingual", "lunatus", "pole_occipital")),
        ("sylvian_fissure", ("lat_fis",)),
        ("medial_wall", ("medial_wall",)),
    ]
    for cat, toks in rules:
        if any(tok in l for tok in toks):
            return cat
    return "other"


def _build_palette(categories: list[str]) -> dict[str, str]:
    base = plt.get_cmap("tab20")
    pal = {}
    for idx, cat in enumerate(sorted(categories)):
        pal[cat] = matplotlib.colors.to_hex(base(idx % 20))
    return pal


def _plot_violin(
    df: pd.DataFrame,
    region_col: str,
    category_col: str,
    title: str,
    out_file: Path,
) -> None:
    if df.empty:
        print(f"[WARN] Empty dataframe for {title}")
        return

    metric_order = (
        df.groupby("metric")["ICC2_1"].mean().sort_values(ascending=False).index.tolist()
    )
    data = [df.loc[df["metric"] == metric, "ICC2_1"].dropna().to_numpy() for metric in metric_order]
    if not any(len(arr) > 0 for arr in data):
        print(f"[WARN] No plottable ICC values for {title}")
        return

    fig_w = max(11, 0.7 * len(metric_order))
    fig_h = 7.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vp = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.85,
    )
    for body in vp["bodies"]:
        body.set_facecolor("#d9d9d9")
        body.set_edgecolor("#4d4d4d")
        body.set_alpha(0.65)
    vp["cmedians"].set_color("#111111")
    vp["cmedians"].set_linewidth(1.3)

    rng = np.random.default_rng(42)
    categories = sorted(df[category_col].dropna().astype(str).unique().tolist())
    palette = _build_palette(categories)

    for idx, metric in enumerate(metric_order, start=1):
        sub = df[df["metric"] == metric]
        if sub.empty:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(sub))
        x = idx + jitter
        colors = sub[category_col].astype(str).map(palette).fillna("#777777").tolist()
        ax.scatter(
            x,
            sub["ICC2_1"].to_numpy(),
            s=18,
            c=colors,
            alpha=0.8,
            edgecolors="none",
            linewidths=0.0,
            zorder=3,
        )

    ax.set_xlim(0.4, len(metric_order) + 0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ICC(2,1)")
    ax.set_title(title)
    ax.set_xticks(np.arange(1, len(metric_order) + 1))
    ax.set_xticklabels(metric_order, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.2)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6, color=color, label=cat)
        for cat, color in palette.items()
    ]
    ax.legend(
        handles=legend_handles,
        title="Category",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=240)
    plt.close(fig)
    print(f"Wrote: {out_file}")


def _plot_metric_scatter(
    gm_df: pd.DataFrame, wm_df: pd.DataFrame, out_file: Path, agg: str = "median"
) -> None:
    """Plot metric-level GM-vs-WM ICC summary scatter."""
    if gm_df.empty or wm_df.empty:
        print("[WARN] Empty GM or WM dataframe for metric scatter.")
        return

    if agg == "median":
        gm_metric = gm_df.groupby("metric")["ICC2_1"].median().rename("gm_summary_icc")
        wm_metric = wm_df.groupby("metric")["ICC2_1"].median().rename("wm_summary_icc")
        agg_label = "Median"
    elif agg == "mean":
        # pandas mean is NaN-aware by default.
        gm_metric = gm_df.groupby("metric")["ICC2_1"].mean().rename("gm_summary_icc")
        wm_metric = wm_df.groupby("metric")["ICC2_1"].mean().rename("wm_summary_icc")
        agg_label = "Mean"
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    comp = pd.concat([gm_metric, wm_metric], axis=1).dropna()
    if comp.empty:
        print("[WARN] No overlapping metrics between GM and WM for scatter.")
        return

    # Order colors by descending average ICC for visual consistency.
    metric_order = (
        comp.mean(axis=1).sort_values(ascending=False).index.tolist()
    )
    cmap = plt.get_cmap("tab20")
    metric_colors = {
        metric: matplotlib.colors.to_hex(cmap(i % 20)) for i, metric in enumerate(metric_order)
    }

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    for metric in metric_order:
        x = comp.loc[metric, "gm_summary_icc"]
        y = comp.loc[metric, "wm_summary_icc"]
        color = metric_colors[metric]
        ax.scatter(x, y, s=75, c=color, edgecolors="black", linewidths=0.35, alpha=0.95)
        ax.text(x + 0.004, y + 0.004, metric, fontsize=8, color=color)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(f"{agg_label} ICC across GM parcels")
    ax.set_ylabel(f"{agg_label} ICC across WM bundles")
    ax.set_title(f"Metric-Level GM vs WM {agg_label} ICC")
    ax.grid(alpha=0.25)
    ax.axline((0, 0), slope=1, color="#777777", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=260)
    plt.close(fig)
    print(f"Wrote: {out_file}")


def _prep_df(df: pd.DataFrame, region_col: str, category_fn) -> pd.DataFrame:
    out = df.copy()
    out["metric"] = out["metric"].astype(str).map(canonicalize_metric)
    out = out[out["metric"].isin(SELECTED_METRICS)].copy()
    out = out[out["ICC2_1"].notna()].copy()
    out["category"] = out[region_col].astype(str).map(category_fn)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--icc-dir",
        default="/cbica/projects/nibs/derivatives/ICC",
        help="Directory with ICC summary CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        default="/cbica/projects/nibs/derivatives/ICC",
        help="Output directory for violin figures.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    icc_dir = Path(args.icc_dir)
    out_dir = Path(args.out_dir)

    gm_path = icc_dir / "icc_summary_a2009s_median.csv"
    wm_path = icc_dir / "icc_summary_wm_bundles_median.csv"
    if not gm_path.exists():
        raise FileNotFoundError(gm_path)
    if not wm_path.exists():
        raise FileNotFoundError(wm_path)

    gm_df = pd.read_csv(gm_path)
    wm_df = pd.read_csv(wm_path)

    gm_plot_df = _prep_df(gm_df, region_col="parcel", category_fn=gm_category)
    wm_plot_df = _prep_df(wm_df, region_col="bundle", category_fn=wm_category)

    _plot_violin(
        gm_plot_df,
        region_col="parcel",
        category_col="category",
        title="Gray Matter Median ICC by Metric",
        out_file=out_dir / "icc_violin_gm_median.png",
    )
    _plot_violin(
        wm_plot_df,
        region_col="bundle",
        category_col="category",
        title="White Matter Median ICC by Metric",
        out_file=out_dir / "icc_violin_wm_median.png",
    )
    _plot_metric_scatter(
        gm_plot_df,
        wm_plot_df,
        out_file=out_dir / "icc_scatter_gm_vs_wm_median_by_metric.png",
        agg="median",
    )
    _plot_metric_scatter(
        gm_plot_df,
        wm_plot_df,
        out_file=out_dir / "icc_scatter_gm_vs_wm_mean_by_metric.png",
        agg="mean",
    )


if __name__ == "__main__":
    main()
