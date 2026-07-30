#!/usr/bin/env python3
"""Compute test-retest discriminability for WM bundle and a2009s profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_icc_from_a2009s_stats import build_value_table, collect_rows
from compute_icc_from_bundle_scalarstats import collect_scalarstats


DEFAULT_WM_GLOBS = [
    "/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-*/sub-*/ses-*/dwi/sub-*_ses-*_*_scalarstats.tsv",
    "/cbica/projects/nibs/derivatives/bundle_myelin_stats/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-T1w_model-*_scalarstats.tsv",
]
DEFAULT_A2009S_GLOB = (
    "/cbica/projects/nibs/derivatives/parcel_myelin_stats/"
    "sub-*/sub-*_ses-*_run-*_desc-a2009s_scalarstats.csv"
)


def _value_column(df: pd.DataFrame, stat: str, prefer_masked: bool) -> pd.Series:
    masked_col = f"masked_{stat}"
    if prefer_masked and masked_col in df.columns:
        masked = pd.to_numeric(df[masked_col], errors="coerce")
        raw = pd.to_numeric(df[stat], errors="coerce")
        return masked.where(np.isfinite(masked.to_numpy(dtype=float)), raw)
    return pd.to_numeric(df[stat], errors="coerce")


def _zscore_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    keep = np.isfinite(stds.to_numpy(dtype=float)) & (stds.to_numpy(dtype=float) > 0)
    matrix = matrix.loc[:, keep]
    stds = stds.loc[matrix.columns]
    means = means.loc[matrix.columns]
    return (matrix - means) / stds


def _pairwise_distances(values: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        diffs = values[:, None, :] - values[None, :, :]
        return np.sqrt(np.sum(diffs * diffs, axis=2))

    if metric != "correlation":
        raise ValueError(f"Unsupported distance metric: {metric}")

    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    scaled = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    corr = np.clip(scaled @ scaled.T, -1.0, 1.0)
    return 1.0 - corr


def _score_profile_matrix(
    matrix: pd.DataFrame,
    group_name: str,
    profile_type: str,
    stat: str,
    distance_metric: str,
) -> dict[str, object] | None:
    if matrix.empty:
        return None

    matrix = matrix.sort_index()
    subjects = matrix.index.get_level_values("subject").astype(str).to_numpy()
    sessions = matrix.index.get_level_values("session").astype(str).to_numpy()
    values = matrix.to_numpy(dtype=float)

    paired_subjects = pd.Series(sessions, index=subjects).groupby(level=0).nunique()
    paired_subjects = paired_subjects[paired_subjects >= 2].index
    keep_rows = np.isin(subjects, paired_subjects)
    matrix = matrix.iloc[keep_rows, :]
    subjects = subjects[keep_rows]
    sessions = sessions[keep_rows]
    values = matrix.to_numpy(dtype=float)

    if len(np.unique(subjects)) < 2 or len(np.unique(sessions)) < 2 or values.shape[1] < 2:
        return None

    distances = _pairwise_distances(values, distance_metric)
    scores: list[float] = []
    nearest_correct: list[float] = []
    genuine_distances: list[float] = []
    impostor_distances: list[float] = []
    rank_percentiles: list[float] = []

    for i, subject in enumerate(subjects):
        valid = np.arange(len(subjects)) != i
        same = (subjects == subject) & valid
        other = (subjects != subject) & valid
        if not same.any() or not other.any():
            continue

        genuine = distances[i, same]
        impostor = distances[i, other]
        genuine_min = float(np.min(genuine))

        scores.append(float(np.mean(impostor > genuine_min)))
        nearest_idx = np.argmin(np.where(valid, distances[i, :], np.inf))
        nearest_correct.append(float(subjects[nearest_idx] == subject))
        genuine_distances.append(genuine_min)
        impostor_distances.append(float(np.mean(impostor)))

        all_valid_distances = distances[i, valid]
        rank = 1 + int(np.sum(all_valid_distances < genuine_min))
        denom = max(len(all_valid_distances) - 1, 1)
        rank_percentiles.append(float(1.0 - (rank - 1) / denom))

    if not scores:
        return None

    return {
        "profile_type": profile_type,
        "profile_group": group_name,
        "stat": stat,
        "distance_metric": distance_metric,
        "discriminability": float(np.mean(scores)),
        "nearest_neighbor_accuracy": float(np.mean(nearest_correct)),
        "mean_genuine_distance": float(np.mean(genuine_distances)),
        "mean_impostor_distance": float(np.mean(impostor_distances)),
        "mean_rank_percentile": float(np.mean(rank_percentiles)),
        "n_subjects": int(len(np.unique(subjects))),
        "n_sessions": int(len(np.unique(sessions))),
        "n_profiles": int(len(subjects)),
        "n_features": int(values.shape[1]),
    }


def _build_profile_matrix(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    min_feature_coverage: float,
    min_profile_coverage: float,
    zscore_features: bool,
) -> pd.DataFrame:
    grouped = (
        df[["subject", "session", feature_col, value_col]]
        .dropna(subset=[value_col])
        .groupby(["subject", "session", feature_col], as_index=False)[value_col]
        .mean()
    )
    matrix = grouped.pivot_table(
        index=["subject", "session"],
        columns=feature_col,
        values=value_col,
        aggfunc="mean",
    )
    if matrix.empty:
        return matrix

    feature_coverage = matrix.notna().mean(axis=0)
    matrix = matrix.loc[:, feature_coverage >= min_feature_coverage]
    profile_coverage = matrix.notna().mean(axis=1)
    matrix = matrix.loc[profile_coverage >= min_profile_coverage, :]
    if matrix.empty:
        return matrix

    matrix = matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
    matrix = matrix.dropna(axis=1, how="any")
    if zscore_features:
        matrix = _zscore_columns(matrix)
    return matrix


def _compute_discriminability(
    long_df: pd.DataFrame,
    profile_type: str,
    stat: str,
    min_feature_coverage: float,
    min_profile_coverage: float,
    distance_metric: str,
    zscore_features: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    value_col = "value"

    all_df = long_df.copy()
    all_df["metric_feature"] = all_df["metric"].astype(str) + "|" + all_df["feature"].astype(str)
    all_matrix = _build_profile_matrix(
        all_df,
        feature_col="metric_feature",
        value_col=value_col,
        min_feature_coverage=min_feature_coverage,
        min_profile_coverage=min_profile_coverage,
        zscore_features=zscore_features,
    )
    all_score = _score_profile_matrix(
        all_matrix,
        group_name="ALL_METRICS",
        profile_type=profile_type,
        stat=stat,
        distance_metric=distance_metric,
    )
    if all_score is not None:
        rows.append(all_score)

    for metric_name, metric_df in long_df.groupby("metric", sort=True):
        matrix = _build_profile_matrix(
            metric_df,
            feature_col="feature",
            value_col=value_col,
            min_feature_coverage=min_feature_coverage,
            min_profile_coverage=min_profile_coverage,
            zscore_features=zscore_features,
        )
        score = _score_profile_matrix(
            matrix,
            group_name=str(metric_name),
            profile_type=profile_type,
            stat=stat,
            distance_metric=distance_metric,
        )
        if score is not None:
            rows.append(score)

    return pd.DataFrame(rows).sort_values(["profile_type", "profile_group"]).reset_index(drop=True)


def load_wm_long_df(input_globs: list[str], stat: str, prefer_masked: bool) -> pd.DataFrame:
    df = collect_scalarstats(input_globs)
    if df.empty:
        raise RuntimeError(f"No WM bundle scalarstats found for globs: {input_globs}")
    df = df.copy()
    df["subject"] = df["subject_id"].astype(str)
    df["session"] = df["session_id"].astype(str)
    df["feature"] = df["bundle"].astype(str)
    df["value"] = _value_column(df, stat=stat, prefer_masked=prefer_masked)
    return df[["subject", "session", "metric", "feature", "value"]]


def load_a2009s_long_df(input_glob: str, stat: str) -> pd.DataFrame:
    rows = collect_rows(input_glob)
    if rows.empty:
        raise RuntimeError(f"No a2009s parcel stats found for glob: {input_glob}")
    value_df = build_value_table(rows, stat=stat)
    value_df = value_df.copy()
    value_df["feature"] = value_df["parcel"].astype(str)
    value_df = value_df.rename(columns={"subject": "subject", "session": "session"})
    value_df["value"] = pd.to_numeric(value_df["value"], errors="coerce")
    return value_df[["subject", "session", "metric", "feature", "value"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        choices=("wm", "a2009s", "both"),
        default="both",
        help="Which profile discriminability analysis to run.",
    )
    parser.add_argument(
        "--stat",
        choices=("mean", "median"),
        default="median",
        help="Scalar statistic to use.",
    )
    parser.add_argument(
        "--prefer-masked",
        action="store_true",
        help="For WM bundle TSVs, prefer masked_mean/masked_median when available.",
    )
    parser.add_argument(
        "--distance-metric",
        choices=("correlation", "euclidean"),
        default="correlation",
        help="Distance metric between subject-session profiles.",
    )
    parser.add_argument(
        "--no-zscore",
        action="store_true",
        help="Do not z-score features before computing distances.",
    )
    parser.add_argument(
        "--min-feature-coverage",
        type=float,
        default=0.8,
        help="Minimum fraction of profiles with finite data required for a feature.",
    )
    parser.add_argument(
        "--min-profile-coverage",
        type=float,
        default=0.8,
        help="Minimum fraction of retained features required for a subject-session profile.",
    )
    parser.add_argument(
        "--wm-input-globs",
        nargs="+",
        default=DEFAULT_WM_GLOBS,
        help="Input globs for WM bundle scalarstats TSVs.",
    )
    parser.add_argument(
        "--a2009s-input-glob",
        default=DEFAULT_A2009S_GLOB,
        help="Input glob for a2009s parcel stats CSVs.",
    )
    parser.add_argument(
        "--outdir",
        default="/cbica/projects/nibs/derivatives/ICC",
        help="Output directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    zscore_features = not args.no_zscore

    if args.analysis in {"wm", "both"}:
        wm_df = load_wm_long_df(
            input_globs=args.wm_input_globs,
            stat=args.stat,
            prefer_masked=args.prefer_masked,
        )
        wm_out = _compute_discriminability(
            wm_df,
            profile_type="wm_bundles",
            stat=args.stat,
            min_feature_coverage=args.min_feature_coverage,
            min_profile_coverage=args.min_profile_coverage,
            distance_metric=args.distance_metric,
            zscore_features=zscore_features,
        )
        suffix = f"{args.stat}_{args.distance_metric}"
        if args.prefer_masked:
            suffix = f"masked_preferred_{suffix}"
        out_csv = outdir / f"discriminability_wm_bundles_{suffix}.csv"
        wm_out.to_csv(out_csv, index=False)
        print(f"Wrote: {out_csv}", flush=True)

    if args.analysis in {"a2009s", "both"}:
        a2009s_df = load_a2009s_long_df(args.a2009s_input_glob, stat=args.stat)
        a2009s_out = _compute_discriminability(
            a2009s_df,
            profile_type="a2009s_parcels",
            stat=args.stat,
            min_feature_coverage=args.min_feature_coverage,
            min_profile_coverage=args.min_profile_coverage,
            distance_metric=args.distance_metric,
            zscore_features=zscore_features,
        )
        out_csv = outdir / f"discriminability_a2009s_{args.stat}_{args.distance_metric}.csv"
        a2009s_out.to_csv(out_csv, index=False)
        print(f"Wrote: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
