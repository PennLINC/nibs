#!/usr/bin/env python3
"""Compute test-retest ICCs from QSIRecon bundle scalarstats TSV files."""

from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path
from collections import Counter

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


PATH_RE = re.compile(r"sub-(?P<sub>[^_/]+).*(ses-(?P<ses>[^_/]+))")
REQUIRED_COLUMNS = {"bundle", "variable_name", "mean", "median"}
DKI_METRICS = {
    "FA",
    "KFA",
    "KFA-Micro",
    "AD",
    "AD-Micro",
    "ADE-Micro",
    "AWF-Micro",
    "AxonALD-Micro",
    "AK",
    "MD",
    "MD-Micro",
    "MK",
    "MKT",
    "RD",
    "RD-Micro",
    "RDE-Micro",
    "RK",
    "Linearity",
    "Planarity",
    "Sphericity",
    "Trace-Micro",
    "Tortuosity-Micro",
}
NODDI_METRICS = {
    "ICVF-Modulated",
    "ICVF",
    "ISOVF",
    "NRMSE",
    "RMSE",
    "OD-Modulated",
    "OD",
    "TF",
}
MAPMRI_METRICS = {"NG", "NGPar", "NGPerp", "PA", "PAth", "RTAP", "RTOP", "RTPP"}
MYELIN_METRICS = {
    "MEGRE",
    "QSM-SEPIA-E5",
    "QSM-X-R2p-E5-X",
    "QSM-X-R2p-E5-Para",
    "QSM-X-R2p-E5-Dia",
    "ihMTw",
    "ihMTR",
    "MTR",
    "ihMTsat",
    "ihMTsat-B1c",
    "R1",
    "R1-B1c",
    "MPRAGE-MyelinW",
    "SPACE-MyelinW",
    "Scaled MPRAGE-MyelinW",
    "Scaled SPACE-MyelinW",
}
ALL_ALLOWED_METRICS = DKI_METRICS | NODDI_METRICS | MAPMRI_METRICS | MYELIN_METRICS

DKI_STD_MAP = {
    "fa": "FA",
    "kfa": "KFA",
    "ad": "AD",
    "ak": "AK",
    "md": "MD",
    "mk": "MK",
    "mkt": "MKT",
    "rd": "RD",
    "rk": "RK",
    "linearity": "Linearity",
    "planarity": "Planarity",
    "sphericity": "Sphericity",
}
DKI_MICRO_MAP = {
    "kfa": "KFA-Micro",
    "ad": "AD-Micro",
    "ade": "ADE-Micro",
    "awf": "AWF-Micro",
    "axonald": "AxonALD-Micro",
    "md": "MD-Micro",
    "rd": "RD-Micro",
    "rde": "RDE-Micro",
    "trace": "Trace-Micro",
    "tortuosity": "Tortuosity-Micro",
}
NODDI_MAP = {
    "icvf": "ICVF",
    "isovf": "ISOVF",
    "nrmse": "NRMSE",
    "rmse": "RMSE",
    "od": "OD",
    "tf": "TF",
}
MAPMRI_MAP = {
    "ng": "NG",
    "ngpar": "NGPar",
    "ngperp": "NGPerp",
    "pa": "PA",
    "path": "PAth",
    "rtap": "RTAP",
    "rtop": "RTOP",
    "rtpp": "RTPP",
}
DIRECT_NAME_MAP = {metric.lower(): metric for metric in ALL_ALLOWED_METRICS}


def compute_icc2_fallback(values: np.ndarray, subjects: np.ndarray, sessions: np.ndarray) -> float:
    """Compute ICC(2,1) with complete-case fallback."""
    subs_unique, sub_idx = np.unique(subjects, return_inverse=True)
    sess_unique, ses_idx = np.unique(sessions, return_inverse=True)
    n_sub, n_ses = len(subs_unique), len(sess_unique)
    if n_sub < 2 or n_ses < 2:
        return np.nan

    matrix = np.full((n_sub, n_ses), np.nan, dtype=float)
    for val, i_sub, i_ses in zip(values, sub_idx, ses_idx):
        matrix[i_sub, i_ses] = val

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


def _parse_from_path(path: str) -> tuple[str | None, str | None]:
    match = PATH_RE.search(path)
    if not match:
        return None, None
    return match.group("sub"), f"ses-{match.group('ses')}"


def _extract_param_token(source_file: str) -> str:
    match = re.search(r"_param-([^_]+)", source_file)
    if not match:
        return ""
    return match.group(1).lower()


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _infer_metric_name(row: pd.Series, source_tsv: str) -> str | None:
    var_name = str(row.get("variable_name", "")).strip()
    qsirecon_suffix = str(row.get("qsirecon_suffix", "")).strip()
    source_file = str(row.get("source_file", "")).strip()
    lowered_var = var_name.lower()
    norm_var = _norm_token(var_name)
    lowered_tsv = source_tsv.lower()
    lowered_src = source_file.lower()
    lowered_suffix = qsirecon_suffix.lower()

    # Generated myelin spreadsheet: variable_name should already be canonical.
    if "/bundle_myelin_stats/" in lowered_tsv:
        metric = DIRECT_NAME_MAP.get(lowered_var)
        return metric if metric in MYELIN_METRICS else None

    # DWI DKI spreadsheet
    is_dki = (
        "qsirecon-dipydki" in lowered_tsv
        or "qsirecon-dipydki" in lowered_src
        or "dipydki" in lowered_suffix
        or "dki" in lowered_suffix
    )
    if is_dki:
        param = _extract_param_token(lowered_src)
        is_micro = (
            "model-dkimicro" in lowered_src
            or "dkimicro" in lowered_suffix
            or "micro" in norm_var
        )
        if is_micro:
            metric = DKI_MICRO_MAP.get(param) or DKI_MICRO_MAP.get(lowered_var)
            if metric is None:
                micro_aliases = (
                    ("axonald", "AxonALD-Micro"),
                    ("tortuosity", "Tortuosity-Micro"),
                    ("trace", "Trace-Micro"),
                    ("awf", "AWF-Micro"),
                    ("ade", "ADE-Micro"),
                    ("rde", "RDE-Micro"),
                    ("kfa", "KFA-Micro"),
                    ("ad", "AD-Micro"),
                    ("md", "MD-Micro"),
                    ("rd", "RD-Micro"),
                )
                for token, out_name in micro_aliases:
                    if token in norm_var:
                        metric = out_name
                        break
        else:
            metric = DKI_STD_MAP.get(param) or DKI_STD_MAP.get(lowered_var)
            if metric is None:
                std_aliases = (
                    ("linearity", "Linearity"),
                    ("planarity", "Planarity"),
                    ("sphericity", "Sphericity"),
                    ("mkt", "MKT"),
                    ("kfa", "KFA"),
                    ("fa", "FA"),
                    ("ak", "AK"),
                    ("mk", "MK"),
                    ("rk", "RK"),
                    ("ad", "AD"),
                    ("md", "MD"),
                    ("rd", "RD"),
                )
                for token, out_name in std_aliases:
                    if token in norm_var:
                        metric = out_name
                        break
        return metric if metric in DKI_METRICS else None

    # DWI NODDI spreadsheet
    is_noddi = (
        "qsirecon-noddi" in lowered_tsv
        or "qsirecon-noddi" in lowered_src
        or "noddi" in lowered_suffix
    )
    if is_noddi:
        param = _extract_param_token(lowered_src) or lowered_var
        metric = NODDI_MAP.get(param)
        if metric is None:
            for token, out_name in (
                ("isovf", "ISOVF"),
                ("icvf", "ICVF"),
                ("nrmse", "NRMSE"),
                ("rmse", "RMSE"),
                ("od", "OD"),
                ("tf", "TF"),
            ):
                if token in norm_var:
                    metric = out_name
                    break
        is_modulated = (
            "desc-modulated" in lowered_src
            or "modulated" in lowered_src
            or "modulated" in lowered_var
            or "modulated" in lowered_suffix
        )
        if metric in {"ICVF", "OD"} and is_modulated:
            metric = f"{metric}-Modulated"
        return metric if metric in NODDI_METRICS else None

    # DWI MAPMRI spreadsheet
    is_mapmri = (
        "qsirecon-tortoise_model-mapmri" in lowered_tsv
        or "qsirecon-tortoise_model-mapmri" in lowered_src
        or "mapmri" in lowered_suffix
    )
    if is_mapmri:
        param = _extract_param_token(lowered_src) or lowered_var
        metric = MAPMRI_MAP.get(param)
        if metric is None:
            for token, out_name in (
                ("ngperp", "NGPerp"),
                ("ngpar", "NGPar"),
                ("ng", "NG"),
                ("path", "PAth"),
                ("pa", "PA"),
                ("rtap", "RTAP"),
                ("rtop", "RTOP"),
                ("rtpp", "RTPP"),
            ):
                if token in norm_var:
                    metric = out_name
                    break
        return metric if metric in MAPMRI_METRICS else None

    # Unknown source folder -> reject rather than mixing source families.
    return None


def collect_scalarstats(input_globs: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    all_files: set[str] = set()
    dropped_counter: Counter[tuple[str, str]] = Counter()
    for input_glob in input_globs:
        for file_path in glob(input_glob):
            all_files.add(file_path)

    for file_path in sorted(all_files):
        df = pd.read_csv(file_path, sep="\t")
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise RuntimeError(f"Missing required columns {missing} in {file_path}")

        if "subject_id" not in df.columns or df["subject_id"].isna().all():
            parsed_sub, _ = _parse_from_path(file_path)
            if parsed_sub is None:
                raise RuntimeError(f"Could not infer subject_id from {file_path}")
            df["subject_id"] = parsed_sub

        if "session_id" not in df.columns or df["session_id"].isna().all():
            _, parsed_ses = _parse_from_path(file_path)
            if parsed_ses is None:
                raise RuntimeError(f"Could not infer session_id from {file_path}")
            df["session_id"] = parsed_ses

        # Source-aware canonical metric mapping.
        df["metric"] = df.apply(lambda row: _infer_metric_name(row, file_path), axis=1)
        dropped_df = df[df["metric"].isna()]
        for _, drow in dropped_df.iterrows():
            dropped_counter[(str(drow.get("variable_name", "")), str(drow.get("qsirecon_suffix", "")))] += 1
        df = df[df["metric"].isin(ALL_ALLOWED_METRICS)].copy()
        if df.empty:
            continue
        df["source_tsv"] = file_path
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    all_df["subject_id"] = all_df["subject_id"].astype(str).str.replace("^sub-", "", regex=True)
    all_df["session_id"] = all_df["session_id"].astype(str)
    all_df["bundle"] = all_df["bundle"].astype(str)
    all_df["metric"] = all_df["metric"].astype(str)
    if dropped_counter:
        print("[WARN] Dropped rows with unmapped metrics (top 20):", flush=True)
        for (var_name, suffix), count in dropped_counter.most_common(20):
            print(f"  variable_name={var_name} qsirecon_suffix={suffix} n={count}", flush=True)
    return all_df


def compute_icc_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    grp = df.groupby(["metric", "bundle"], sort=True)
    for (metric, bundle), dfg in grp:
        dfg = dfg[np.isfinite(dfg[value_col].to_numpy(dtype=float))].copy()
        if dfg.empty:
            continue

        ses_count = dfg.groupby("subject_id")["session_id"].nunique()
        valid_subs = ses_count[ses_count >= 2].index
        dfg = dfg[dfg["subject_id"].isin(valid_subs)]
        if dfg["subject_id"].nunique() < 2 or dfg["session_id"].nunique() < 2:
            continue

        subjects = dfg["subject_id"].to_numpy()
        sessions = dfg["session_id"].to_numpy()
        values = dfg[value_col].to_numpy(dtype=float)

        icc = np.nan
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
                icc = float(icc_row["ICC"])
                ci95 = str(icc_row.get("CI95%", ""))
                f_val = float(icc_row.get("F", np.nan))
                df1 = float(icc_row.get("df1", np.nan))
                df2 = float(icc_row.get("df2", np.nan))
                pval = float(icc_row.get("pval", np.nan))
            except Exception:
                icc = compute_icc2_fallback(values, subjects, sessions)
        else:
            icc = compute_icc2_fallback(values, subjects, sessions)

        out_rows.append(
            {
                "metric": metric,
                "bundle": bundle,
                "stat": value_col,
                "ICC2_1": icc,
                "CI95": ci95,
                "F": f_val,
                "df1": df1,
                "df2": df2,
                "pval": pval,
                "n_subjects": int(dfg["subject_id"].nunique()),
                "n_sessions": int(dfg["session_id"].nunique()),
            }
        )

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows).sort_values(["metric", "bundle"]).reset_index(drop=True)


def plot_heatmap(df_icc: pd.DataFrame, out_png: Path, title_suffix: str) -> None:
    pivot = df_icc.pivot(index="metric", columns="bundle", values="ICC2_1")
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    col_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot.loc[row_order, col_order]

    fig_w = max(12, 0.22 * len(pivot.columns))
    fig_h = max(6, 0.28 * len(pivot.index))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, label="ICC(2,1)")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.title(f"WM Bundle ICC Heatmap ({title_suffix})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-globs",
        nargs="+",
        default=[
            "/cbica/projects/nibs/derivatives/qsirecon/derivatives/qsirecon-*/sub-*/ses-*/dwi/sub-*_ses-*_*_scalarstats.tsv",
            "/cbica/projects/nibs/derivatives/bundle_myelin_stats/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-T1w_model-gqi_scalarstats.tsv",
        ],
        help="One or more globs for bundle scalarstats TSV files.",
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

    all_df = collect_scalarstats(args.input_globs)
    if all_df.empty:
        raise RuntimeError(f"No scalarstats TSV files found for globs: {args.input_globs}")

    for stat in ("mean", "median"):
        icc_df = compute_icc_table(all_df, value_col=stat)
        if icc_df.empty:
            raise RuntimeError(f"No valid ICC rows for {stat}. Check session coverage.")

        out_csv = outdir / f"icc_summary_wm_bundles_{stat}.csv"
        out_png = outdir / f"icc_heatmap_wm_bundles_{stat}.png"
        icc_df.to_csv(out_csv, index=False)
        plot_heatmap(icc_df, out_png, stat)

        print(f"Wrote: {out_csv}", flush=True)
        print(f"Wrote: {out_png}", flush=True)


if __name__ == "__main__":
    main()
