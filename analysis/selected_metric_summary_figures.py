#!/usr/bin/env python3
"""Generate selected metric summary figures.

This script is generated from selected_metric_summary_figures.ipynb and is
intended for cluster execution. It writes the same selected/all-metric
correlation matrices, ICC figures, and nearest-neighbor figures.
"""


# %% [markdown] # Selected metric summary figures

# %% Notebook code cell 1
from __future__ import annotations

import importlib.util
import os
import re
import warnings
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import argparse
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value)
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

sns.set_theme(style="whitegrid", context="notebook")
mpl.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
    }
)

# %% [markdown] ## Paths and analysis settings

# %% Notebook code cell 3
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=os.environ.get("NIBS_PROJECT_ROOT", "/cbica/projects/nibs"),
        help="Project root containing derivatives and code/data.",
    )
    parser.add_argument(
        "--qc-mode",
        choices=("metricqc", "completeqc"),
        default=os.environ.get("NIBS_QC_MODE", "metricqc"),
        help="QC mode for ICC/discriminability inputs and raw profile correlations.",
    )
    parser.add_argument(
        "--qc-file",
        default=None,
        help="Manual modality QC TSV. Defaults to <project-root>/code/data/manual_qc_modality.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <project-root>/derivatives/ICC/selected_metric_summary_figures.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Raw profile cache directory. Defaults to <output-dir>/cache.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse existing cached raw profiles instead of rebuilding them.",
    )
    parser.add_argument(
        "--min-common-regions",
        type=int,
        default=10,
        help="Minimum matched regions for each within-profile Spearman correlation.",
    )
    return parser.parse_args()


ARGS = parse_args()
PROJECT_ROOT = Path(ARGS.project_root)
ICC_DIR = PROJECT_ROOT / "derivatives" / "ICC"
OUTPUT_DIR = Path(ARGS.output_dir) if ARGS.output_dir else ICC_DIR / "selected_metric_summary_figures"
CACHE_DIR = Path(ARGS.cache_dir) if ARGS.cache_dir else OUTPUT_DIR / "cache"
QC_MODE = ARGS.qc_mode
QC_FILE = Path(ARGS.qc_file) if ARGS.qc_file else PROJECT_ROOT / "code" / "data" / "manual_qc_modality.tsv"

WM_INPUT_GLOBS = [
    str(
        PROJECT_ROOT
        / "derivatives/qsirecon/derivatives/qsirecon-*/sub-*/ses-*/dwi/"
        "sub-*_ses-*_*_scalarstats.tsv"
    ),
    str(
        PROJECT_ROOT
        / "derivatives/bundle_myelin_stats/sub-*/ses-*/dwi/"
        "sub-*_ses-*_acq-HBCD75_run-01_space-T1w_model-*_scalarstats.tsv"
    ),
]
GM_INPUT_GLOB = str(
    PROJECT_ROOT
    / "derivatives/DKTatlas_myelin_stats/sub-*/"
    "sub-*_ses-*_run-*_desc-DKTatlas_scalarstats.csv"
)

# Exact statistic-specific inputs. These deliberately do not fall back to mean.
WM_ICC_CSV = ICC_DIR / f"icc_summary_wm_bundles_masked_median_{QC_MODE}.csv"
GM_ICC_CSV = ICC_DIR / f"icc_summary_DKTatlas_median_{QC_MODE}.csv"
WM_NN_CSV = ICC_DIR / f"discriminability_wm_bundles_masked_preferred_median_correlation_{QC_MODE}.csv"
GM_NN_CSV = ICC_DIR / f"discriminability_DKTatlas_median_correlation_{QC_MODE}.csv"

REBUILD_RAW_CACHE = not ARGS.use_cache
MIN_COMMON_REGIONS = ARGS.min_common_regions

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"QC mode: {QC_MODE}")

# %% [markdown] ## Selected metrics and colors

# %% Notebook code cell 5
@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    family: str
    aliases: tuple[str, ...]


SELECTED_METRICS = [
    MetricSpec("tortoise_fa", "FA", "Tensor", ("TORTOISE-InnerShell-FA",)),
    MetricSpec("tortoise_md", "MD", "Tensor", ("TORTOISE-InnerShell-MD",)),
    MetricSpec("tortoise_rd", "RD", "Tensor", ("TORTOISE-InnerShell-RD",)),
    MetricSpec("noddi_icvf", "ICVF", "NODDI", ("NODDI-ICVF",)),
    MetricSpec("dki_mkt", "MKT", "DKI", ("DKI-MKT",)),
    MetricSpec("dki_rk", "RK", "DKI", ("DKI-RK",)),
    MetricSpec("mapmri_rtop", "RTOP", "MAPMRI", ("MAPMRI-RTOP",)),
    MetricSpec("mapmri_rtap", "RTAP", "MAPMRI", ("MAPMRI-RTAP",)),
    MetricSpec("mapmri_ng", "NG", "MAPMRI", ("MAPMRI-NG",)),
    MetricSpec("gqi_gfa", "GFA", "GQI", ("DSIStudio-GQI-GFA",)),
    MetricSpec(
        "qsm_dia",
        "QSM-X-R2p-E5-Dia",
        "QSM",
        ("QSM-X-R2p-E5-Dia", "QSM-X-R2'-E5-Dia"),
    ),
    MetricSpec(
        "qsm_para",
        "QSM-X-R2p-E5-Para",
        "QSM",
        ("QSM-X-R2p-E5-Para", "QSM-X-R2'-E5-Para"),
    ),
    MetricSpec(
        "qsm_x",
        "QSM-X-R2p-E5-X",
        "QSM",
        ("QSM-X-R2p-E5-X", "QSM-X-R2'-E5-X"),
    ),
    MetricSpec(
        "qsm_e5",
        "QSM-SEPIA-E5",
        "QSM",
        ("QSM-SEPIA-E5", "QSM-X-R2p-E5", "QSM-X-R2'-E5"),
    ),
    MetricSpec(
        "scaled_mprage_myelinw",
        "Scaled MPRAGE-MyelinW",
        "T1w/T2w",
        ("Scaled MPRAGE-MyelinW",),
    ),
    MetricSpec(
        "scaled_space_myelinw",
        "Scaled SPACE-MyelinW",
        "T1w/T2w",
        ("Scaled SPACE-MyelinW", "Scaled Space-MyelinW"),
    ),
    MetricSpec("ihmtr", "ihMTR", "ihMTR", ("ihMTR",)),
    MetricSpec("ihmtsat_b1c", "ihMTsat-B1c", "ihMTR", ("ihMTsat-B1c",)),
    MetricSpec("g_ihmtsat", "G-ihMTsat", "g-ratio", ("G-ihMTsat",)),
    MetricSpec("g_ihmtr", "G-ihMTR", "g-ratio", ("G-ihMTR",)),
    MetricSpec("r1_b1c", "R1-B1c", "R1", ("R1-B1c",)),
    MetricSpec("r1", "R1", "R1", ("R1",)),
]

FAMILY_COLORS = {
    "Tensor": "#EE7733",
    "NODDI": "#AA3377",
    "DKI": "#CC3311",
    "MAPMRI": "#228833",
    "GQI": "#EECC66",
    "QSM": "#4477AA",
    "T1w/T2w": "#8C564B",
    "ihMTR": "#CC66AA",
    "g-ratio": "#555555",
    "R1": "#66CCEE",
}
ALL_FAMILY_COLORS = {
    **FAMILY_COLORS,
    "MT": "#882255",
    "MEGRE": "#44AA99",
    "Other": "#999999",
}
SOURCE_IMAGE_COLORS = {
    "DWI": "#4477AA",
    "QSM": "#AA3377",
    "T1w/T2w": "#CCBB44",
    "ihMT": "#228833",
    "g-ratio": "#555555",
    "R1": "#EE7733",
    "MT": "#66CCEE",
    "MEGRE": "#882255",
    "Other": "#999999",
}
FAMILY_ORDER = {family: index for index, family in enumerate(FAMILY_COLORS)}
SOURCE_IMAGE_ORDER = {
    source: index for index, source in enumerate(SOURCE_IMAGE_COLORS)
}


def source_image_from_family(family: object) -> str:
    family_label = str(family)
    if family_label in {"Tensor", "NODDI", "DKI", "MAPMRI", "GQI"}:
        return "DWI"
    if family_label == "ihMTR":
        return "ihMT"
    if family_label in SOURCE_IMAGE_COLORS:
        return family_label
    return "Other"


def shade_color(base_color: str, amount: float) -> str:
    """Lighten positive amounts toward white; darken negatives toward black."""
    rgb = np.asarray(mpl.colors.to_rgb(base_color), dtype=float)
    if amount >= 0:
        shaded = rgb + (1.0 - rgb) * amount
    else:
        shaded = rgb * (1.0 + amount)
    return mpl.colors.to_hex(np.clip(shaded, 0.0, 1.0))


def build_metric_colors() -> dict[str, str]:
    colors: dict[str, str] = {}
    for source_image, base_color in SOURCE_IMAGE_COLORS.items():
        source_specs = [
            spec
            for spec in SELECTED_METRICS
            if source_image_from_family(spec.family) == source_image
        ]
        if len(source_specs) == 1:
            shade_amounts = [0.0]
        else:
            shade_amounts = np.linspace(0.42, -0.28, len(source_specs))
        for spec, amount in zip(source_specs, shade_amounts):
            colors[spec.label] = shade_color(base_color, float(amount))
    return colors


METRIC_COLORS = build_metric_colors()


def norm_token(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def build_alias_map() -> dict[str, MetricSpec]:
    aliases: dict[str, MetricSpec] = {}
    for spec in SELECTED_METRICS:
        for alias in (spec.key, spec.label, *spec.aliases):
            aliases[norm_token(alias)] = spec
        if spec.label.startswith("QSM-X-R2p"):
            aliases[norm_token(spec.label.replace("R2p", "R2"))] = spec
            aliases[norm_token(spec.label.replace("R2p", "R2'"))] = spec
    return aliases


ALIAS_MAP = build_alias_map()
SPEC_BY_KEY = {spec.key: spec for spec in SELECTED_METRICS}
SPEC_BY_LABEL = {spec.label: spec for spec in SELECTED_METRICS}


def metric_spec(metric: object) -> MetricSpec | None:
    text = str(metric).strip()
    candidates = [text]
    if "__" in text:
        candidates.append(text.split("__")[-1])
    for candidate in candidates:
        spec = ALIAS_MAP.get(norm_token(candidate))
        if spec is not None:
            return spec
    return None


def metric_family(metric: object) -> str:
    label = str(metric).strip()
    selected = metric_spec(label)
    if selected is not None:
        return selected.family
    if label.startswith("DKI-"):
        return "DKI"
    if label.startswith("NODDI-"):
        return "NODDI"
    if label.startswith("MAPMRI-"):
        return "MAPMRI"
    if label.startswith("DSIStudio-GQI-"):
        return "GQI"
    if label.startswith(("DSIStudio-Tensor-", "TORTOISE-")):
        return "Tensor"
    if label.startswith("QSM-"):
        return "QSM"
    if "MyelinW" in label:
        return "T1w/T2w"
    if label.startswith("G-"):
        return "g-ratio"
    if label.startswith("R1"):
        return "R1"
    if label.startswith("ihMT"):
        return "ihMTR"
    if label in {"MTR", "MTsat"}:
        return "MT"
    if label == "MEGRE":
        return "MEGRE"
    return "Other"


def metric_source_image(metric: object) -> str:
    return source_image_from_family(metric_family(metric))


metric_table = pd.DataFrame(
    [{"Metric": spec.label, "Family": spec.family} for spec in SELECTED_METRICS]
)
display(metric_table)

# %% [markdown] ## Load the regional median profiles

# %% Notebook code cell 7
WM_PATH_RE = re.compile(r"sub-(?P<subject>[^/_]+).*?(?P<session>ses-[^/_]+)")
RUN_TOKEN_RE = re.compile(r"(?:^|[/_])run-(?P<run>[^_/\\.]+)", re.IGNORECASE)
GM_FILE_RE = re.compile(
    r"sub-(?P<subject>[^_]+)_(?P<session>ses-[^_]+)_(?P<run>run-[^_]+)_"
)
EXCLUDED_BUNDLE_TOKENS = (
    "anteriorcommissure",
    "dentatorubrothalamictractlr",
    "dentatorubrothalamictractrl",
)
EXCLUDED_GM_METRICS = {"G-ihMTsat", "G-ihMTR"}


def normalize_subject(value: object) -> str:
    return re.sub(r"^sub-", "", str(value).strip())


def is_pilot_subject(value: object) -> bool:
    return normalize_subject(value).upper().startswith("PILOT")


def session_label(value: object) -> str:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"Could not parse session number from: {value}")
    return f"Session {int(match.group(1)):02d}"


def load_qc_table(path: Path = QC_FILE) -> pd.DataFrame:
    if not path.exists():
        alt = PROJECT_ROOT / "data" / "manual_qc_modality.tsv"
        path = alt if alt.exists() else path
    qc = pd.read_csv(path, sep="\t")
    qc = qc.copy()
    qc["participant_id"] = qc["participant_id"].map(normalize_subject)
    qc = qc.loc[~qc["participant_id"].map(is_pilot_subject)].copy()
    return qc.set_index("participant_id", drop=False)


def required_modalities(metric: object) -> tuple[str, ...]:
    label = str(metric).strip()
    if label in {"FA", "MD", "RD", "ICVF", "MKT", "RK", "RTOP", "RTAP", "NG", "GFA"}:
        return ("dMRI",)
    if label.startswith(("DKI-", "DSIStudio-", "MAPMRI-", "NODDI-", "TORTOISE-")):
        return ("dMRI",)
    if label == "QSM-SEPIA-E5" or label == "MEGRE":
        return ("MEGRE",)
    if label.startswith("QSM-X-R2"):
        return ("MEGRE", "MESE")
    if label in {"ihMTw", "ihMTR", "MTR"}:
        return ("ihMTRAGE",)
    if label in {"ihMTsat", "ihMTsat-B1c"}:
        return ("MP2RAGE", "ihMTRAGE", "B1+")
    if label == "R1":
        return ("MP2RAGE",)
    if label == "R1-B1c":
        return ("MP2RAGE", "B1+")
    if label in {"MPRAGE-MyelinW", "Scaled MPRAGE-MyelinW"}:
        return ("MPRAGE T1w", "SPACE T2w")
    if label in {"SPACE-MyelinW", "Scaled SPACE-MyelinW"}:
        return ("SPACE T1w", "SPACE T2w")
    if label == "G-ihMTR":
        return ("dMRI", "ihMTRAGE")
    if label == "G-ihMTsat":
        return ("MP2RAGE", "dMRI", "ihMTRAGE", "B1+")
    raise ValueError(f"No QC modality mapping defined for metric: {label}")


def qc_passes(qc: pd.DataFrame, subject: object, session: object, modalities: tuple[str, ...]) -> bool:
    subject_id = normalize_subject(subject)
    if subject_id not in qc.index:
        return False
    row = qc.loc[subject_id]
    prefix = session_label(session)
    for modality in modalities:
        column = f"{prefix}--{modality}"
        if column not in qc.columns:
            raise RuntimeError(f"QC file is missing required column: {column}")
        value = row[column]
        if pd.isna(value) or int(value) != 1:
            return False
    return True


def apply_profile_qc(profiles: pd.DataFrame, qc: pd.DataFrame, qc_mode: str = QC_MODE) -> pd.DataFrame:
    data = profiles.loc[~profiles["subject"].map(is_pilot_subject)].copy()
    if qc_mode == "metricqc":
        keep = [
            qc_passes(qc, row.subject, row.session, required_modalities(row.metric_label))
            for row in data.itertuples(index=False)
        ]
        return data.loc[keep].copy()
    if qc_mode == "completeqc":
        modalities = sorted(
            {
                modality
                for metric in data["metric_label"].dropna().astype(str).unique()
                for modality in required_modalities(metric)
            }
        )
        subjects = sorted({normalize_subject(value) for value in data["subject"].unique()})
        complete_subjects = {
            subject
            for subject in subjects
            if all(qc_passes(qc, subject, f"ses-{session:02d}", tuple(modalities)) for session in (1, 2))
        }
        return data.loc[data["subject"].map(normalize_subject).isin(complete_subjects)].copy()
    raise ValueError(f"Unsupported QC_MODE: {qc_mode}")


manual_qc = load_qc_table()


def normalize_bundle_id(bundle: object) -> str:
    return norm_token(bundle)


def extract_param_token(*texts: object) -> str:
    for text in texts:
        match = re.search(r"_param-([^_./]+)", str(text).lower())
        if match:
            return norm_token(match.group(1))
    return ""


def infer_selected_wm_metric(row: pd.Series, source_tsv: str) -> MetricSpec | None:
    variable_name = str(row.get("variable_name", "")).strip()
    source_file = str(row.get("source_file", ""))
    suffix = str(row.get("qsirecon_suffix", ""))
    source = " ".join([source_tsv, source_file, suffix]).lower()
    variable = norm_token(variable_name)
    param = extract_param_token(source_file, source_tsv)

    # The generated myelin scalarstats use canonical variable names. DWI
    # scalarstats do not: short names such as FA and ICVF must be resolved
    # from their reconstruction source to avoid mixing model families.
    if "/bundle_myelin_stats/" in source_tsv.lower():
        return metric_spec(variable_name)

    if "noddi" in source:
        is_icvf = param == "icvf" or "icvf" in variable
        is_modulated = "modulated" in source or "modulated" in variable
        if is_icvf and not is_modulated:
            return SPEC_BY_KEY["noddi_icvf"]
        return None

    if ("dipydki" in source or "dki" in suffix.lower()) and not any(
        token in source for token in ("dkimicro", "msdki")
    ):
        if param == "mkt" or variable.endswith("mkt"):
            return SPEC_BY_KEY["dki_mkt"]
        if param == "rk" or variable.endswith("rk"):
            return SPEC_BY_KEY["dki_rk"]

    if "dsistudio" in source or "gqi" in source:
        if param == "gfa" or variable.endswith("gfa"):
            return SPEC_BY_KEY["gqi_gfa"]

    if "model-mapmri" in source or "model_mapmri" in source or "mapmri" in suffix.lower():
        mapmri_lookup = {
            "rtop": "mapmri_rtop",
            "rtap": "mapmri_rtap",
            "ng": "mapmri_ng",
        }
        for token, key in mapmri_lookup.items():
            if param == token or variable.endswith(token):
                return SPEC_BY_KEY[key]

    is_inner_tensor = (
        "tortoise" in source
        and ("model-tensor" in source or "model_tensor" in source)
        and "model-mapmri" not in source
    )
    if is_inner_tensor:
        tensor_lookup = {
            "fa": "tortoise_fa",
            "md": "tortoise_md",
            "rd": "tortoise_rd",
        }
        for token, key in tensor_lookup.items():
            if param == token or variable == token or variable.endswith(token):
                return SPEC_BY_KEY[key]
    return None


def normalize_run_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    token = re.sub(r"^run-", "", str(value).strip(), flags=re.IGNORECASE)
    if not token or token.lower() in {"nan", "none", "<na>"}:
        return None
    if re.fullmatch(r"[0-9]+(?:\\.0+)?", token):
        token = str(int(float(token))).zfill(2)
    return f"run-{token}"


def parse_wm_profile(file_path: str, df: pd.DataFrame) -> tuple[str, str, str]:
    match = WM_PATH_RE.search(file_path)
    parsed_subject = match.group("subject") if match else None
    parsed_session = match.group("session") if match else None

    subject = parsed_subject
    if "subject_id" in df and df["subject_id"].notna().any():
        subject = str(df.loc[df["subject_id"].notna(), "subject_id"].iloc[0])
        subject = re.sub(r"^sub-", "", subject)

    session = parsed_session
    if "session_id" in df and df["session_id"].notna().any():
        session = str(df.loc[df["session_id"].notna(), "session_id"].iloc[0])

    run_candidates: set[str] = set()
    for column in ("run_id", "run"):
        if column in df:
            run_candidates.update(
                run
                for run in (normalize_run_id(value) for value in df[column].dropna().unique())
                if run is not None
            )

    if not run_candidates:
        source_texts = [file_path]
        if "source_file" in df:
            source_texts.extend(df["source_file"].dropna().astype(str).unique())
        for text in source_texts:
            run_match = RUN_TOKEN_RE.search(str(text))
            if run_match:
                run = normalize_run_id(run_match.group("run"))
                if run is not None:
                    run_candidates.add(run)

    if not subject or not session:
        raise ValueError(f"Could not determine subject/session for {file_path}")
    if len(run_candidates) != 1:
        raise ValueError(
            f"Expected one run in {file_path}, found {sorted(run_candidates)}"
        )
    return subject, session, next(iter(run_candidates))


def collect_selected_wm_profiles(input_globs: list[str]) -> pd.DataFrame:
    files = sorted({path for pattern in input_globs for path in glob(pattern)})
    if not files:
        raise FileNotFoundError(f"No WM scalarstats matched: {input_globs}")

    records: list[dict[str, object]] = []
    required = {"bundle", "variable_name", "masked_median"}
    for file_path in files:
        df = pd.read_csv(file_path, sep="\t")
        missing = required.difference(df.columns)
        if missing:
            raise RuntimeError(f"{file_path} is missing columns: {sorted(missing)}")
        subject, session, run = parse_wm_profile(file_path, df)
        if is_pilot_subject(subject):
            continue
        for _, row in df.iterrows():
            spec = infer_selected_wm_metric(row, file_path)
            if spec is None:
                continue
            bundle = normalize_bundle_id(row["bundle"])
            if any(token in bundle for token in EXCLUDED_BUNDLE_TOKENS):
                continue
            value = pd.to_numeric(row["masked_median"], errors="coerce")
            if not np.isfinite(value):
                continue
            records.append(
                {
                    "subject": subject,
                    "session": session,
                    "run": run,
                    "region": bundle,
                    "metric_key": spec.key,
                    "metric_label": spec.label,
                    "family": spec.family,
                    "value": float(value),
                }
            )
    if not records:
        raise RuntimeError("No selected WM masked_median values were found.")
    return (
        pd.DataFrame(records)
        .groupby(
            [
                "subject",
                "session",
                "run",
                "region",
                "metric_key",
                "metric_label",
                "family",
            ],
            as_index=False,
        )["value"]
        .median()
    )


def collect_selected_gm_profiles(input_glob: str) -> pd.DataFrame:
    files = sorted(glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No GM scalarstats matched: {input_glob}")

    records: list[pd.DataFrame] = []
    base_required = {"parcel_name", "parcel_hemi"}
    for file_path in files:
        match = GM_FILE_RE.search(Path(file_path).name)
        if match is None:
            raise ValueError(f"Could not parse subject/session/run from {file_path}")
        if is_pilot_subject(match.group("subject")):
            continue
        df = pd.read_csv(file_path)
        missing = base_required.difference(df.columns)
        if missing:
            raise RuntimeError(f"{file_path} is missing columns: {sorted(missing)}")

        # Exact GM statistic: only columns ending in "_median" are eligible.
        for column in [col for col in df.columns if col.endswith("_median")]:
            spec = metric_spec(column[: -len("_median")])
            if spec is not None and spec.label in EXCLUDED_GM_METRICS:
                continue
            if spec is None:
                continue
            part = pd.DataFrame(
                {
                    "subject": match.group("subject"),
                    "session": match.group("session"),
                    "run": match.group("run"),
                    "region": (
                        df["parcel_hemi"].astype(str)
                        + "_"
                        + df["parcel_name"].astype(str)
                    ),
                    "metric_key": spec.key,
                    "metric_label": spec.label,
                    "family": spec.family,
                    "value": pd.to_numeric(df[column], errors="coerce"),
                }
            )
            records.append(part)
    if not records:
        raise RuntimeError("No selected GM median values were found.")
    out = pd.concat(records, ignore_index=True)
    out = out[np.isfinite(out["value"].to_numpy(dtype=float))].copy()
    return (
        out.groupby(
            [
                "subject",
                "session",
                "run",
                "region",
                "metric_key",
                "metric_label",
                "family",
            ],
            as_index=False,
        )["value"]
        .median()
    )


def load_or_build_profile_cache(
    cache_path: Path,
    builder,
    *builder_args,
) -> pd.DataFrame:
    if cache_path.exists() and not REBUILD_RAW_CACHE:
        print(f"Reading cache: {cache_path}")
        return pd.read_csv(cache_path)
    data = builder(*builder_args)
    data.to_csv(cache_path, index=False, compression="gzip")
    print(f"Wrote cache: {cache_path}")
    return data

# %% Notebook code cell 8
wm_profiles = load_or_build_profile_cache(
    CACHE_DIR / "selected_scaled_t1t2_wm_masked_median_run_profiles.csv.gz",
    collect_selected_wm_profiles,
    WM_INPUT_GLOBS,
)
gm_profiles = load_or_build_profile_cache(
    CACHE_DIR / "selected_scaled_t1t2_gm_median_profiles.csv.gz",
    collect_selected_gm_profiles,
    GM_INPUT_GLOB,
)

wm_profiles = apply_profile_qc(wm_profiles, manual_qc, QC_MODE)
gm_profiles = apply_profile_qc(gm_profiles, manual_qc, QC_MODE)

selected_labels = {spec.label for spec in SELECTED_METRICS}
missing_wm_profiles = sorted(selected_labels - set(wm_profiles["metric_label"]))
missing_gm_profiles = sorted(selected_labels - set(gm_profiles["metric_label"]))
if missing_wm_profiles:
    warnings.warn(f"Selected metrics absent from WM profiles: {', '.join(missing_wm_profiles)}")
if missing_gm_profiles:
    warnings.warn(f"Selected metrics absent from GM profiles: {', '.join(missing_gm_profiles)}")


def profile_coverage(
    df: pd.DataFrame, profile_columns: list[str], tissue: str
) -> pd.DataFrame:
    region_counts = (
        df.groupby(["metric_label", "family"])["region"]
        .nunique()
        .rename("n_regions")
    )
    profile_counts = (
        df[profile_columns + ["metric_label", "family"]]
        .drop_duplicates()
        .groupby(["metric_label", "family"])
        .size()
        .rename("n_profiles")
    )
    return (
        pd.concat([profile_counts, region_counts], axis=1)
        .reset_index()
        .assign(tissue=tissue)
        .sort_values(["family", "metric_label"])
    )


coverage = pd.concat(
    [
        profile_coverage(wm_profiles, ["subject", "session", "run"], "WM"),
        profile_coverage(gm_profiles, ["subject", "session", "run"], "GM"),
    ],
    ignore_index=True,
)
display(coverage)

# %% [markdown] ## Figures 1 and 2: regional-profile Spearman matrices

# %% Notebook code cell 10
def mean_profile_spearman(
    long_df: pd.DataFrame,
    profile_columns: list[str],
    min_common_regions: int = MIN_COMMON_REGIONS,
    metric_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = long_df.copy()
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean[np.isfinite(clean["value"].to_numpy(dtype=float))]

    wide = clean.pivot_table(
        index=profile_columns + ["region"],
        columns="metric_label",
        values="value",
        aggfunc="median",
    )
    if metric_order is None:
        selected_order = [
            spec.label for spec in SELECTED_METRICS if spec.label in wide.columns
        ]
    else:
        selected_order = [label for label in metric_order if label in wide.columns]
    wide = wide.reindex(columns=selected_order)

    corr = pd.DataFrame(np.nan, index=selected_order, columns=selected_order)
    counts = pd.DataFrame(0, index=selected_order, columns=selected_order, dtype=int)

    for metric in selected_order:
        region_counts = wide[metric].notna().groupby(level=profile_columns).sum()
        corr.loc[metric, metric] = 1.0
        counts.loc[metric, metric] = int((region_counts >= min_common_regions).sum())

    for i, metric_a in enumerate(selected_order):
        for metric_b in selected_order[i + 1 :]:
            profile_rhos: list[float] = []
            pair = wide[[metric_a, metric_b]]
            for _, profile in pair.groupby(level=profile_columns, sort=False):
                xy = profile.dropna()
                if len(xy) < min_common_regions:
                    continue
                if xy[metric_a].nunique() < 2 or xy[metric_b].nunique() < 2:
                    continue
                result = spearmanr(xy[metric_a], xy[metric_b])
                # SciPy <1.7 exposes .correlation; newer versions expose .statistic.
                rho_value = result.statistic if hasattr(result, "statistic") else result.correlation
                rho = float(rho_value)
                if np.isfinite(rho):
                    profile_rhos.append(rho)

            if profile_rhos:
                z_values = np.arctanh(np.clip(profile_rhos, -0.999999, 0.999999))
                mean_rho = float(np.tanh(np.mean(z_values)))
                corr.loc[metric_a, metric_b] = mean_rho
                corr.loc[metric_b, metric_a] = mean_rho
                counts.loc[metric_a, metric_b] = len(profile_rhos)
                counts.loc[metric_b, metric_a] = len(profile_rhos)
    return corr, counts


def correlation_linkage(corr: pd.DataFrame) -> np.ndarray | None:
    if len(corr) < 2:
        return None
    safe = corr.fillna(0.0).to_numpy(dtype=float)
    distance = np.clip(1.0 - np.abs(safe), 0.0, 1.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return linkage(
        squareform(distance, checks=False),
        method="average",
        optimal_ordering=True,
    )


def plot_correlation_matrix(
    corr: pd.DataFrame,
    tissue_title: str,
    output_stem: str,
    metric_families: dict[str, str] | None = None,
    family_colors: dict[str, str] | None = None,
    legend_title: str = "Family",
    figure_size: tuple[float, float] = (11.5, 11.0),
    label_fontsize: float = 8,
):
    z_matrix = correlation_linkage(corr)
    plot_data = corr.copy()
    np.fill_diagonal(plot_data.values, np.nan)

    if metric_families is None:
        metric_families = {
            label: SPEC_BY_LABEL[label].family for label in plot_data.index
        }
    if family_colors is None:
        family_colors = FAMILY_COLORS
    family_by_metric = pd.Series(
        {
            label: family_colors[metric_families[label]]
            for label in plot_data.index
        }
    )
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    cmap.set_bad("#e6e6e6")

    grid = sns.clustermap(
        plot_data,
        row_linkage=z_matrix,
        col_linkage=z_matrix,
        row_cluster=z_matrix is not None,
        col_cluster=z_matrix is not None,
        row_colors=family_by_metric,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0,
        figsize=figure_size,
        dendrogram_ratio=(0.12, 0.025),
        colors_ratio=0.025,
        cbar_pos=(0.27, 0.055, 0.46, 0.022),
        cbar_kws={
            "orientation": "horizontal",
            "label": r"Mean Spearman $\rho$",
            "ticks": [-1, -0.5, 0, 0.5, 1],
        },
    )
    grid.ax_col_dendrogram.set_visible(False)
    grid.ax_heatmap.set_aspect("equal", adjustable="box")
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.ax_heatmap.tick_params(axis="both", length=0)
    plt.setp(
        grid.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=label_fontsize,
    )
    plt.setp(
        grid.ax_heatmap.get_yticklabels(), rotation=0, fontsize=label_fontsize
    )

    # Overlay a black square for each self-correlation after clustering.
    row_order = list(grid.data2d.index)
    column_order = list(grid.data2d.columns)
    column_position = {label: index for index, label in enumerate(column_order)}
    for row_index, label in enumerate(row_order):
        col_index = column_position[label]
        grid.ax_heatmap.add_patch(
            Rectangle(
                (col_index, row_index),
                1,
                1,
                facecolor="black",
                edgecolor="black",
                linewidth=0,
                zorder=5,
            )
        )

    handles = [
        Patch(facecolor=color, edgecolor="none", label=family)
        for family, color in family_colors.items()
        if family in {metric_families[label] for label in plot_data.index}
    ]
    grid.ax_heatmap.legend(
        handles=handles,
        title=legend_title,
        loc="upper left",
        bbox_to_anchor=(1.25, 0.55),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    grid.fig.suptitle(tissue_title, fontsize=18, y=0.97)
    grid.fig.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.16)
    grid.cax.set_position([0.27, 0.055, 0.46, 0.022])

    # Keep the family strip and row dendrogram exactly aligned with the
    # heatmap after labels and the color bar have changed the layout.
    grid.fig.canvas.draw()
    heatmap_position = grid.ax_heatmap.get_position()
    color_position = grid.ax_row_colors.get_position()
    grid.ax_row_colors.set_position(
        [color_position.x0, heatmap_position.y0, color_position.width, heatmap_position.height]
    )
    grid.ax_row_colors.set_ylim(grid.ax_heatmap.get_ylim())
    dendrogram_position = grid.ax_row_dendrogram.get_position()
    grid.ax_row_dendrogram.set_position(
        [
            dendrogram_position.x0,
            heatmap_position.y0,
            dendrogram_position.width,
            heatmap_position.height,
        ]
    )

    for extension in ("png", "pdf"):
        out_path = OUTPUT_DIR / f"{output_stem}.{extension}"
        grid.fig.savefig(out_path, bbox_inches="tight")
        print(f"Wrote: {out_path}")
    return grid

# %% Notebook code cell 11
wm_spearman, wm_spearman_n = mean_profile_spearman(
    wm_profiles,
    profile_columns=["subject", "session", "run"],
)
wm_spearman.to_csv(OUTPUT_DIR / "selected_metric_wm_spearman.csv")
wm_spearman_n.to_csv(OUTPUT_DIR / "selected_metric_wm_spearman_n_profiles.csv")
wm_corr_grid = plot_correlation_matrix(
    wm_spearman,
    tissue_title="White Matter",
    output_stem="selected_metric_wm_spearman",
    metric_families={label: metric_source_image(label) for label in wm_spearman.index},
    family_colors=SOURCE_IMAGE_COLORS,
    legend_title="Source image",
)
plt.show()

# %% Notebook code cell 12
gm_spearman, gm_spearman_n = mean_profile_spearman(
    gm_profiles,
    profile_columns=["subject", "session", "run"],
)
gm_spearman.to_csv(OUTPUT_DIR / "selected_metric_gm_spearman.csv")
gm_spearman_n.to_csv(OUTPUT_DIR / "selected_metric_gm_spearman_n_profiles.csv")
gm_corr_grid = plot_correlation_matrix(
    gm_spearman,
    tissue_title="Gray Matter",
    output_stem="selected_metric_gm_spearman",
    metric_families={label: metric_source_image(label) for label in gm_spearman.index},
    family_colors=SOURCE_IMAGE_COLORS,
    legend_title="Source image",
)
plt.show()

# %% [markdown] ## Supplementary correlation matrices: all available metrics

# %% Notebook code cell 14
def load_full_wm_metric_inference():
    candidates: list[Path] = []
    configured_analysis_dir = os.environ.get("NIBS_ANALYSIS_DIR")
    if configured_analysis_dir:
        candidates.append(
            Path(configured_analysis_dir) / "compute_icc_from_bundle_stats.py"
        )
    candidates.extend(
        [
            Path.cwd() / "analysis" / "compute_icc_from_bundle_stats.py",
            Path.cwd() / "compute_icc_from_bundle_stats.py",
            PROJECT_ROOT / "code" / "analysis" / "compute_icc_from_bundle_stats.py",
        ]
    )
    module_path = next((path for path in candidates if path.exists()), None)
    if module_path is None:
        searched = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            "Could not locate compute_icc_from_bundle_stats.py. Searched:\n"
            f"{searched}\nSet NIBS_ANALYSIS_DIR to the analysis directory."
        )

    module_spec = importlib.util.spec_from_file_location(
        "_nibs_full_wm_metric_mapping", module_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load metric mapping from {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    previous_backend = mpl.get_backend()
    module_spec.loader.exec_module(module)
    if mpl.get_backend() != previous_backend:
        plt.switch_backend(previous_backend)
    return module._infer_metric_name


infer_all_wm_metric = load_full_wm_metric_inference()


def collect_all_wm_profiles(input_globs: list[str]) -> pd.DataFrame:
    files = sorted({path for pattern in input_globs for path in glob(pattern)})
    if not files:
        raise FileNotFoundError(f"No WM scalarstats matched: {input_globs}")

    records: list[dict[str, object]] = []
    required = {"bundle", "variable_name", "masked_median"}
    for file_path in files:
        df = pd.read_csv(file_path, sep="\t")
        missing = required.difference(df.columns)
        if missing:
            raise RuntimeError(f"{file_path} is missing columns: {sorted(missing)}")
        subject, session, run = parse_wm_profile(file_path, df)
        if is_pilot_subject(subject):
            continue
        for _, row in df.iterrows():
            metric_label = infer_all_wm_metric(row, file_path)
            if metric_label is None:
                continue
            bundle = normalize_bundle_id(row["bundle"])
            if any(token in bundle for token in EXCLUDED_BUNDLE_TOKENS):
                continue
            value = pd.to_numeric(row["masked_median"], errors="coerce")
            if not np.isfinite(value):
                continue
            records.append(
                {
                    "subject": subject,
                    "session": session,
                    "run": run,
                    "region": bundle,
                    "metric_key": norm_token(metric_label),
                    "metric_label": metric_label,
                    "family": metric_family(metric_label),
                    "value": float(value),
                }
            )
    if not records:
        raise RuntimeError("No recognized WM masked_median values were found.")
    return (
        pd.DataFrame(records)
        .groupby(
            [
                "subject",
                "session",
                "run",
                "region",
                "metric_key",
                "metric_label",
                "family",
            ],
            as_index=False,
        )["value"]
        .median()
    )


def collect_all_gm_profiles(input_glob: str) -> pd.DataFrame:
    files = sorted(glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No GM scalarstats matched: {input_glob}")

    records: list[pd.DataFrame] = []
    for file_path in files:
        match = GM_FILE_RE.search(Path(file_path).name)
        if match is None:
            raise ValueError(f"Could not parse subject/session/run from {file_path}")
        if is_pilot_subject(match.group("subject")):
            continue
        df = pd.read_csv(file_path)
        required = {"parcel_name", "parcel_hemi"}
        missing = required.difference(df.columns)
        if missing:
            raise RuntimeError(f"{file_path} is missing columns: {sorted(missing)}")
        region = df["parcel_hemi"].astype(str) + "_" + df["parcel_name"].astype(str)
        for column in [name for name in df.columns if name.endswith("_median")]:
            metric_label = column[: -len("_median")]
            if metric_label in EXCLUDED_GM_METRICS:
                continue
            records.append(
                pd.DataFrame(
                    {
                        "subject": match.group("subject"),
                        "session": match.group("session"),
                        "run": match.group("run"),
                        "region": region,
                        "metric_key": norm_token(metric_label),
                        "metric_label": metric_label,
                        "family": metric_family(metric_label),
                        "value": pd.to_numeric(df[column], errors="coerce"),
                    }
                )
            )
    if not records:
        raise RuntimeError("No GM median metrics were found.")
    out = pd.concat(records, ignore_index=True)
    out = out[np.isfinite(out["value"].to_numpy(dtype=float))].copy()
    return (
        out.groupby(
            [
                "subject",
                "session",
                "run",
                "region",
                "metric_key",
                "metric_label",
                "family",
            ],
            as_index=False,
        )["value"]
        .median()
    )


def discovered_metric_order(profiles: pd.DataFrame) -> list[str]:
    family_order = {family: index for index, family in enumerate(ALL_FAMILY_COLORS)}
    metric_rows = profiles[["metric_label", "family"]].drop_duplicates()
    return sorted(
        metric_rows["metric_label"],
        key=lambda label: (
            family_order.get(metric_family(label), len(family_order)),
            label.lower(),
        ),
    )


all_wm_profiles = load_or_build_profile_cache(
    CACHE_DIR / "all_wm_masked_median_run_profiles.csv.gz",
    collect_all_wm_profiles,
    WM_INPUT_GLOBS,
)
all_gm_profiles = load_or_build_profile_cache(
    CACHE_DIR / "all_gm_median_run_profiles.csv.gz",
    collect_all_gm_profiles,
    GM_INPUT_GLOB,
)

all_wm_profiles = apply_profile_qc(all_wm_profiles, manual_qc, QC_MODE)
all_gm_profiles = apply_profile_qc(all_gm_profiles, manual_qc, QC_MODE)

all_metric_coverage = pd.concat(
    [
        profile_coverage(all_wm_profiles, ["subject", "session", "run"], "WM"),
        profile_coverage(all_gm_profiles, ["subject", "session", "run"], "GM"),
    ],
    ignore_index=True,
)
display(all_metric_coverage)

# %% Notebook code cell 15
all_wm_order = discovered_metric_order(all_wm_profiles)
all_wm_sources = {
    label: metric_source_image(label) for label in all_wm_order
}
all_wm_spearman, all_wm_spearman_n = mean_profile_spearman(
    all_wm_profiles,
    profile_columns=["subject", "session", "run"],
    metric_order=all_wm_order,
)
all_wm_spearman.to_csv(OUTPUT_DIR / "all_metric_wm_spearman.csv")
all_wm_spearman_n.to_csv(OUTPUT_DIR / "all_metric_wm_spearman_n_profiles.csv")
all_wm_corr_grid = plot_correlation_matrix(
    all_wm_spearman,
    tissue_title="White Matter",
    output_stem="all_metric_wm_spearman",
    metric_families=all_wm_sources,
    family_colors=SOURCE_IMAGE_COLORS,
    legend_title="Source image",
    figure_size=(20, 20),
    label_fontsize=5,
)
plt.show()

# %% Notebook code cell 16
all_gm_order = discovered_metric_order(all_gm_profiles)
all_gm_sources = {
    label: metric_source_image(label) for label in all_gm_order
}
all_gm_spearman, all_gm_spearman_n = mean_profile_spearman(
    all_gm_profiles,
    profile_columns=["subject", "session", "run"],
    metric_order=all_gm_order,
)
all_gm_spearman.to_csv(OUTPUT_DIR / "all_metric_gm_spearman.csv")
all_gm_spearman_n.to_csv(OUTPUT_DIR / "all_metric_gm_spearman_n_profiles.csv")
all_gm_corr_grid = plot_correlation_matrix(
    all_gm_spearman,
    tissue_title="Gray Matter",
    output_stem="all_metric_gm_spearman",
    metric_families=all_gm_sources,
    family_colors=SOURCE_IMAGE_COLORS,
    legend_title="Source image",
    figure_size=(20, 20),
    label_fontsize=5,
)
plt.show()

# %% [markdown] ## Figure 3: ICC distributions and GM-vs-WM comparison

# %% Notebook code cell 18
def read_selected_icc(
    path: Path,
    region_column: str,
    expected_stat: str | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required ICC file: {path}\n"
            "This notebook does not substitute a mean-based ICC file."
        )
    df = pd.read_csv(path)
    required = {"metric", region_column, "ICC2_1"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    if expected_stat is not None and "stat" in df.columns:
        observed = set(df["stat"].dropna().astype(str))
        if observed != {expected_stat}:
            raise RuntimeError(
                f"{path} contains stat={sorted(observed)}, expected only {expected_stat!r}."
            )

    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        spec = metric_spec(row["metric"])
        value = pd.to_numeric(row["ICC2_1"], errors="coerce")
        if spec is None or not np.isfinite(value):
            continue
        records.append(
            {
                "metric_key": spec.key,
                "metric_label": spec.label,
                "family": spec.family,
                region_column: row[region_column],
                "ICC2_1": float(value),
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError(f"No selected metrics were found in {path}")
    return out.drop_duplicates(["metric_key", region_column], keep="first")


wm_icc = read_selected_icc(
    WM_ICC_CSV,
    region_column="bundle",
    expected_stat="masked_median",
)
gm_icc = read_selected_icc(
    GM_ICC_CSV,
    region_column="parcel",
)

missing_wm_icc = sorted(
    {spec.label for spec in SELECTED_METRICS} - set(wm_icc["metric_label"])
)
missing_gm_icc = sorted(
    {spec.label for spec in SELECTED_METRICS} - set(gm_icc["metric_label"])
)
if missing_wm_icc:
    warnings.warn(f"Selected metrics absent from WM ICC: {', '.join(missing_wm_icc)}")
if missing_gm_icc:
    warnings.warn(f"Selected metrics absent from GM ICC: {', '.join(missing_gm_icc)}")

display(
    pd.concat(
        [
            wm_icc.groupby(["metric_label", "family"])["ICC2_1"]
            .agg(["mean", "median", "count"])
            .assign(tissue="WM"),
            gm_icc.groupby(["metric_label", "family"])["ICC2_1"]
            .agg(["mean", "median", "count"])
            .assign(tissue="GM"),
        ]
    ).reset_index()
)

# %% Notebook code cell 19
def plot_horizontal_violins(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_limits: tuple[float, float],
    x_label: str,
) -> list[str]:
    order = (
        data.groupby("metric_label")["ICC2_1"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    positions = np.arange(len(order))
    for position, label in zip(positions, order):
        values = (
            data.loc[data["metric_label"] == label, "ICC2_1"]
            .dropna()
            .to_numpy(dtype=float)
        )
        color = METRIC_COLORS[label]
        if len(values) >= 2 and np.nanstd(values) > 0:
            parts = ax.violinplot(
                [values],
                positions=[position],
                vert=False,
                widths=0.76,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method="scott",
            )
            body = parts["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor("#222222")
            body.set_linewidth(0.55)
            body.set_alpha(0.88)
        else:
            ax.scatter(values, np.full(len(values), position), color=color, s=25)

        q1, median, q3 = np.nanpercentile(values, [25, 50, 75])
        ax.hlines(position, q1, q3, color="#222222", linewidth=2.0, zorder=4)
        ax.scatter(
            median,
            position,
            s=22,
            color="white",
            edgecolor="#222222",
            linewidth=0.5,
            zorder=5,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(*x_limits)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.axvline(0, color="#777777", linestyle=":", linewidth=0.8, zorder=0)
    ax.grid(False)
    return order


def axes_fraction_point(ax: plt.Axes, x: float, y: float) -> tuple[float, float]:
    display_xy = ax.transData.transform((x, y))
    axes_xy = ax.transAxes.inverted().transform(display_xy)
    return float(axes_xy[0]), float(axes_xy[1])


def label_box(x: float, y: float, label: str) -> tuple[float, float, float, float]:
    width = min(0.23, max(0.048, 0.0072 * len(label)))
    height = 0.031
    return (x, x + width, y - height * 0.45, y + height * 0.55)


def boxes_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return (
        first[0] < second[1]
        and first[1] > second[0]
        and first[2] < second[3]
        and first[3] > second[2]
    )


def clamp_label(x: float, y: float, label: str) -> tuple[float, float]:
    width = label_box(x, y, label)[1] - x
    return min(max(x, 0.008), 0.992 - width), min(max(y, 0.025), 0.975)


def place_scatter_labels(ax: plt.Axes, comparison: pd.DataFrame) -> None:
    ax.figure.canvas.draw()
    offsets = [
        (0.014, 0.032),
        (0.014, -0.040),
        (-0.080, 0.032),
        (-0.080, -0.040),
        (-0.045, 0.070),
        (-0.045, -0.080),
        (0.035, 0.080),
        (0.035, -0.090),
        (-0.140, 0.080),
        (-0.140, -0.090),
        (-0.220, 0.115),
        (-0.220, 0.045),
        (-0.220, -0.040),
        (-0.220, -0.120),
        (-0.320, 0.145),
        (-0.320, 0.070),
        (-0.320, -0.010),
        (-0.320, -0.095),
        (-0.420, 0.170),
        (-0.420, 0.085),
        (-0.420, 0.000),
        (-0.420, -0.100),
    ]
    ordered = comparison.assign(
        density=comparison["gm_mean_icc"] + comparison["wm_mean_icc"]
    ).sort_values("density", ascending=False)

    labels: list[dict[str, object]] = []
    occupied: list[tuple[float, float, float, float]] = []
    for _, row in ordered.iterrows():
        point_axes = axes_fraction_point(
            ax,
            float(row["gm_mean_icc"]),
            float(row["wm_mean_icc"]),
        )
        label = str(row["metric_label"])
        best = clamp_label(point_axes[0] + 0.014, point_axes[1] + 0.032, label)
        best_score = np.inf
        for dx, dy in offsets:
            candidate = clamp_label(point_axes[0] + dx, point_axes[1] + dy, label)
            box = label_box(candidate[0], candidate[1], label)
            overlaps = sum(boxes_overlap(box, other) for other in occupied)
            distance = abs(candidate[0] - point_axes[0]) + abs(candidate[1] - point_axes[1])
            score = overlaps + 0.03 * distance
            if score < best_score:
                best = candidate
                best_score = score
            if overlaps == 0:
                break
        occupied.append(label_box(best[0], best[1], label))
        labels.append(
            {
                "label": label,
                "family": row["family"],
                "point_x": float(row["gm_mean_icc"]),
                "point_y": float(row["wm_mean_icc"]),
                "point_axes_x": point_axes[0],
                "point_axes_y": point_axes[1],
                "x": best[0],
                "y": best[1],
            }
        )

    # Resolve any residual collisions after candidate placement.
    for _ in range(240):
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                box_i = label_box(
                    float(labels[i]["x"]),
                    float(labels[i]["y"]),
                    str(labels[i]["label"]),
                )
                box_j = label_box(
                    float(labels[j]["x"]),
                    float(labels[j]["y"]),
                    str(labels[j]["label"]),
                )
                if not boxes_overlap(box_i, box_j):
                    continue
                push = min(box_i[3], box_j[3]) - max(box_i[2], box_j[2])
                push = push / 2 + 0.004
                if float(labels[i]["y"]) >= float(labels[j]["y"]):
                    labels[i]["y"] = float(labels[i]["y"]) + push
                    labels[j]["y"] = float(labels[j]["y"]) - push
                else:
                    labels[i]["y"] = float(labels[i]["y"]) - push
                    labels[j]["y"] = float(labels[j]["y"]) + push
                labels[i]["x"], labels[i]["y"] = clamp_label(
                    float(labels[i]["x"]),
                    float(labels[i]["y"]),
                    str(labels[i]["label"]),
                )
                labels[j]["x"], labels[j]["y"] = clamp_label(
                    float(labels[j]["x"]),
                    float(labels[j]["y"]),
                    str(labels[j]["label"]),
                )
                moved = True
        if not moved:
            break

    for item in labels:
        color = METRIC_COLORS[str(item["label"])]
        point_axes = (float(item["point_axes_x"]), float(item["point_axes_y"]))
        label_axes = (float(item["x"]), float(item["y"]))
        distance = abs(label_axes[0] - point_axes[0]) + abs(
            label_axes[1] - point_axes[1]
        )
        arrowprops = None
        if distance > 0.065:
            arrowprops = {
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.55,
                "alpha": 0.85,
                "shrinkA": 1,
                "shrinkB": 2,
            }
        ax.annotate(
            str(item["label"]),
            xy=(float(item["point_x"]), float(item["point_y"])),
            xycoords="data",
            xytext=label_axes,
            textcoords="axes fraction",
            fontsize=7,
            color=color,
            arrowprops=arrowprops,
            clip_on=False,
            zorder=5,
        )


def metric_icc_means(
    data: pd.DataFrame, value_name: str
) -> pd.DataFrame:
    return (
        data.groupby(["metric_key", "metric_label", "family"], as_index=False)[
            "ICC2_1"
        ]
        .mean()
        .rename(columns={"ICC2_1": value_name})
    )


def icc_axis_limits(data: pd.DataFrame) -> tuple[float, float]:
    """Return compact 0.05-rounded limits for one tissue's ICC distribution."""
    values = pd.to_numeric(data["ICC2_1"], errors="coerce").dropna().to_numpy()
    if not len(values):
        return (0.0, 1.0)
    span = max(float(values.max() - values.min()), 0.10)
    padding = max(0.03, 0.04 * span)
    lower = np.floor((float(values.min()) - padding) * 20.0) / 20.0
    upper = np.ceil((float(values.max()) + padding) * 20.0) / 20.0
    return (max(-1.0, lower), min(1.02, upper))


def plot_icc_figure(
    wm_data: pd.DataFrame,
    gm_data: pd.DataFrame,
) -> tuple[plt.Figure, pd.DataFrame]:
    wm_limits = icc_axis_limits(wm_data)
    gm_limits = icc_axis_limits(gm_data)

    fig = plt.figure(figsize=(14.0, 11.5), constrained_layout=False)
    layout = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.15, 0.82],
        hspace=0.28,
        wspace=0.12,
    )
    ax_wm = fig.add_subplot(layout[0, 0])
    ax_gm = fig.add_subplot(layout[0, 1])
    ax_scatter = fig.add_subplot(layout[1, :])

    plot_horizontal_violins(
        ax_wm,
        wm_data,
        x_limits=wm_limits,
        x_label="ICC(2,1) across WM bundles",
    )
    plot_horizontal_violins(
        ax_gm,
        gm_data,
        x_limits=gm_limits,
        x_label="ICC(2,1) across GM parcels",
    )
    ax_gm.yaxis.tick_right()
    ax_gm.tick_params(axis="y", labelleft=False, labelright=True, pad=4)

    wm_means = metric_icc_means(wm_data, "wm_mean_icc")
    gm_means = metric_icc_means(gm_data, "gm_mean_icc")
    comparison = pd.merge(
        gm_means,
        wm_means[["metric_key", "wm_mean_icc"]],
        on="metric_key",
        how="inner",
    ).dropna(subset=["gm_mean_icc", "wm_mean_icc"])

    scatter_values = comparison[["gm_mean_icc", "wm_mean_icc"]].to_numpy(dtype=float)
    if scatter_values.size and np.isfinite(scatter_values).any():
        scatter_min_data = float(np.nanmin(scatter_values))
        scatter_max_data = float(np.nanmax(scatter_values))
        scatter_data_span = max(scatter_max_data - scatter_min_data, 0.05)
        scatter_padding = max(0.015, 0.08 * scatter_data_span)
        scatter_lower = np.floor((scatter_min_data - scatter_padding) * 20.0) / 20.0
        scatter_upper = np.ceil((scatter_max_data + scatter_padding) * 20.0) / 20.0
        scatter_limits = (max(-1.0, scatter_lower), min(1.02, scatter_upper))
    else:
        scatter_limits = (0.0, 1.0)
    scatter_min, scatter_max = scatter_limits
    scatter_span = scatter_max - scatter_min
    ax_scatter.set_facecolor("white")
    ax_scatter.fill(
        [scatter_min, scatter_max, scatter_max],
        [scatter_min, scatter_min, scatter_max],
        color="#eeeeee",
        zorder=0,
    )
    ax_scatter.text(
        scatter_min + 0.72 * scatter_span,
        scatter_min + 0.13 * scatter_span,
        "GM ICC > WM ICC",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        fontstyle="italic",
        zorder=2,
    )
    ax_scatter.text(
        scatter_min + 0.23 * scatter_span,
        scatter_min + 0.82 * scatter_span,
        "WM ICC > GM ICC",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
        fontstyle="italic",
        zorder=2,
    )
    for _, row in comparison.iterrows():
        color = METRIC_COLORS[row["metric_label"]]
        ax_scatter.scatter(
            row["gm_mean_icc"],
            row["wm_mean_icc"],
            s=70,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
    ax_scatter.plot(
        scatter_limits,
        scatter_limits,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )
    ax_scatter.set_xlim(*scatter_limits)
    ax_scatter.set_ylim(*scatter_limits)
    ax_scatter.set_xlabel("Mean ICC across GM parcels")
    ax_scatter.set_ylabel("Mean ICC across WM bundles")
    ax_scatter.grid(False)
    place_scatter_labels(ax_scatter, comparison)

    used_sources = [
        source
        for source in SOURCE_IMAGE_COLORS
        if source in {source_image_from_family(family) for family in comparison["family"]}
    ]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=SOURCE_IMAGE_COLORS[source],
            markeredgecolor="black",
            markeredgewidth=0.45,
            label=source,
        )
        for source in used_sources
    ]
    ax_scatter.legend(
        handles=handles,
        title="Source image",
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        frameon=False,
    )

    for axis, label in ((ax_wm, "A"), (ax_gm, "B"), (ax_scatter, "C")):
        axis.text(
            -0.11,
            1.025,
            label,
            transform=axis.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
    fig.subplots_adjust(
        left=0.10,
        right=0.84,
        top=0.97,
        bottom=0.07,
        hspace=0.28,
        wspace=0.12,
    )
    return fig, comparison.sort_values("metric_label").reset_index(drop=True)

# %% Notebook code cell 20
icc_figure, icc_metric_means = plot_icc_figure(wm_icc, gm_icc)
for extension in ("png", "pdf"):
    out_path = OUTPUT_DIR / f"selected_metric_icc_violin_scatter.{extension}"
    icc_figure.savefig(out_path, bbox_inches="tight")
    print(f"Wrote: {out_path}")
icc_metric_means.to_csv(
    OUTPUT_DIR / "selected_metric_icc_violin_scatter_values.csv",
    index=False,
)
plt.show()

# %% [markdown] ## Figure 4: nearest-neighbor accuracy

# %% Notebook code cell 22
def read_selected_discriminability(path: Path, tissue_prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing discriminability summary: {path}")
    df = pd.read_csv(path)
    required = {
        "profile_group",
        "discriminability",
        "nearest_neighbor_accuracy",
        "n_profiles",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        if str(row["profile_group"]) == "ALL_METRICS":
            continue
        spec = metric_spec(row["profile_group"])
        discriminability = pd.to_numeric(row["discriminability"], errors="coerce")
        accuracy = pd.to_numeric(row["nearest_neighbor_accuracy"], errors="coerce")
        n_profiles = pd.to_numeric(row["n_profiles"], errors="coerce")
        if spec is None or not np.isfinite(discriminability) or not np.isfinite(accuracy):
            continue
        records.append(
            {
                "metric_key": spec.key,
                "metric_label": spec.label,
                "family": spec.family,
                f"{tissue_prefix}_discriminability": float(discriminability),
                f"{tissue_prefix}_accuracy": float(accuracy),
                f"{tissue_prefix}_n_profiles": n_profiles,
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError(f"No selected metrics were found in {path}")
    return out.drop_duplicates("metric_key", keep="first")


wm_nn = read_selected_discriminability(WM_NN_CSV, "wm")
gm_nn = read_selected_discriminability(GM_NN_CSV, "gm")
nn_values = pd.merge(
    gm_nn,
    wm_nn[
        [
            "metric_key",
            "metric_label",
            "family",
            "wm_discriminability",
            "wm_accuracy",
            "wm_n_profiles",
        ]
    ],
    on="metric_key",
    how="outer",
)

for column in ("metric_label", "family"):
    if f"{column}_x" in nn_values.columns and f"{column}_y" in nn_values.columns:
        nn_values[column] = nn_values[f"{column}_x"].combine_first(nn_values[f"{column}_y"])
        nn_values = nn_values.drop(columns=[f"{column}_x", f"{column}_y"])

nn_values["gm_correct"] = pd.Series(pd.NA, index=nn_values.index, dtype="Int64")
gm_has_values = nn_values[["gm_accuracy", "gm_n_profiles"]].notna().all(axis=1)
nn_values.loc[gm_has_values, "gm_correct"] = (
    nn_values.loc[gm_has_values, "gm_accuracy"]
    * nn_values.loc[gm_has_values, "gm_n_profiles"]
).round().astype("Int64")

nn_values["wm_correct"] = pd.Series(pd.NA, index=nn_values.index, dtype="Int64")
wm_has_values = nn_values[["wm_accuracy", "wm_n_profiles"]].notna().all(axis=1)
nn_values.loc[wm_has_values, "wm_correct"] = (
    nn_values.loc[wm_has_values, "wm_accuracy"]
    * nn_values.loc[wm_has_values, "wm_n_profiles"]
).round().astype("Int64")
display(nn_values.sort_values(["family", "metric_label"]))

# %% Notebook code cell 23
def plot_discriminability_bars(
    comparison: pd.DataFrame,
    score_suffix: str,
    x_label: str,
    annotate_counts: bool,
) -> plt.Figure:
    figure_height = max(9.0, 0.37 * len(comparison) + 3.2)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, figure_height),
        sharex=True,
        constrained_layout=True,
    )

    panel_specs = [
        (axes[0], f"wm_{score_suffix}", "wm_n_profiles", "WM"),
        (axes[1], f"gm_{score_suffix}", "gm_n_profiles", "GM"),
    ]
    for ax, score_column, n_column, tissue_label in panel_specs:
        plot_data = comparison.dropna(subset=[score_column, "metric_label"]).copy()
        plot_data["source_image"] = plot_data["family"].map(source_image_from_family)
        plot_data["source_order"] = (
            plot_data["source_image"].map(SOURCE_IMAGE_ORDER).fillna(len(SOURCE_IMAGE_ORDER))
        )
        # barh places the final row at the top, so ascending data gives
        # highest scores at the top. Source image is the secondary tie-breaker.
        plot_data = plot_data.sort_values(
            [score_column, "source_order", "metric_label"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        y_positions = np.arange(len(plot_data))
        colors = [
            METRIC_COLORS.get(label, SOURCE_IMAGE_COLORS["Other"])
            for label in plot_data["metric_label"]
        ]
        ax.barh(
            y_positions,
            plot_data[score_column],
            color=colors,
            edgecolor="black",
            linewidth=0.35,
            height=0.72,
            zorder=2,
        )

        score_values = plot_data[score_column].astype(float)
        below_one = score_values < 1.0
        if score_suffix == "accuracy" and below_one.any() and (~below_one).any():
            separator = int(below_one[below_one].index.max()) + 0.5
            ax.axhline(
                separator,
                color="black",
                linestyle=":",
                linewidth=1.2,
                zorder=4,
            )

        for index, row in plot_data.iterrows():
            score = float(row[score_column])
            n_profiles = row[n_column]
            if annotate_counts and pd.notna(n_profiles):
                denominator = int(round(float(n_profiles)))
                numerator = int(round(score * denominator))
                text = f"{score:.0%} ({numerator}/{denominator})"
            else:
                text = f"{score:.2f}" if score_suffix == "discriminability" else f"{score:.0%}"

            if score > 0.88:
                x_position, alignment = score - 0.012, "right"
            else:
                x_position, alignment = score + 0.012, "left"
            ax.text(
                x_position,
                index,
                text,
                va="center",
                ha=alignment,
                fontsize=7.5,
                color="#111111",
                zorder=3,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_data["metric_label"], fontsize=8)
        ax.set_ylabel(
            tissue_label,
            rotation=0,
            labelpad=30,
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.22, linewidth=0.6)
        ax.grid(axis="y", visible=False)

    axes[1].set_xlabel(x_label)
    if score_suffix == "accuracy":
        axes[1].xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    used_sources = [
        source
        for source in SOURCE_IMAGE_COLORS
        if source in {source_image_from_family(family) for family in comparison["family"]}
    ]
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=7,
            markerfacecolor=SOURCE_IMAGE_COLORS[source],
            markeredgecolor="black",
            markeredgewidth=0.45,
            label=source,
        )
        for source in used_sources
    ]
    axes[0].legend(
        handles=handles,
        title="Source image",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    return fig

# %% Notebook code cell 24
disc_figure = plot_discriminability_bars(
    nn_values,
    score_suffix="discriminability",
    x_label="Discriminability",
    annotate_counts=False,
)
for extension in ("png", "pdf"):
    out_path = OUTPUT_DIR / f"selected_metric_discriminability.{extension}"
    disc_figure.savefig(out_path, bbox_inches="tight")
    print(f"Wrote: {out_path}")
nn_values.to_csv(
    OUTPUT_DIR / "selected_metric_discriminability_values.csv",
    index=False,
)
plt.show()

nn_figure = plot_discriminability_bars(
    nn_values,
    score_suffix="accuracy",
    x_label="Nearest-neighbor accuracy",
    annotate_counts=True,
)
for extension in ("png", "pdf"):
    out_path = OUTPUT_DIR / f"selected_metric_nearest_neighbor_accuracy.{extension}"
    nn_figure.savefig(out_path, bbox_inches="tight")
    print(f"Wrote: {out_path}")
nn_values.to_csv(
    OUTPUT_DIR / "selected_metric_nearest_neighbor_accuracy_values.csv",
    index=False,
)
plt.show()

# %% [markdown] ## Output checklist
