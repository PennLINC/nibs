"""
Process QSM data using chi-separation.

Steps:
1. Run chi-separation on SEPIA-preprocessed QSM data.

Notes:
- Must be run after process_qsm_sepia.py.
- Requires chi-sep MATLAB toolbox.
"""

from __future__ import annotations

import argparse
import os
from pprint import pformat
from bids.layout import BIDSLayout, Query

import subprocess
import sys


# =========================================================
# STREAMING COMMAND RUNNER (FIXED)
# =========================================================
def run_commanda(cmd):
    """
    Run subprocess with real-time stdout/stderr streaming.
    """

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Command failed with return code {process.returncode}")


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = "/project/nibs_data/chisep_20260522"

CFG = {
    "project_root": PROJECT_ROOT,
    "bids_dir": os.path.join(PROJECT_ROOT, "dset"),
    "code_dir": os.path.join(PROJECT_ROOT, "nibs"),
    "work_dir": os.path.join(PROJECT_ROOT, "work"),
    "derivatives": {
        "qsm": os.path.join(PROJECT_ROOT, "derivatives", "qsm"),
    },
}

CODE_DIR = CFG["code_dir"]


# =========================================================
# INPUT COLLECTION
# =========================================================
def collect_run_data(layout, bids_filters: dict) -> dict[str, str]:

    queries = {
        "megre_echo1_mag": {
            "datatype": "anat",
            "acquisition": "QSM",
            "part": "mag",
            "echo": 1,
            "space": Query.NONE,
            "desc": Query.NONE,
            "suffix": "MEGRE",
            "extension": [".nii", ".nii.gz"],
        },
        "r2prime_e12345": {
            "datatype": "anat",
            "space": "MEGRE",
            "desc": "MEGRE+E12345",
            "suffix": "R2primemap",
            "extension": [".nii", ".nii.gz"],
        },
        "r2prime_e2345": {
            "datatype": "anat",
            "space": "MEGRE",
            "desc": "MEGRE+E2345",
            "suffix": "R2primemap",
            "extension": [".nii", ".nii.gz"],
        },
        "r2star_e12345": {
            "datatype": "anat",
            "space": "MEGRE",
            "desc": "MEGRE+E12345",
            "suffix": "R2starmap",
            "extension": [".nii", ".nii.gz"],
        },
        "r2star_e2345": {
            "datatype": "anat",
            "space": "MEGRE",
            "desc": "MEGRE+E2345",
            "suffix": "R2starmap",
            "extension": [".nii", ".nii.gz"],
        },
    }

    run_data = {}

    for key, q in queries.items():
        query = {**bids_filters, **q}
        files = layout.get(**query)

        if len(files) != 1:
            raise ValueError(f"{key}: expected 1 file, got {len(files)}")

        run_data[key] = files[0].path

    print("Collected run data:\n", pformat(run_data), flush=True)
    return run_data


# =========================================================
# CHISEP PROCESSING
# =========================================================
def process_run(run_data, subject_id, session_id):

    work_dir = CFG["work_dir"]
    input_file = run_data["megre_echo1_mag"]

    sepia_e12345 = os.path.join(
        work_dir, "qsm-E12345+sepia", f"sub-{subject_id}", f"ses-{session_id}", "anat"
    )

    sepia_e2345 = os.path.join(
        work_dir, "qsm-E2345+sepia", f"sub-{subject_id}", f"ses-{session_id}", "anat"
    )

    def out_dir(name):
        return os.path.join(
            work_dir,
            f"qsm-{name}",
            f"sub-{subject_id}",
            f"ses-{session_id}",
            "anat",
        )

    combos = [
        ("E12345+chisep+r2p", sepia_e12345, run_data["r2prime_e12345"], out_dir("E12345+chisep+r2p"), 1, 1, 0, run_data["r2star_e12345"]),
        ("E2345+chisep+r2p", sepia_e2345, run_data["r2prime_e2345"], out_dir("E2345+chisep+r2p"), 1, 1, 0, run_data["r2star_e2345"]),
        ("E12345+chisep+r2primenet", sepia_e12345, "", out_dir("E12345+chisep+r2primenet"), 1, 0, 0, ""),
        ("E2345+chisep+r2primenet", sepia_e2345, "", out_dir("E2345+chisep+r2primenet"), 1, 0, 0, ""),
        ("E12345+chisep+r2s", sepia_e12345, "", out_dir("E12345+chisep+r2s"), 1, 0, 1, ""),
        ("E2345+chisep+r2s", sepia_e2345, "", out_dir("E2345+chisep+r2s"), 1, 0, 1, ""),
    ]

    matlab_dir = os.path.join(CODE_DIR, "processing")

    for label, sepia_folder, r2p, out, echo_start, has_r2p, is_scaling, r2s in combos:

        print(f"\n==============================")
        print(f"RUNNING: {label}")
        print(f"==============================\n", flush=True)

        matlab_cmd = (
            "try; "
            f"addpath(genpath('{matlab_dir}')); "
            "disp('===== MATLAB DIAGNOSTICS ====='); "
            "disp(which('importONNXNetwork')); "
            "ver; "
            "license('test','Neural_Network_Toolbox'); "
            f"disp('START: {label}'); "
            f"process_qsm_chisep('{input_file}','{sepia_folder}','{r2p}','{out}',"
            f"{echo_start},{has_r2p},{is_scaling},'{r2s}'); "
            "disp('END'); "
            "exit(0); "
            "catch ME; "
            "disp(getReport(ME,'extended')); "
            "exit(1); "
            "end;"
        )

        cmd = [
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            matlab_cmd,
        ]
        run_commanda(cmd)

        print(f"\nDONE: {label}\n", flush=True)


# =========================================================
# CLI
# =========================================================
def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--subject-id", required=True)
    p.add_argument("--session-id", required=True)
    return p


def main(subject_id, session_id):

    layout = BIDSLayout(
        CFG["bids_dir"],
        config=os.path.join(CODE_DIR, "configuration", "nibs_bids_config.json"),
        validate=False,
        derivatives=[CFG["derivatives"]["qsm"]],
    )

    print(f"Processing sub-{subject_id}, ses-{session_id}", flush=True)

    bids_filters = {
        "subject": subject_id,
        "session": session_id,
    }

    try:
        run_data = collect_run_data(layout, bids_filters)
    except ValueError as e:
        print("Failed to collect data:", e)
        return

    process_run(run_data, subject_id, session_id)

    print("DONE!", flush=True)


def _main():
    args = get_parser().parse_args()
    main(args.subject_id, args.session_id)


if __name__ == "__main__":
    _main()