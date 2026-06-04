from __future__ import annotations

import argparse
import os
import subprocess
from pprint import pformat
from bids.layout import BIDSLayout, Query


# =========================================================
# RUNNER
# =========================================================
def run_commanda(cmd):
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
    "bids_dir": os.path.join(PROJECT_ROOT, "dset"),
    "code_dir": os.path.join(PROJECT_ROOT, "nibs"),
    "work_dir": os.path.join(PROJECT_ROOT, "work"),
    "derivatives": {
        "qsm": os.path.join(PROJECT_ROOT, "derivatives", "qsm"),
    },
}

CODE_DIR = CFG["code_dir"]


# =========================================================
# SAFE MATLAB STRING
# =========================================================
def mat_str(x):
    if x is None or x == "":
        return "''"
    return f"'{x}'"


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
        },
        "r2prime_e2345": {
            "datatype": "anat",
            "space": "MEGRE",
            "desc": "MEGRE+E2345",
            "suffix": "R2primemap",
        },
        "r2star_e12345": {
            "datatype": "anat",
            "space": "MEGRE",
            "desc": "MEGRE+E12345",
            "suffix": "R2starmap",
        },
        "r2star_e2345": {
            "datatype": "anat",
            "space": "MEGRE",
            "desc": "MEGRE+E2345",
            "suffix": "R2starmap",
        },
        "mask": {
            "datatype": "anat",
            "space": "MEGRE",
            "desc": "brain",
            "suffix": "mask",
        },
    }

    run_data = {}

    for key, q in queries.items():
        files = layout.get(**{**bids_filters, **q})

        if len(files) != 1:
            raise ValueError(f"{key}: expected 1 file, got {len(files)}")

        run_data[key] = files[0].path

    print("Collected run data:\n", pformat(run_data), flush=True)
    return run_data


# =========================================================
# PROCESS RUN
# =========================================================
def process_run(run_data, subject_id, session_id):

    work_dir = CFG["work_dir"]

    input_file = run_data["megre_echo1_mag"]

    if not input_file or not os.path.exists(input_file):
        raise ValueError(f"Invalid input_file: {input_file}")

    sepia_e12345 = os.path.join(
        work_dir,
        f"qsm-E12345+sepia/sub-{subject_id}/ses-{session_id}/anat"
    )

    sepia_e2345 = os.path.join(
        work_dir,
        f"qsm-E2345+sepia/sub-{subject_id}/ses-{session_id}/anat"
    )

    def out_dir(name):
        return os.path.join(
            work_dir,
            f"qsm-{name}/sub-{subject_id}/ses-{session_id}/anat"
        )

    mask = run_data["mask"]
    # echo_start: concat files hold all 5 echoes, so E12345 uses echo_start=1
    # (all echoes) and E2345 uses echo_start=2 (drop the first echo).
    combos = [
        # Use precomputed R2' and R2* maps.
        ("E12345+chisep+r2p", sepia_e12345, run_data["r2prime_e12345"], out_dir("E12345+chisep+r2p"), 1, 1, 0, run_data["r2star_e12345"]),
        ("E2345+chisep+r2p", sepia_e2345, run_data["r2prime_e2345"], out_dir("E2345+chisep+r2p"), 2, 1, 0, run_data["r2star_e2345"]),
        # Estimate R2* with ARLO and R2' with Chi-sepnet.
        ("E12345+chisep+r2primenet", sepia_e12345, "", out_dir("E12345+chisep+r2primenet"), 1, 0, 0, ""),
        ("E2345+chisep+r2primenet", sepia_e2345, "", out_dir("E2345+chisep+r2primenet"), 2, 0, 0, ""),
        # Estimate R2* with ARLO and R2' with scaling.
        ("E12345+chisep+r2s", sepia_e12345, "", out_dir("E12345+chisep+r2s"), 1, 0, 1, ""),
        ("E2345+chisep+r2s", sepia_e2345, "", out_dir("E2345+chisep+r2s"), 2, 0, 1, ""),
    ]

    matlab_dir = os.path.join(CODE_DIR, "processing")

    for label, sepia_folder, r2p, out, echo_start, has_r2p, is_scaling, r2s in combos:

        print("\n==============================")
        print(f"RUNNING: {label}")
        print("==============================\n", flush=True)

        matlab_cmd = (
            "try; "
            f"addpath(genpath('{matlab_dir}')); "
            "diary('matlab_debug.log'); "
            "disp('START'); "

            f"process_qsm_chisep('{input_file}',"
            f"'{sepia_folder}',"
            f"{mat_str(r2p)},"
            f"'{out}',"
            f"{echo_start},{has_r2p},{is_scaling},{mat_str(r2s)},{mat_str(mask)}); "

            "disp('END'); "
            "diary off; "
            "exit(0); "
            "catch ME; "
            "disp(getReport(ME,'extended')); "
            "diary off; "
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
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()

    layout = BIDSLayout(
        CFG["bids_dir"],
        config=os.path.join(CODE_DIR, "configuration", "nibs_bids_config.json"),
        validate=False,
        derivatives=[CFG["derivatives"]["qsm"]],
    )

    print(f"Processing sub-{args.subject_id}, ses-{args.session_id}", flush=True)

    run_data = collect_run_data(layout, {
        "subject": args.subject_id,
        "session": args.session_id,
    })

    process_run(run_data, args.subject_id, args.session_id)

    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
