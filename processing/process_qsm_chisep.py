"""Process QSM data.

Steps:

0.  Load matlab/R2023B.
1.  Run chi-separation QSM estimation by calling the MATLAB script.

Notes:

- Remember to name the QSM files with the suffix "Chimap".
- Chimap outputs should be in parts per million (ppm).
"""

import argparse
import os
import subprocess
from glob import glob

CODE_DIR = '/home/tsalo/nibs/code/nibs'


def process_run(in_dir):
    """Process a single run of QSM data.

    Parameters
    ----------
    in_dir : str
        Path to the input directory.
    """
    # Collect MATLAB-compatible NIfTIs for QSM
    mag = os.path.join(in_dir, 'python_mag.nii')
    pha = os.path.join(in_dir, 'python_phase.nii')
    r2 = os.path.join(in_dir, 'python_r2.nii')
    sepia_head = os.path.join(CODE_DIR, 'sepia_header.mat')

    # Now run the chi-separation QSM estimation with R2' map
    chisep_r2p_dir = os.path.join(in_dir, 'chisep_r2p')
    os.makedirs(chisep_r2p_dir, exist_ok=True)

    subprocess.run(
        [
            "module",
            "load",
            "matlab/2023a;",
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            f"addpath('{CODE_DIR}');",
            "try;",
            f"run_chisep_script({mag}, {pha}, {sepia_head}, {chisep_r2p_dir}, {r2});",
            "catch e;",
            "disp(e.message);",
            "end;",
            "exit;",
        ],
    )
    chisep_r2p_iron_file = os.path.join(chisep_r2p_dir, 'Paramagnetic.nii.gz')
    if not os.path.isfile(chisep_r2p_iron_file):
        raise FileNotFoundError(f'chi-separation QSM output file {chisep_r2p_iron_file} not found')

    chisep_r2p_myelin_file = os.path.join(chisep_r2p_dir, 'Diamagnetic.nii.gz')
    if not os.path.isfile(chisep_r2p_myelin_file):
        raise FileNotFoundError(f'chi-separation QSM output file {chisep_r2p_myelin_file} not found')

    # Run X-separation QSM estimation without R2' map
    chisep_no_r_dir = os.path.join(in_dir, 'chisep_no_r2p')
    os.makedirs(chisep_no_r_dir, exist_ok=True)

    subprocess.run(
        [
            "module",
       	    "load",
       	    "matlab/2023a;",
            "matlab",
            "-nodisplay",
            "-nosplash",
            "-nodesktop",
            "-r",
            f"addpath('{CODE_DIR}');",
            "try;",
            f"run_chisep_script({mag}, {pha}, {sepia_head}, {chisep_no_r_dir});",
            "catch e;",
            "disp(e.message);",
            "end;",
            "exit;",
        ],
    )

    chisep_no_r2p_iron_file = os.path.join(chisep_no_r_dir, 'Paramagnetic.nii.gz')
    if not os.path.isfile(chisep_no_r2p_iron_file):
        raise FileNotFoundError(f'chi-separation QSM output file {chisep_no_r2p_iron_file} not found')

    chisep_no_r2p_myelin_file = os.path.join(chisep_no_r_dir, 'Diamagnetic.nii.gz')
    if not os.path.isfile(chisep_no_r2p_myelin_file):
        raise FileNotFoundError(f'chi-separation QSM output file {chisep_no_r2p_myelin_file} not found')


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        required=True,
    )
    return parser


def _main(argv=None):
    """Run the process_qsm_chisep workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = '/home/tsalo/nibs/data'

    run_folders = sorted(glob(os.path.join(in_dir, f'sub{subject_id}*')))
    print(f'Found {len(run_folders)} run folders')
    for run_folder in run_folders:
        print(f'Processing run folder {run_folder}')
        process_run(run_folder)

    print('DONE!')


if __name__ == '__main__':
    _main()
