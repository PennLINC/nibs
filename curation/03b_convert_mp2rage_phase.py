"""Convert MP2RAGE DICOMs to NIfTI."""
import os
import re
from glob import glob


def run_command(command, env=None):
    """Run a given shell command with certain environment variables set.

    Copied from XCP-D.
    """
    import subprocess

    merged_env = os.environ
    if env:
        merged_env.update(env)

    process = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        env=merged_env,
    )
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break

    if process.returncode != 0:
        raise RuntimeError(
            f'Non zero return code: {process.returncode}\n{command}\n\n{process.stdout.read()}'
        )


if __name__ == '__main__':
    in_dir = '/cbica/projects/nibs/sourcedata/scitran/bbl/NIBS_857664'
    out_dir = '/cbica/projects/nibs/dset'

    subses_dirs = sorted(glob(os.path.join(in_dir, '*_*')))
    for subses_dir in subses_dirs:
        subses_id = os.path.basename(subses_dir)
        sub_id, ses_id = subses_id.split('_')
        sub_dir = os.path.join(out_dir, f'sub-{sub_id}', f'ses-{ses_id}')
        sub_out_dir = os.path.join(sub_dir, 'anat')

        dicom_dirs = sorted(
            glob(
                os.path.join(
                    subses_dir,
                    'CAMRIS^Satterthwaite',
                    'anat-MP2RAGE_RR_INV*_ND',
                    '*.dicom',
                ),
            )
        )
        if len(dicom_dirs) != 2:
            print(f'Expected 2 dicom dirs, got {len(dicom_dirs)}: {dicom_dirs}')
            continue

        for dicom_dir in dicom_dirs:
            dicom_dir_id = os.path.basename(dicom_dir)
            dicom_dir_id = dicom_dir_id.split('_')[0]
            inv = re.search(r'INV(\d)_', dicom_dir).group(1)
            nii_file = os.path.join(
                sub_out_dir,
                f'sub-{sub_id}_ses-{ses_id}_rec-defaced_run-01_inv-{inv}_part-phase_MP2RAGE.nii.gz',
            )
            cmd = f'dcm2niix -b y -f {nii_file} -o {sub_out_dir} {dicom_dir}'
            print(cmd)
