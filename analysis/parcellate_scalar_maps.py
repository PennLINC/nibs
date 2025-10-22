"""Plot correlation matrices between myelin measures.

Parallelized across subjects/sessions with a process pool.
"""

import json
import os
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import ants
import nibabel as nb
import numpy as np
import pandas as pd
from nilearn import masking
import templateflow.api as tflow


def process_subject(
    subject,
    session,
    patterns,
    temp_dir,
    deriv_dir,
    wb_mask_path,
    cortical_gm_mask_path,
    deep_gm_mask_path,
    gm_mask_path,
    wm_mask_path,
    voxel_counts,
    carpet_dseg_path,
):
    """Process a single subject."""
    # Load shared resources inside the worker to avoid pickling large objects
    carpet_ants_img = ants.image_read(carpet_dseg_path)
    carpet_nb_img = nb.load(carpet_dseg_path)
    wb_img = nb.load(wb_mask_path)
    cortical_gm_img = nb.load(cortical_gm_mask_path)
    deep_gm_img = nb.load(deep_gm_mask_path)
    gm_img = nb.load(gm_mask_path)
    wm_img = nb.load(wm_mask_path)
    n_scalars = len(patterns)

    cortical_gm_arr = np.zeros((n_scalars, voxel_counts['cortical_gm']))
    deep_gm_arr = np.zeros((n_scalars, voxel_counts['deep_gm']))
    wm_arr = np.zeros((n_scalars, voxel_counts['wm']))
    gm_arr = np.zeros((n_scalars, voxel_counts['gm']))
    wb_arr = np.zeros((n_scalars, voxel_counts['wb']))

    scalar_counter = -1
    for scalar_name, scalar_pattern in patterns.items():
        pattern = scalar_pattern.format(subject=subject, session=session)
        clean_scalar_name = scalar_name.replace(' ', '_').replace('*', 'starsymbol')
        files = sorted(glob(os.path.join(deriv_dir, pattern)))
        scalar_counter += 1
        if len(files) == 0:
            print(f"No files found for {pattern}", flush=True)
            cortical_gm_arr[scalar_counter, :] = np.nan
            deep_gm_arr[scalar_counter, :] = np.nan
            wm_arr[scalar_counter, :] = np.nan
            gm_arr[scalar_counter, :] = np.nan
            wb_arr[scalar_counter, :] = np.nan
            continue
        elif len(files) != 1:
            print(f"Multiple files found for {pattern}", flush=True)
            cortical_gm_arr[scalar_counter, :] = np.nan
            deep_gm_arr[scalar_counter, :] = np.nan
            wm_arr[scalar_counter, :] = np.nan
            gm_arr[scalar_counter, :] = np.nan
            wb_arr[scalar_counter, :] = np.nan
            continue
        else:
            resampled_file = None
            if carpet_nb_img.header.get_zooms() != nb.load(files[0]).header.get_zooms():
                # Resample image to same resolution as dseg
                resampled_ants_img = ants.image_read(files[0]).resample_image_to_target(
                    carpet_ants_img,
                    interp_type='lanczosWindowedSinc',
                )
                resampled_file = os.path.join(
                    temp_dir,
                    f'{subject}_{session}_{clean_scalar_name}.nii.gz',
                )
                ants.image_write(resampled_ants_img, resampled_file)
                img = nb.load(resampled_file)
            else:
                img = nb.load(files[0])

            scalar_wb_arr = masking.apply_mask(img, wb_img)
            if scalar_wb_arr.ndim != 1:
                print(
                    f"Scalar {scalar_name} has {scalar_wb_arr.ndim} dimensions",
                    flush=True,
                )
                cortical_gm_arr[scalar_counter, :] = np.nan
                deep_gm_arr[scalar_counter, :] = np.nan
                wm_arr[scalar_counter, :] = np.nan
                gm_arr[scalar_counter, :] = np.nan
                wb_arr[scalar_counter, :] = np.nan
                continue

            cortical_gm_arr[scalar_counter, :] = np.nan_to_num(
                masking.apply_mask(img, cortical_gm_img),
            )
            deep_gm_arr[scalar_counter, :] = np.nan_to_num(masking.apply_mask(img, deep_gm_img))
            gm_arr[scalar_counter, :] = np.nan_to_num(masking.apply_mask(img, gm_img))
            wm_arr[scalar_counter, :] = np.nan_to_num(masking.apply_mask(img, wm_img))
            wb_arr[scalar_counter, :] = np.nan_to_num(scalar_wb_arr)

            if resampled_file:
                os.remove(resampled_file)

    # Save out arrays to disk
    cortical_gm_arr_file = os.path.join(temp_dir, f'{subject}_{session}_cortical_gm.npy')
    np.save(cortical_gm_arr_file, cortical_gm_arr)
    deep_gm_arr_file = os.path.join(temp_dir, f'{subject}_{session}_deep_gm.npy')
    np.save(deep_gm_arr_file, deep_gm_arr)
    wm_arr_file = os.path.join(temp_dir, f'{subject}_{session}_wm.npy')
    np.save(wm_arr_file, wm_arr)
    gm_arr_file = os.path.join(temp_dir, f'{subject}_{session}_gm.npy')
    np.save(gm_arr_file, gm_arr)
    wb_arr_file = os.path.join(temp_dir, f'{subject}_{session}_wb.npy')
    np.save(wb_arr_file, wb_arr)
    print(f"Processed {subject} {session}", flush=True)


if __name__ == "__main__":
    bids_dir = "/cbica/projects/nibs/dset"
    deriv_dir = "/cbica/projects/nibs/derivatives"
    temp_dir = "/cbica/projects/nibs/work/correlation_matrices"
    os.makedirs(temp_dir, exist_ok=True)
    out_dir = "../data"

    n_jobs = 4

    with open("patterns.json", "r") as f:
        patterns = json.load(f)

    flat_patterns = {k: v for subdict in patterns.values() for k, v in subdict.items()}

    carpet_dseg = tflow.get(
        "MNI152NLin2009cAsym",
        resolution="01",
        desc="carpet",
        suffix="dseg",
        extension="nii.gz",
    )
    carpet_dseg = str(carpet_dseg)
    carpet_ants_img = ants.image_read(carpet_dseg)
    carpet_nb_img = nb.load(carpet_dseg)
    carpet_dseg_data = carpet_nb_img.get_fdata()

    carpet_tsv = tflow.get("MNI152NLin2009cAsym", desc="carpet", suffix="dseg", extension="tsv")
    carpet_df = pd.read_table(carpet_tsv, index_col="index")

    voxel_counts = {}

    cortical_gm_idx = carpet_df.loc[100:201].index.values
    cortical_gm_mask = np.isin(carpet_dseg_data, cortical_gm_idx).astype(int)
    cortical_gm_img = nb.Nifti1Image(cortical_gm_mask, carpet_nb_img.affine, carpet_nb_img.header)
    voxel_counts['cortical_gm'] = np.sum(cortical_gm_mask)

    deep_gm_idx = carpet_df.loc[30:99].index.values
    deep_gm_mask = np.isin(carpet_dseg_data, deep_gm_idx).astype(int)
    deep_gm_img = nb.Nifti1Image(deep_gm_mask, carpet_nb_img.affine, carpet_nb_img.header)
    voxel_counts['deep_gm'] = np.sum(deep_gm_mask)

    wm_idx = carpet_df.loc[1:2].index.values
    wm_mask = np.isin(carpet_dseg_data, wm_idx).astype(int)
    wm_img = nb.Nifti1Image(wm_mask, carpet_nb_img.affine, carpet_nb_img.header)
    voxel_counts['wm'] = np.sum(wm_mask)

    gm_mask = cortical_gm_mask + deep_gm_mask
    gm_img = nb.Nifti1Image(gm_mask, carpet_nb_img.affine, carpet_nb_img.header)
    voxel_counts['gm'] = np.sum(gm_mask)

    wb_mask = wm_mask + gm_mask
    wb_img = nb.Nifti1Image(wb_mask, carpet_nb_img.affine, carpet_nb_img.header)
    voxel_counts['wb'] = np.sum(wb_mask)

    pretty_atlas = (wm_mask) + (deep_gm_mask * 2) + (cortical_gm_mask * 3)
    pretty_atlas_img = nb.Nifti1Image(pretty_atlas, carpet_nb_img.affine, carpet_nb_img.header)
    pretty_atlas_img.to_filename(os.path.join(temp_dir, 'pretty_atlas.nii.gz'))

    # Persist mask images once so workers can load from disk
    cortical_gm_mask_path = os.path.join(temp_dir, 'mask_cortical_gm.nii.gz')
    deep_gm_mask_path = os.path.join(temp_dir, 'mask_deep_gm.nii.gz')
    gm_mask_path = os.path.join(temp_dir, 'mask_gm.nii.gz')
    wm_mask_path = os.path.join(temp_dir, 'mask_wm.nii.gz')
    wb_mask_path = os.path.join(temp_dir, 'mask_wb.nii.gz')
    cortical_gm_img.to_filename(cortical_gm_mask_path)
    deep_gm_img.to_filename(deep_gm_mask_path)
    gm_img.to_filename(gm_mask_path)
    wm_img.to_filename(wm_mask_path)
    wb_img.to_filename(wb_mask_path)

    # Build list of (subject, session) tasks
    subject_dirs = sorted(glob(os.path.join(bids_dir, 'sub-*')))
    subjects = [os.path.basename(subject_dir) for subject_dir in subject_dirs]
    subjects = [subject for subject in subjects if not subject.startswith('sub-PILOT')]
    tasks = []
    for subject in subjects:
        session_dirs = sorted(glob(os.path.join(bids_dir, subject, 'ses-*')))
        sessions = [os.path.basename(session_dir) for session_dir in session_dirs]
        for session in sessions:
            tasks.append((subject, session))

    if n_jobs == 1:
        for subject, session in tasks:
            print(f'Processing {subject} {session}', flush=True)
            process_subject(
                subject=subject,
                session=session,
                patterns=flat_patterns,
                temp_dir=temp_dir,
                deriv_dir=deriv_dir,
                wb_mask_path=wb_mask_path,
                cortical_gm_mask_path=cortical_gm_mask_path,
                deep_gm_mask_path=deep_gm_mask_path,
                gm_mask_path=gm_mask_path,
                wm_mask_path=wm_mask_path,
                voxel_counts=voxel_counts,
                carpet_dseg_path=carpet_dseg,
            )
    else:
        print(f"Running with {n_jobs} workers across {len(tasks)} tasks", flush=True)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    process_subject,
                    subject,
                    session,
                    flat_patterns,
                    temp_dir,
                    deriv_dir,
                    wb_mask_path,
                    cortical_gm_mask_path,
                    deep_gm_mask_path,
                    gm_mask_path,
                    wm_mask_path,
                    voxel_counts,
                    carpet_dseg,
                )
                for subject, session in tasks
            ]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    # Surface exceptions but keep other tasks running
                    print(f"Task failed: {e}", flush=True)
