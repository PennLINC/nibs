from __future__ import annotations

from typing import Optional

from heudiconv.utils import SeqInfo


def create_key(
    template: Optional[str],
    outtype: tuple[str, ...] = ('nii.gz',),
    annotation_classes: None = None,
) -> tuple[str, tuple[str, ...], None]:
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return (template, outtype, annotation_classes)


def infotodict(
    seqinfo: list[SeqInfo],
) -> dict[tuple[str, tuple[str, ...], None], list]:
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    session: scan index for longitudinal acq
    """
    # for this example, we want to include copies of the DICOMs just for our T1
    # and functional scans
    outdicom = ('dicom', 'nii.gz')

    t1_mprage_norm = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_acq-MPRAGE_rec-norm_run-{item:02d}_T1w',
        outtype=outdicom,
    )
    t1_space_norm = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_acq-SPACE_rec-norm_run-{item:02d}_T1w',
        outtype=outdicom,
    )
    t2_space_norm = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_acq-SPACE_rec-norm_run-{item:02d}_T2w',
        outtype=outdicom,
    )
    t2_tse_norm = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_acq-TSE_rec-norm_run-{item:02d}_T2w',
        outtype=outdicom,
    )
    megre_mag = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_acq-QSM_run-{item:02d}_part-mag_MEGRE',
        outtype=outdicom,
    )
    megre_phase = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_acq-QSM_run-{item:02d}_part-phase_MEGRE',
        outtype=outdicom,
    )
    mese_echo1_pa = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_dir-PA_run-{item:02d}_echo-1_MESE',
        outtype=outdicom,
    )
    mese_echo1_ap = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_dir-AP_run-{item:02d}_echo-1_MESE',
        outtype=outdicom,
    )
    mese_echo2_ap = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_dir-AP_run-{item:02d}_echo-2_MESE',
        outtype=outdicom,
    )
    mese_echo3_ap = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_dir-AP_run-{item:02d}_echo-3_MESE',
        outtype=outdicom,
    )
    mese_echo4_ap = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_dir-AP_run-{item:02d}_echo-4_MESE',
        outtype=outdicom,
    )
    ihmtrage = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_run-{item:02d}_ihMTRAGE',
        outtype=outdicom,
    )
    mp2rage_inv1_mag = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_rec-norm_run-{item:02d}_inv-1_part-mag_MP2RAGE',
        outtype=outdicom,
    )
    mp2rage_inv1_phase = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_rec-norm_run-{item:02d}_inv-1_part-phase_MP2RAGE',
        outtype=outdicom,
    )
    mp2rage_inv2_mag = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_rec-norm_run-{item:02d}_inv-2_part-mag_MP2RAGE',
        outtype=outdicom,
    )
    mp2rage_inv2_phase = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_rec-norm_run-{item:02d}_inv-2_part-phase_MP2RAGE',
        outtype=outdicom,
    )
    mp2rage_uni = create_key(
        '{bids_subject_session_dir}/anat/{bids_subject_session_prefix}_rec-norm_run-{item:02d}_UNIT1',
        outtype=outdicom,
    )
    tb1tfl_anat = create_key(
        '{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_acq-anat_run-{item:02d}_TB1TFL',
        outtype=outdicom,
    )
    tb1tfl_famp = create_key(
        '{bids_subject_session_dir}/fmap/{bids_subject_session_prefix}_acq-famp_run-{item:02d}_TB1TFL',
        outtype=outdicom,
    )
    dwi_ap_sbref = create_key(
        '{bids_subject_session_dir}/dwi/{bids_subject_session_prefix}_acq-HBCD75_rec-norm_dir-AP_run-{item:02d}_sbref',
        outtype=outdicom,
    )
    dwi_pa_sbref = create_key(
        '{bids_subject_session_dir}/dwi/{bids_subject_session_prefix}_acq-HBCD75_rec-norm_dir-PA_run-{item:02d}_sbref',
        outtype=outdicom,
    )
    dwi_ap_mag = create_key(
        '{bids_subject_session_dir}/dwi/{bids_subject_session_prefix}_acq-HBCD75_rec-norm_dir-AP_run-{item:02d}_part-mag_dwi',
        outtype=outdicom,
    )
    dwi_ap_phase = create_key(
        '{bids_subject_session_dir}/dwi/{bids_subject_session_prefix}_acq-HBCD75_rec-norm_dir-AP_run-{item:02d}_part-phase_dwi',
        outtype=outdicom,
    )
    dwi_pa_mag = create_key(
        '{bids_subject_session_dir}/dwi/{bids_subject_session_prefix}_acq-HBCD75_rec-norm_dir-PA_run-{item:02d}_part-mag_dwi',
        outtype=outdicom,
    )
    dwi_pa_phase = create_key(
        '{bids_subject_session_dir}/dwi/{bids_subject_session_prefix}_acq-HBCD75_rec-norm_dir-PA_run-{item:02d}_part-phase_dwi',
        outtype=outdicom,
    )

    info: dict[tuple[str, tuple[str, ...], None], list] = {
        t1_mprage_norm: [],
        t1_space_norm: [],
        t2_space_norm: [],
        t2_tse_norm: [],
        megre_mag: [],
        megre_phase: [],
        mese_echo1_pa: [],
        mese_echo1_ap: [],
        mese_echo2_ap: [],
        mese_echo3_ap: [],
        mese_echo4_ap: [],
        ihmtrage: [],
        mp2rage_inv1_mag: [],
        mp2rage_inv1_phase: [],
        mp2rage_inv2_mag: [],
        mp2rage_inv2_phase: [],
        mp2rage_uni: [],
        tb1tfl_anat: [],
        tb1tfl_famp: [],
        dwi_ap_sbref: [],
        dwi_pa_sbref: [],
        dwi_ap_mag: [],
        dwi_ap_phase: [],
        dwi_pa_mag: [],
        dwi_pa_phase: [],
    }
    for s in seqinfo:
        # Anatomical scans (we only want the last one)
        if (
            ('anat-T1w_acq-MPRAGE' in s.protocol_name) or ('t1_mprage_sag_p3' in s.protocol_name)
        ) and ('NORM' in s.image_type):
            info[t1_mprage_norm].append(s.series_id)
        elif (
            (
                ('anat-T1w_acq-SPACE' in s.protocol_name)
                or ('t1_space_sag_p3_iso' in s.protocol_name)
            )
            and ('NORM' in s.image_type)
        ):
            info[t1_space_norm].append(s.series_id)
        elif ('anat-T2w_acq-SPACE' in s.protocol_name) and ('NORM' in s.image_type):
            # Some scans have DIS3D enabled. Some don't. I'll let CuBIDS figure that out.
            info[t2_space_norm].append(s.series_id)
        elif ('t2_tse_tra_512' in s.protocol_name) and ('NORM' in s.image_type):
            info[t2_tse_norm].append(s.series_id)
        elif 'ihMT' in s.protocol_name:
            info[ihmtrage].append(s.series_id)
        # XXX: Need to modify when we get phase data.
        # I don't know what differentiates phase and mag in the metadata.
        elif 'anat-MP2RAGE_INV1' in s.series_description:
            info[mp2rage_inv1_mag].append([s.series_id])
        elif 'anat-MP2RAGE_INV2' in s.series_description:
            info[mp2rage_inv2_mag].append([s.series_id])
        elif 'anat-MP2RAGE_UNI' in s.series_description:
            info[mp2rage_uni].append([s.series_id])
        elif 'anat-mese_echo-1_dir-pa' in s.protocol_name.lower():
            # dir was capitalized as DIR in some scans
            info[mese_echo1_pa].append([s.series_id])
        elif 'anat-mese_echo-1_dir-ap' in s.protocol_name.lower():
            info[mese_echo1_ap].append([s.series_id])
        elif 'anat-mese_echo-2_dir-ap' in s.protocol_name.lower():
            info[mese_echo2_ap].append([s.series_id])
        elif 'anat-mese_echo-3_dir-ap' in s.protocol_name.lower():
            info[mese_echo3_ap].append([s.series_id])
        elif 'anat-mese_echo-4_dir-ap' in s.protocol_name.lower():
            info[mese_echo4_ap].append([s.series_id])
        # Field maps
        elif ('fmap-TB1TFL' in s.protocol_name) and ('FLIP ANGLE MAP' not in s.image_type):
            info[tb1tfl_anat].append([s.series_id])
        elif ('fmap-TB1TFL' in s.protocol_name) and ('FLIP ANGLE MAP' in s.image_type):
            info[tb1tfl_famp].append([s.series_id])
        # QSM scans
        elif ('swi-swi_acq-QSM' in s.protocol_name) and ('M' in s.image_type):
            info[megre_mag].append([s.series_id])
        elif ('swi-swi_acq-QSM' in s.protocol_name) and ('P' in s.image_type):
            info[megre_phase].append([s.series_id])
        # DWI scans
        elif 'dwi-dwi_acq-HBCD75_dir-AP_SBRef' in s.series_description:
            info[dwi_ap_sbref].append([s.series_id])
        elif 'dwi-dwi_acq-HBCD75_dir-PA_SBRef' in s.series_description:
            info[dwi_pa_sbref].append([s.series_id])
        elif ('dwi-dwi_acq-HBCD75_dir-AP' in s.protocol_name) and ('NORM' in s.image_type):
            info[dwi_ap_mag].append([s.series_id])
        elif ('dwi-dwi_acq-HBCD75_dir-AP' in s.protocol_name) and ('NORM' not in s.image_type):
            info[dwi_ap_phase].append([s.series_id])
        elif ('dwi-dwi_acq-HBCD75_dir-PA' in s.protocol_name) and ('NORM' in s.image_type):
            info[dwi_pa_mag].append([s.series_id])
        elif ('dwi-dwi_acq-HBCD75_dir-PA' in s.protocol_name) and ('NORM' not in s.image_type):
            info[dwi_pa_phase].append([s.series_id])

    return info
