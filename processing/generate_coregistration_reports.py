"""Create coregistration plots showing ses-01/ses-02 T1w coregistration for various derivatives.
"""

from __future__ import annotations

import argparse
import os

from bids.layout import BIDSLayout, Query
from nireports.assembler.report import Report

from utils import (
    get_filename,
    load_config,
    plot_coregistration,
)

CFG = load_config()
CODE_DIR = CFG['code_dir']


def get_tissue_segmentation(layout: object, subject_id: str) -> str | None:
    """Find a subject's sMRIPrep tissue segmentation in native T1w space.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the sMRIPrep derivatives.
    subject_id : str
        Subject label, without the ``sub-`` prefix.

    Returns
    -------
    dseg : str or None
        Path to the segmentation, or None if it could not be resolved
        unambiguously.

    Notes
    -----
    sMRIPrep writes a single subject-level segmentation on the same grid as the
    preprocessed T1w image, which is the grid every session's derivatives are
    resampled onto, so one segmentation applies to both sessions.
    """
    dseg_files = layout.get(
        subject=subject_id,
        datatype='anat',
        session=[Query.NONE, Query.ANY],
        run=[Query.NONE, Query.ANY],
        space=Query.NONE,
        desc=Query.NONE,
        suffix='dseg',
        extension=['.nii', '.nii.gz'],
    )
    if len(dseg_files) != 1:
        print(
            f'Expected 1 T1w-space dseg for subject {subject_id}, got {len(dseg_files)}. '
            'Coregistration plots will omit the tissue overlay.'
        )
        return None

    return dseg_files[0].path


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        default=None,
        help='Subject to process. If not provided, all subjects are processed.',
    )
    return parser


def _main(argv=None):
    """Run the coregistration report workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = CFG['bids_dir']
    out_dir = os.path.join(CFG['bids_dir'], 'derivatives', 'myelin')
    os.makedirs(out_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports_spec_coregistration.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'configuration', 'nibs_bids_config.json'),
        validate=False,
        derivatives=[
            CFG['derivatives']['smriprep'],
            CFG['derivatives']['pymp2rage'],
            CFG['derivatives']['ihmt'],
            CFG['derivatives']['mese'],
            CFG['derivatives']['megre'],
            #CFG['derivatives']['t1wt2w_ratio'],
            #CFG['derivatives']['g_ratio'],
            #CFG['derivatives']['qsiprep'],
        ],
    )

    # Define images to look for
    query_dict = {
        "mp2rage": {
            "space": "T1w",
            "desc": "B1correctedbrain",
            "suffix": "UNIT1",
            "extension": ".nii.gz",
        },
        "ihmt": {
            'acquisition': 'nosat',
            'mt': 'off',
            "space": "T1w",
            "suffix": "ihMTRAGE",
            "extension": ".nii.gz",
        },
        "mese": {
            "space": "T1w",
            "suffix": "MESEref",
            "extension": ".nii.gz",
        },
        'megre': {
            'space': 'T1w',
            'desc': 'rmsbrain',
            'suffix': 'MEGRE',
            'extension': '.nii.gz',
        },
    }

    if subject_id:
        subjects = [subject_id]
    else:
        subjects = layout.get_subjects()

    for subject_id in subjects:
        print(f'Processing subject {subject_id}')
        dseg = get_tissue_segmentation(layout, subject_id)
        for modality, query in query_dict.items():
            files = []
            for session in ['01', '02']:
                session_file = layout.get(
                    subject=subject_id,
                    session=session,
                    **query,
                )
                if not session_file:
                    print(f'No files found for subject {subject_id} and session {session}: {query}')
                    continue
                elif len(session_file) > 1:
                    print(f'Multiple files found for subject {subject_id} and session {session}: {query}')
                    continue

                files.append(session_file[0])

            if len(files) != 2:
                # At least one session not present, no need to plot
                print(f'At least one session missing for subject {subject_id}')
                continue

            out_file = get_filename(
                name_source=f'sub-{subject_id}/anat/sub-{subject_id}_space-T1w_desc-{modality}_{query["suffix"]}.nii.gz',
                layout=layout,
                out_dir=out_dir,
                entities={'space': 'T1w'},
            )
            plot_coregistration(
                name_source=out_file,
                layout=layout,
                in_file=files[1],
                t1_file=files[0],
                out_dir=out_dir,
                source_space='ses-01',
                target_space='ses-02',
                dseg=dseg,
                output_space='T1w',
            )

        report_dir = os.path.join(out_dir, f'sub-{subject_id}')
        robj = Report(
            report_dir,
            run_uuid=None,
            bootstrap_file=bootstrap_file,
            out_filename=f'sub-{subject_id}.html',
            reportlets_dir=out_dir,
            plugins=None,
            plugin_meta=None,
            subject=subject_id,
        )
        robj.generate_report()

    print('DONE!')


if __name__ == '__main__':
    _main()
