"""Calculate T2/R2/S0 maps from MESE data.

This is still just a draft.
I need to calculate SDC from the first echo and apply that to the T2 map.
Plus we need proper output names.

Steps:

1.  Calculate T2 map from AP MESE data.
2.  Calculate distortion map from AP and PA echo-1 data with SDCFlows.
    -   topup vs. 3dQwarp vs. something else?
    -   Currently disabled.
3.  Apply SDC transform to AP echo-1 image.
    - Currently disabled.  This is not needed for the T2 map.
4.  Coregister SDCed AP echo-1 image to preprocessed T1w from sMRIPrep.
    -   Currently using non-SDCed MESE data.
5.  Write out coregistration transform to preprocessed T1w.
6.  Warp T2 map to MNI152NLin2009cAsym (distortion map, coregistration transform,
    normalization transform from sMRIPrep).
7.  Warp S0 map to MNI152NLin2009cAsym.

Notes:

- The T2 map will be used for QSM processing.
- sMRIPrep's preprocessed T1w image is used as the "native T1w space".
- This must be run after sMRIPrep.
"""

import json
import os
from pprint import pprint

import ants
from bids.layout import BIDSLayout, Query
from nipype.interfaces.base import BaseInterfaceInputSpec, SimpleInterface, TraitedSpec, traits
from nireports.assembler.report import Report

from utils import (
    coregister_to_t1,
    fit_monoexponential,
    get_filename,
    plot_coregistration,
    plot_scalar_map,
)

os.environ['SUBJECTS_DIR'] = '/cbica/projects/nibs/derivatives/smriprep/sourcedata/freesurfer'


def collect_run_data(layout, bids_filters):
    queries = {
        # MESE images from raw BIDS dataset
        'mese_mag_ap': {
            'part': ['mag', Query.NONE],
            'echo': Query.ANY,
            'direction': 'AP',
            'suffix': 'MESE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mese_mag_pa': {
            'part': ['mag', Query.NONE],
            'echo': 1,
            'direction': 'PA',
            'suffix': 'MESE',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-space T1w image from sMRIPrep
        't1w': {
            'datatype': 'anat',
            'session': Query.NONE,
            'run': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep T1w-space brain mask
        't1w_mask': {
            'datatype': 'anat',
            'session': Query.NONE,
            'run': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep MNI-space brain mask
        'mni_mask': {
            'datatype': 'anat',
            'session': Query.NONE,
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space T1w image from sMRIPrep
        't1w_mni': {
            'datatype': 'anat',
            'session': Query.NONE,
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # Normalization transform from sMRIPrep
        't1w2mni_xfm': {
            'datatype': 'anat',
            'session': Query.NONE,
            'run': [Query.NONE, Query.ANY],
            'from': 'T1w',
            'to': 'MNI152NLin2009cAsym',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
        },
        # MNI-space dseg from sMRIPrep
        'dseg_mni': {
            'datatype': 'anat',
            'session': Query.NONE,
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'dseg',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    run_data = {}
    for key, query in queries.items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if key == 'mese_mag_ap' and len(files) != 4:
            raise ValueError(f'Expected 4 files for {key}, got {len(files)}')
        elif key == 'mese_mag_ap':
            files = [f.path for f in files]
        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')
        else:
            files = files[0].path

        run_data[key] = files

    pprint(run_data)

    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    """Process a single run of MESE data.

    TODO: Use SDCFlows to calculate and possibly apply distortion map.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary containing the paths to the MESE data.
    out_dir : str
        Path to the output directory.
    temp_dir : str
        Path to the temporary directory.
        Not currently used.
    """
    name_source = run_data['mese_mag_ap'][0]
    mese_ap_metadata = [layout.get_metadata(f) for f in run_data['mese_mag_ap']]
    mese_pa_metadata = layout.get_metadata(run_data['mese_mag_pa'])
    echo_times = [m['EchoTime'] for m in mese_ap_metadata]  # TEs in seconds

    # Coregister echoes 2-4 of AP MESE data to echo 1
    mese_ap_echo1 = run_data['mese_mag_ap'][0]
    mese_space_ap_files = [mese_ap_echo1]
    hmc_transforms = []
    for echo_file in run_data['mese_mag_ap'][1:]:
        echo = layout.get_file(echo_file).entities['echo']
        echo_meseref_filename = get_filename(
            name_source=echo_file,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MESE', 'suffix': 'MESE'},
            dismiss_entities=['part'],
        )
        echo_hmc_transform = coregister_to_t1(
            name_source=echo_file,
            layout=layout,
            in_file=echo_file,
            t1_file=mese_ap_echo1,
            source_space=f'Echo-{echo}',
            target_space='Echo-1',
            out_dir=out_dir,
        )
        echo_hmc_img = ants.apply_transforms(
            fixed=ants.image_read(mese_ap_echo1),
            moving=ants.image_read(echo_file),
            transformlist=[echo_hmc_transform],
        )
        echo_hmc_img.to_filename(echo_meseref_filename)
        mese_space_ap_files.append(echo_meseref_filename)
        hmc_transforms.append(echo_hmc_transform)

    # Calculate T2 map from AP MESE data
    t2_img, r2_img, s0_img, r_squared_img = fit_monoexponential(
        in_files=mese_space_ap_files,
        echo_times=echo_times,
    )
    t2_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'MESE',
            'suffix': 'T2map',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    t2_img.to_filename(t2_filename)

    r2_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'MESE',
            'suffix': 'R2map',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    r2_img.to_filename(r2_filename)

    s0_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'MESE',
            'suffix': 'S0map',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    s0_img.to_filename(s0_filename)

    r_squared_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'anat',
            'space': 'MESE',
            'desc': 'monoexp',
            'suffix': 'Rsquaredmap',
            'extension': '.nii.gz',
        },
        dismiss_entities=['echo', 'part'],
    )
    r_squared_img.to_filename(r_squared_filename)

    # Calculate distortion map from AP and PA echo-1 data
    mese_mag_ap_echo1 = run_data['mese_mag_ap'][0]
    mese_mag_pa_echo1 = run_data['mese_mag_pa']
    in_data = [mese_mag_ap_echo1, mese_mag_pa_echo1]
    metadata = [mese_ap_metadata[0], mese_pa_metadata]
    pepolar_estimate_wf = init_fieldmap_wf(name='pepolar_estimate_wf')
    pepolar_estimate_wf.inputs.inputnode.in_data = in_data
    pepolar_estimate_wf.inputs.inputnode.metadata = metadata
    pepolar_estimate_wf.base_dir = os.path.join(temp_dir, 'pepolar_estimate_wf')
    wf_res = pepolar_estimate_wf.run()
    fmap_file = wf_res.outputs.outputnode.fmap
    fmap_ref_file = wf_res.outputs.outputnode.fmap_ref

    mese_mag_ap_echo1_sdc = ants.apply_transforms(
        fixed=ants.image_read(mese_mag_ap_echo1),
        moving=ants.image_read(fmap_ref_file),
        transformlist=[fmap_file],
    )
    mese_mag_ap_echo1_sdc_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MESE', 'desc': 'SDC', 'suffix': 'MESE'},
        dismiss_entities=['part'],
    )
    ants.image_write(mese_mag_ap_echo1_sdc, mese_mag_ap_echo1_sdc_file)

    # Coregister AP echo-1 data to preprocessed T1w
    coreg_transform = coregister_to_t1(
        name_source=name_source,
        layout=layout,
        in_file=mese_mag_ap_echo1_sdc_file,
        t1_file=run_data['t1w'],
        source_space='MESE',
        target_space='T1w',
        out_dir=out_dir,
    )

    # Warp T1w-space T1map and T1w image to MNI152NLin2009cAsym using normalization transform
    # from sMRIPrep and coregistration transform to sMRIPrep's T1w space.
    # XXX: This ignores the SDC transform.
    image_types = ['T2map', 'R2map', 'S0map', 'Rsquaredmap']
    images = [t2_filename, r2_filename, s0_filename, r_squared_filename]
    for i_file, file_ in enumerate(images):
        suffix = os.path.basename(file_).split('_')[-1].split('.')[0]
        mni_file = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym', 'suffix': suffix},
            dismiss_entities=['echo', 'part'],
        )
        mni_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm'], coreg_transform, fmap_file],
        )
        ants.image_write(mni_img, mni_file)

        plot_coregistration(
            name_source=mni_file,
            layout=layout,
            in_file=mni_file,
            t1_file=run_data['t1w_mni'],
            out_dir=out_dir,
            source_space=suffix,
            target_space='MNI152NLin2009cAsym',
        )

        t1w_file = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w', 'suffix': suffix},
            dismiss_entities=['echo', 'part'],
        )
        t1w_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w']),
            moving=ants.image_read(file_),
            transformlist=[coreg_transform, fmap_file],
        )
        ants.image_write(t1w_img, t1w_file)

        plot_coregistration(
            name_source=t1w_file,
            layout=layout,
            in_file=t1w_file,
            t1_file=run_data['t1w'],
            out_dir=out_dir,
            source_space=suffix,
            target_space='T1w',
        )

        scalar_report = get_filename(
            name_source=mni_file,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'figures', 'desc': 'scalar', 'extension': '.svg'},
        )
        if image_types[i_file] == 'T2map':
            kwargs = {'vmin': 0, 'vmax': 0.08}
        elif image_types[i_file] == 'R2map':
            kwargs = {'vmin': 0, 'vmax': 20}
        elif image_types[i_file] == 'S0map':
            kwargs = {}
        elif image_types[i_file] == 'Rsquaredmap':
            kwargs = {'vmin': 0, 'vmax': 1}

        plot_scalar_map(
            underlay=run_data['t1w_mni'],
            overlay=mni_file,
            mask=run_data['mni_mask'],
            dseg=run_data['dseg_mni'],
            out_file=scalar_report,
            **kwargs,
        )


def init_fieldmap_wf(name='fieldmap_wf'):
    """Initialize a fieldmap workflow.

    Parameters
    ----------
    name : str, optional
        Name of the workflow.
    """
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_data', 'metadata']), name='inputnode')
    outputnode = pe.Node(CopyFiles(), name='outputnode')

    qwarp_wf = init_3dQwarp_wf(name='qwarp_wf')

    workflow.connect([
        (inputnode, qwarp_wf, [
            ('in_data', 'inputnode.in_data'),
            ('metadata', 'inputnode.metadata'),
        ]),
        (qwarp_wf, outputnode, [
            ('outputnode.fmap', 'fmap'),
            ('outputnode.fmap_ref', 'fmap_ref'),
        ]),
    ])  # fmt:skip

    return workflow


class _CopyFilesInputSpec(BaseInterfaceInputSpec):
    fmap = traits.File(exists=True)
    fmap_ref = traits.File(exists=True)


class _CopyFilesOutputSpec(TraitedSpec):
    fmap = traits.File(exists=True)
    fmap_ref = traits.File(exists=True)


class CopyFiles(SimpleInterface):
    input_spec = _CopyFilesInputSpec
    output_spec = _CopyFilesOutputSpec

    def _run_interface(self, runtime):
        import shutil

        self._results['fmap'] = os.path.abspath('fmap.nii.gz')
        self._results['fmap_ref'] = os.path.abspath('fmap_ref.nii.gz')

        shutil.copyfile(self.inputs.fmap, self._results['fmap'])
        shutil.copyfile(self.inputs.fmap_ref, self._results['fmap_ref'])

        return runtime


def init_3dQwarp_wf(omp_nthreads=1, name='pepolar_estimate_wf'):
    """
    Create the PEPOLAR field estimation workflow based on AFNI's ``3dQwarp``.

    This workflow takes in two EPI files that MUST have opposed
    :abbr:`PE (phase-encoding)` direction.
    Therefore, EPIs with orthogonal PE directions are not supported.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.pepolar import init_3dQwarp_wf
            wf = init_3dQwarp_wf()

    Parameters
    ----------
    debug : :obj:`bool`
        Whether a fast configuration of topup (less accurate) should be applied.
    name : :obj:`str`
        Name for this workflow
    omp_nthreads : :obj:`int`
        Parallelize internal tasks across the number of CPUs given by this option.

    Inputs
    ------
    in_data : :obj:`list` of :obj:`str`
        A list of two EPI files, the first of which will be taken as reference.

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap.
    fmap_ref : :obj:`str`
        The path of an unwarped conversion of the first element of ``in_data``.

    """
    from nipype.interfaces import afni
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
    from niworkflows.interfaces.fixes import FixHeaderRegistration as Registration
    from niworkflows.interfaces.freesurfer import StructuralReference
    from niworkflows.interfaces.header import CopyHeader
    from sdcflows import data
    from sdcflows.interfaces.utils import ConvertWarp, Flatten
    from sdcflows.utils.misc import front as _front
    from sdcflows.utils.misc import last as _last

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_data', 'metadata']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['fmap', 'fmap_ref']), name='outputnode')

    flatten = pe.Node(Flatten(), name='flatten')
    sort_pe = pe.Node(
        niu.Function(function=_sorted_pe, output_names=['sorted', 'qwarp_args']),
        name='sort_pe',
        run_without_submitting=True,
    )

    merge_pes = pe.MapNode(
        StructuralReference(
            auto_detect_sensitivity=True,
            initial_timepoint=1,
            fixed_timepoint=True,  # Align to first image
            intensity_scaling=True,
            # 7-DOF (rigid + intensity)
            no_iteration=True,
            subsample_threshold=200,
            out_file='template.nii.gz',
        ),
        name='merge_pes',
        iterfield=['in_files'],
    )

    pe0_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads, name='pe0_wf')
    pe1_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads, name='pe1_wf')

    align_pes = pe.Node(
        Registration(
            from_file=data.load('translation_rigid.json'),
            output_warped_image=True,
        ),
        name='align_pes',
        n_procs=omp_nthreads,
    )

    qwarp = pe.Node(
        afni.QwarpPlusMinus(
            blur=[-1, -1],
            environ={'OMP_NUM_THREADS': f'{min(omp_nthreads, 4)}'},
            minpatch=9,
            nopadWARP=True,
            noweight=True,
            pblur=[0.05, 0.05],
        ),
        name='qwarp',
        n_procs=min(omp_nthreads, 4),
    )

    to_ants = pe.Node(ConvertWarp(), name='to_ants', mem_gb=0.01)

    cphdr_warp = pe.Node(CopyHeader(), name='cphdr_warp', mem_gb=0.01)

    # fmt: off
    workflow.connect([
        (inputnode, flatten, [("in_data", "in_data"),
                              ("metadata", "in_meta")]),
        (flatten, sort_pe, [("out_list", "inlist")]),
        (sort_pe, qwarp, [("qwarp_args", "args")]),
        (sort_pe, merge_pes, [("sorted", "in_files")]),
        (merge_pes, pe0_wf, [(("out_file", _front), "inputnode.in_file")]),
        (merge_pes, pe1_wf, [(("out_file", _last), "inputnode.in_file")]),
        (pe0_wf, align_pes, [("outputnode.skull_stripped_file", "fixed_image")]),
        (pe1_wf, align_pes, [("outputnode.skull_stripped_file", "moving_image")]),
        (pe0_wf, qwarp, [("outputnode.skull_stripped_file", "in_file")]),
        (align_pes, qwarp, [("warped_image", "base_file")]),
        (inputnode, cphdr_warp, [(("in_data", _front), "hdr_file")]),
        (qwarp, cphdr_warp, [("source_warp", "in_file")]),
        (cphdr_warp, to_ants, [("out_file", "in_file")]),
        (to_ants, outputnode, [("out_file", "fmap")]),
    ])
    # fmt: on
    return workflow


def _sorted_pe(inlist):
    """
    Generate suitable inputs to ``3dQwarp``.

    Example
    -------
    >>> paths, args = _sorted_pe([
    ...     ("dir-AP_epi.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-AP_bold.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-PA_epi.nii.gz", {"PhaseEncodingDirection": "j"}),
    ...     ("dir-PA_bold.nii.gz", {"PhaseEncodingDirection": "j"}),
    ...     ("dir-AP_sbref.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-PA_sbref.nii.gz", {"PhaseEncodingDirection": "j"}),
    ... ])
    >>> paths[0]
    ['dir-AP_epi.nii.gz', 'dir-AP_bold.nii.gz', 'dir-AP_sbref.nii.gz']

    >>> paths[1]
    ['dir-PA_epi.nii.gz', 'dir-PA_bold.nii.gz', 'dir-PA_sbref.nii.gz']

    >>> args
    '-noXdis -noZdis'

    >>> paths, args = _sorted_pe([
    ...     ("dir-AP_epi.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-LR_epi.nii.gz", {"PhaseEncodingDirection": "i"}),
    ... ])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError:

    """
    out_ref = [inlist[0][0]]
    out_opp = []

    ref_pe = inlist[0][1]['PhaseEncodingDirection']
    for d, m in inlist[1:]:
        pe = m['PhaseEncodingDirection']
        if pe == ref_pe:
            out_ref.append(d)
        elif pe[0] == ref_pe[0]:
            out_opp.append(d)
        else:
            raise ValueError('Cannot handle orthogonal PE encodings.')

    return (
        [out_ref, out_opp],
        {'i': '-noYdis -noZdis', 'j': '-noXdis -noZdis', 'k': '-noXdis -noYdis'}[ref_pe[0]],
    )


if __name__ == '__main__':
    code_dir = '/cbica/projects/nibs/code'
    in_dir = '/cbica/projects/nibs/dset'
    smriprep_dir = '/cbica/projects/nibs/derivatives/smriprep'
    out_dir = '/cbica/projects/nibs/derivatives/mese'
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = '/cbica/projects/nibs/work/mese'
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(code_dir, 'processing', 'reports_spec_mese.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    dataset_description = {
        'Name': 'NIBS MESE Derivatives',
        'BIDSVersion': '1.10.0',
        'DatasetType': 'derivative',
        'DatasetLinks': {
            'raw': in_dir,
            'smriprep': smriprep_dir,
        },
        'GeneratedBy': [
            {
                'Name': 'Custom code',
                'Description': 'Custom Python code combining ANTsPy and tedana.',
                'CodeURL': 'https://github.com/PennLINC/nibs',
            }
        ],
    }
    with open(os.path.join(out_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, sort_keys=True, indent=4)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(code_dir, 'nibs_bids_config.json'),
        validate=False,
        derivatives=[smriprep_dir],
    )
    subjects = layout.get_subjects(suffix='MESE')
    for subject in subjects:
        print(f'Processing subject {subject}')
        sessions = layout.get_sessions(subject=subject, suffix='MESE')
        for session in sessions:
            print(f'Processing session {session}')
            mese_files = layout.get(
                subject=subject,
                session=session,
                echo=1,
                part=['mag', Query.NONE],
                direction='AP',
                suffix='MESE',
                extension=['.nii', '.nii.gz'],
            )
            for mese_file in mese_files:
                print(f'Processing MESE file {mese_file.path}')
                entities = mese_file.get_entities()
                entities.pop('echo')
                if 'part' in entities:
                    entities.pop('part')

                entities.pop('direction')
                try:
                    run_data = collect_run_data(layout, entities)
                except ValueError as e:
                    print(f'Failed {mese_file}')
                    print(e)
                    continue
                process_run(layout, run_data, out_dir, temp_dir)

            report_dir = os.path.join(out_dir, f'sub-{subject}', f'ses-{session}')
            robj = Report(
                report_dir,
                run_uuid=None,
                bootstrap_file=bootstrap_file,
                out_filename=f'sub-{subject}_ses-{session}.html',
                reportlets_dir=out_dir,
                plugins=None,
                plugin_meta=None,
                subject=subject,
                session=session,
            )
            robj.generate_report()

    print('DONE!')
