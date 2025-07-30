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
import shutil
from pprint import pprint

import ants
import nibabel as nb
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
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
CODE_DIR = '/cbica/projects/nibs/code'


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
        'mni2t1w_xfm': {
            'datatype': 'anat',
            'session': Query.NONE,
            'run': [Query.NONE, Query.ANY],
            'from': 'MNI152NLin2009cAsym',
            'to': 'T1w',
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
    echo_times = [m['EchoTime'] for m in mese_ap_metadata]  # TEs in seconds

    # Get WM segmentation from sMRIPrep
    wm_seg_img = nb.load(run_data['dseg_mni'])
    wm_seg = wm_seg_img.get_fdata()
    wm_seg = (wm_seg == 2).astype(int)
    wm_seg_file = get_filename(
        name_source=run_data['dseg_mni'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MNI152NLin2009cAsym', 'desc': 'wm', 'suffix': 'mask'},
    )
    wm_seg_img = nb.Nifti1Image(wm_seg, wm_seg_img.affine, wm_seg_img.header)
    wm_seg_img.to_filename(wm_seg_file)

    # Warp WM segmentation to T1w space
    wm_seg_img = ants.image_read(wm_seg_file)
    wm_seg_t1w_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=wm_seg_img,
        transformlist=[run_data['mni2t1w_xfm']],
    )
    wm_seg_t1w_file = get_filename(
        name_source=wm_seg_file,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'wm', 'suffix': 'mask'},
    )
    ants.image_write(wm_seg_t1w_img, wm_seg_t1w_file)
    del wm_seg_img, wm_seg_t1w_img, wm_seg

    # Coregister echoes 2-4 of AP MESE data to echo 1
    hmced_files, brain_mask = iterative_motion_correction(
        name_sources=run_data['mese_mag_ap'],
        layout=layout,
        in_files=run_data['mese_mag_ap'],
        out_dir=out_dir,
        temp_dir=temp_dir,
    )

    # Coregister preprocessed T1w to MESEref space
    mese_mag_ap_echo1 = run_data['mese_mag_ap'][0]
    smriprep_to_mese_xfm = coregister_to_t1(
        name_source=name_source,
        layout=layout,
        in_file=run_data['t1w'],
        t1_file=mese_mag_ap_echo1,
        source_space='T1w',
        target_space='MESEref',
        out_dir=out_dir,
    )
    mese_t1w_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MESEref', 'suffix': 'T1w'},
        dismiss_entities=['echo'],
    )
    mese_t1w_img = ants.apply_transforms(
        fixed=ants.image_read(mese_mag_ap_echo1),
        moving=ants.image_read(run_data['t1w']),
        transformlist=[smriprep_to_mese_xfm],
        interpolator='lanczosWindowedSinc',
    )
    ants.image_write(mese_t1w_img, mese_t1w_file)

    mese_t1w_mask_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MESEref', 'suffix': 'mask'},
        dismiss_entities=['echo'],
    )
    mese_t1w_mask_img = ants.apply_transforms(
        fixed=ants.image_read(mese_mag_ap_echo1),
        moving=ants.image_read(run_data['t1w_mask']),
        transformlist=[smriprep_to_mese_xfm],
        interpolator='nearestNeighbor',
    )
    ants.image_write(mese_t1w_mask_img, mese_t1w_mask_file)

    # Calculate distortion map from AP and PA echo-1 data
    fieldmap_wf = init_fieldmap_wf(name='fieldmap_wf')
    fieldmap_wf.inputs.inputnode.epi_ref = (mese_mag_ap_echo1, mese_ap_metadata[0])
    fieldmap_wf.inputs.inputnode.epi_mask = brain_mask
    fieldmap_wf.inputs.inputnode.anat_ref = mese_t1w_file
    fieldmap_wf.inputs.inputnode.anat_mask = mese_t1w_mask_file
    basename = os.path.basename(mese_mag_ap_echo1).split('.')[0]
    fieldmap_wf.base_dir = os.path.join(temp_dir, basename)
    wf_res = fieldmap_wf.run()
    nodes = get_nodes(wf_res)
    fmap_xfm = nodes['fieldmap_wf.outputnode'].get_output('fmap')

    mese_mag_ap_echo1_sdc = ants.apply_transforms(
        fixed=ants.image_read(mese_mag_ap_echo1),
        moving=ants.image_read(mese_mag_ap_echo1),
        transformlist=[fmap_xfm],
    )
    mese_mag_ap_echo1_sdc_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MESEref', 'desc': 'SDC', 'suffix': 'MESE'},
    )
    ants.image_write(mese_mag_ap_echo1_sdc, mese_mag_ap_echo1_sdc_file)

    # Coregister AP echo-1 data to preprocessed T1w
    mese_to_smriprep_xfm = coregister_to_t1(
        name_source=name_source,
        layout=layout,
        in_file=mese_mag_ap_echo1_sdc_file,
        t1_file=run_data['t1w'],
        source_space='MESEref',
        target_space='T1w',
        out_dir=out_dir,
    )
    mese_mag_ap_echo1_t1_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MESE'},
    )
    mese_mag_ap_echo1_t1_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=ants.image_read(mese_mag_ap_echo1),
        transformlist=[mese_to_smriprep_xfm, fmap_xfm],
        interpolator='lanczosWindowedSinc',
    )
    ants.image_write(mese_mag_ap_echo1_t1_img, mese_mag_ap_echo1_t1_file)
    plot_coregistration(
        name_source=mese_mag_ap_echo1_t1_file,
        layout=layout,
        in_file=mese_mag_ap_echo1_t1_file,
        t1_file=run_data['t1w'],
        out_dir=out_dir,
        source_space='MESEref',
        target_space='T1w',
        wm_seg=wm_seg_t1w_file,
    )

    mese_mag_ap_echo1_mni_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MNI152NLin2009cAsym', 'suffix': 'MESE'},
    )
    mese_mag_ap_echo1_mni_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w_mni']),
        moving=ants.image_read(mese_mag_ap_echo1_t1_img),
        transformlist=[run_data['t1w2mni_xfm'], mese_to_smriprep_xfm, fmap_xfm],
        interpolator='lanczosWindowedSinc',
    )
    ants.image_write(mese_mag_ap_echo1_mni_img, mese_mag_ap_echo1_mni_file)
    plot_coregistration(
        name_source=mese_mag_ap_echo1_mni_file,
        layout=layout,
        in_file=mese_mag_ap_echo1_mni_file,
        t1_file=run_data['t1w_mni'],
        out_dir=out_dir,
        source_space='MESEref',
        target_space='MNI152NLin2009cAsym',
        wm_seg=wm_seg_file,
    )

    # Calculate T2 map from AP MESE data
    t2_img, r2_img, s0_img, r_squared_img = fit_monoexponential(
        in_files=hmced_files,
        echo_times=echo_times,
    )

    # Warp T1w-space T1map and T1w image to MNI152NLin2009cAsym using normalization transform
    # from sMRIPrep and coregistration transform to sMRIPrep's T1w space.
    image_types = ['T2map', 'R2map', 'S0map', 'Rsquaredmap']
    images = [t2_img, r2_img, s0_img, r_squared_img]
    for i_file, img in enumerate(images):
        suffix = image_types[i_file]
        file_ = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={
                'datatype': 'anat',
                'space': 'MESEref',
                'suffix': suffix,
                'extension': '.nii.gz',
            },
            dismiss_entities=['echo'],
        )
        img.to_filename(file_)

        # Warp to T1w space
        t1w_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w'},
        )
        t1w_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w']),
            moving=ants.image_read(file_),
            transformlist=[mese_to_smriprep_xfm, fmap_xfm],
        )
        ants.image_write(t1w_img, t1w_file)

        # Warp to MNI152NLin2009cAsym space
        mni_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym'},
        )
        mni_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm'], mese_to_smriprep_xfm, fmap_xfm],
        )
        ants.image_write(mni_img, mni_file)

        # Plot scalar map
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


def iterative_motion_correction(name_sources, layout, in_files, out_dir, temp_dir):
    """Apply iterative motion correction to a list of images.

    This method is based on the method described in https://doi.org/10.1101/2020.09.11.292649.

    Parameters
    ----------
    name_sources : list of str
        List of names of the source files to use for output file names.
    layout : BIDSLayout
        BIDSLayout object.
    in_files : list of str
        List of input image files.
    out_dir : str
        Directory to write output files.
    temp_dir : str
        Directory to write temporary files.

    Returns
    -------
    hmced_files : list of str
        List of paths to the motion-corrected images.
    brain_mask : str
        Path to the brain mask.
    """
    import antspynet

    # Step 1: Apply N4 bias field correction and skull-stripping to each image.
    brain_n4_files = []
    for i_file, in_file in enumerate(in_files):
        in_img = ants.image_read(in_file)
        # Bias field correction
        n4_img = ants.n4_bias_field_correction(in_img)

        # Skull-stripping
        dseg_img = antspynet.utilities.brain_extraction(n4_img, modality='t2')
        dseg_img = dseg_img['segmentation_image']
        mask_img = ants.threshold_image(
            dseg_img,
            low_thresh=1,
            high_thresh=1,
            inval=1,
            outval=0,
            binary=True,
        )
        if i_file == 0:
            brain_mask = get_filename(
                name_source=in_file,
                layout=layout,
                out_dir=out_dir,
                entities={'space': 'MESEref', 'desc': 'brain', 'suffix': 'mask'},
                dismiss_entities=['echo'],
            )
            ants.image_write(mask_img, brain_mask)

        n4_img_masked = n4_img * mask_img
        name_base = os.path.basename(name_sources[i_file])
        brain_n4_file = os.path.join(temp_dir, f'brain_n4_{i_file}_{name_base}')
        ants.image_write(n4_img_masked, brain_n4_file)
        brain_n4_files.append(brain_n4_file)

    # Step 2: Register each image to the first image.
    transforms = []
    for i_file, brain_n4_file in enumerate(brain_n4_files):
        in_file = in_files[i_file]
        if i_file == 0:
            template_img = ants.image_read(brain_n4_file)
            transform = os.path.join(CODE_DIR, 'processing', 'itkIdentityTransform.txt')
        else:
            reg = ants.registration(
                fixed=template_img,
                moving=ants.image_read(brain_n4_file),
                type_of_transform='Rigid',
            )
            transform = reg['fwdtransforms'][0]

        # Step 2a: Write out individual transforms to ihMTRAGEref space.
        transform_file = get_filename(
            name_source=name_sources[i_file],
            layout=layout,
            out_dir=out_dir,
            entities={
                'from': 'MESE',
                'to': 'MESEref',
                'mode': 'image',
                'suffix': 'xfm',
                'extension': 'txt' if transform.endswith('.txt') else 'mat',
            },
        )
        shutil.copyfile(transform, transform_file)
        transforms.append(transform_file)

    # Step 3: Write out template image as MESEref.nii.gz
    template_file = get_filename(
        name_source=name_sources[0],
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'MESEref'},
        dismiss_entities=['echo', 'direction'],
    )
    ants.image_write(template_img, template_file)

    # Step 5: Apply transforms to original images.
    hmced_files = [in_files[0]]
    for i_file, in_file in enumerate(in_files[1:]):
        transform_file = transforms[i_file]
        out_file = get_filename(
            name_source=name_sources[i_file],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MESEref'},
        )
        out_img = ants.apply_transforms(
            fixed=ants.image_read(template_file),
            moving=ants.image_read(in_file),
            transformlist=[transform_file],
            interpolator='lanczosWindowedSinc',
        )
        ants.image_write(out_img, out_file)

        plot_coregistration(
            name_source=out_file,
            layout=layout,
            in_file=out_file,
            t1_file=template_file,
            out_dir=out_dir,
            source_space='MESE',
            target_space='MESEref',
        )
        hmced_files.append(out_file)

    return hmced_files, brain_mask


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
    from sdcflows.workflows.fit.syn import init_syn_sdc_wf

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'epi_ref',
                'epi_mask',
                'anat_ref',
                'anat_mask',
            ]
        ),
        name='inputnode',
    )
    outputnode = pe.Node(CopyFiles(), name='outputnode')

    sdc_wf = init_syn_sdc_wf(name='sdc_wf', omp_nthreads=1)
    workflow.connect([
        (inputnode, sdc_wf, [
            ('epi_ref', 'inputnode.epi_ref'),
            ('epi_mask', 'inputnode.epi_mask'),
            ('anat_ref', 'inputnode.anat_ref'),
            ('anat_mask', 'inputnode.anat_mask'),
        ]),
        (sdc_wf, outputnode, [('outputnode.fmap', 'fmap')]),
    ])  # fmt:skip

    return workflow


class _CopyFilesInputSpec(BaseInterfaceInputSpec):
    fmap = traits.File(exists=True)


class _CopyFilesOutputSpec(TraitedSpec):
    fmap = traits.File(exists=True)


class CopyFiles(SimpleInterface):
    input_spec = _CopyFilesInputSpec
    output_spec = _CopyFilesOutputSpec

    def _run_interface(self, runtime):
        import shutil

        self._results['fmap'] = os.path.abspath('fmap.nii.gz')

        shutil.copyfile(self.inputs.fmap, self._results['fmap'])

        return runtime


def get_nodes(wf_results):
    """Load nodes from a Nipype workflow's results."""
    return {node.fullname: node for node in wf_results.nodes}


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
