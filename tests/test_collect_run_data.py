"""Tier 2 mock tests for collect_run_data across processing modules."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module-level mocks for heavy dependencies
# ---------------------------------------------------------------------------
# These modules import heavy neuroimaging libraries at the top level.  We mock
# them here so that the tests can run in a lightweight CI environment.
for _mod_name in [
    'ants',
    'antspynet',
    'antspynet.utilities',
    'nibabel',
    'nilearn',
    'nilearn.image',
    'nilearn.masking',
    'nilearn.maskers',
    'nilearn.plotting',
    'nireports',
    'nireports.assembler',
    'nireports.assembler.report',
    'nireports.interfaces',
    'nireports.interfaces.reporting',
    'nireports.interfaces.reporting.base',
    'nireports.reportlets',
    'nireports.reportlets.utils',
    'pymp2rage',
    'bids',
    'bids.layout',
]:
    sys.modules.setdefault(_mod_name, MagicMock())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bids_file(path):
    """Create a mock BIDSFile-like object with .path and .filename attrs."""
    f = MagicMock()
    f.path = path
    f.filename = os.path.basename(path)
    return f


# ===================================================================
# process_mp2rage.collect_run_data
# ===================================================================


class TestCollectRunDataMp2rage:
    """Mock-based tests for process_mp2rage.collect_run_data."""

    @pytest.fixture(autouse=True)
    def _import_collect(self):
        """Import collect_run_data, patching load_config for module-level CFG."""
        # Force a fresh import each time
        sys.modules.pop('process_mp2rage', None)
        from process_mp2rage import collect_run_data
        self.collect_run_data = collect_run_data

    @pytest.fixture()
    def bids_filters(self):
        return {
            'subject': '01',
            'session': '01',
            'run': '01',
            'datatype': 'anat',
        }

    @pytest.fixture()
    def expected_keys(self):
        return [
            'inv1_magnitude',
            'inv1_phase',
            'inv2_magnitude',
            'inv2_phase',
            'b1_famp',
            'b1_anat',
            't1w',
            't1w_mni',
            't1w2mni_xfm',
            'mni2t1w_xfm',
            'dseg_mni',
            'mni_mask',
        ]

    def test_returns_all_keys(self, bids_filters, expected_keys):
        """When layout.get returns exactly 1 file per query, all keys are present."""
        layout = MagicMock()
        layout.get.return_value = [_make_bids_file('/fake/file.nii.gz')]

        run_data = self.collect_run_data(layout, bids_filters)

        for key in expected_keys:
            assert key in run_data, f'Missing key: {key}'

    def test_missing_phase_skipped(self, bids_filters):
        """Phase images are optional; missing phase should be silently skipped."""
        layout = MagicMock()

        def _get_side_effect(**kwargs):
            if kwargs.get('part') == 'phase':
                return []
            return [_make_bids_file('/fake/file.nii.gz')]

        layout.get.side_effect = _get_side_effect

        run_data = self.collect_run_data(layout, bids_filters)

        assert 'inv1_phase' not in run_data
        assert 'inv2_phase' not in run_data
        assert 'inv1_magnitude' in run_data

    def test_multiple_files_raises(self, bids_filters):
        """More than one file for a non-phase query should raise ValueError."""
        layout = MagicMock()
        layout.get.return_value = [
            _make_bids_file('/fake/a.nii.gz'),
            _make_bids_file('/fake/b.nii.gz'),
        ]

        with pytest.raises(ValueError, match='Expected 1 file'):
            self.collect_run_data(layout, bids_filters)


# ===================================================================
# process_g_ratio_scaling_factors.collect_run_data
# ===================================================================


class TestCollectRunDataScalingFactors:
    """Mock-based tests for process_g_ratio_scaling_factors.collect_run_data."""

    @pytest.fixture(autouse=True)
    def _import_collect(self):
        """Import collect_run_data, patching load_config for module-level CFG."""
        with patch('utils.load_config', return_value={
            'code_dir': '/fake/code',
            'bids_dir': '/fake/bids',
            'work_dir': '/fake/work',
            'derivatives': {'smriprep': '/fake/smriprep'},
        }):
            sys.modules.pop('process_g_ratio_scaling_factors', None)
            from process_g_ratio_scaling_factors import collect_run_data
        self.collect_run_data = collect_run_data

    @pytest.fixture()
    def bids_filters(self):
        return {
            'subject': '01',
            'session': '01',
            'run': '01',
            'datatype': 'anat',
        }

    @pytest.fixture()
    def expected_keys(self):
        return [
            'isovf_mni',
            'icvf_mni',
            'mtsat_mni',
            'ihmtr_mni',
            't1w2mni_xfm',
            'fs2t1w_xfm',
            'aseg_fsnative',
            'brain_fsnative',
        ]

    def test_returns_all_keys(self, bids_filters, expected_keys, tmp_path):
        """When layout.get returns one file per query, all keys are present."""
        # Create fake freesurfer files that the function checks with os.path.isfile
        fs_dir = tmp_path / 'smriprep' / 'sourcedata' / 'freesurfer' / 'sub-01' / 'mri'
        fs_dir.mkdir(parents=True)
        (fs_dir / 'aseg.mgz').touch()
        (fs_dir / 'brain.mgz').touch()

        smriprep_dir = str(tmp_path / 'smriprep')

        layout = MagicMock()

        # The function iterates queries in dict insertion order.  The first two
        # (isovf_mni, icvf_mni) have a ``param`` key that gets popped before
        # calling layout.get, then used to filter files by filename.  The
        # remaining four queries have no ``param``.  Return matching filenames.
        _responses = iter([
            [_make_bids_file('/fake/sub-01_param-isovf_dwimap.nii.gz')],
            [_make_bids_file('/fake/sub-01_param-icvf_dwimap.nii.gz')],
            [_make_bids_file('/fake/sub-01_ihMTsatB1sq.nii.gz')],
            [_make_bids_file('/fake/sub-01_ihMTR.nii.gz')],
            [_make_bids_file('/fake/sub-01_xfm.h5')],
            [_make_bids_file('/fake/sub-01_xfm.txt')],
        ])

        layout.get.side_effect = lambda **kw: next(_responses)

        run_data = self.collect_run_data(layout, bids_filters, smriprep_dir=smriprep_dir)

        for key in expected_keys:
            assert key in run_data, f'Missing key: {key}'

    def test_missing_file_raises(self, bids_filters):
        """Zero files for a query should raise ValueError."""
        layout = MagicMock()
        layout.get.return_value = []

        with pytest.raises(ValueError, match='Expected 1 file'):
            self.collect_run_data(layout, bids_filters, smriprep_dir='/fake/smriprep')
