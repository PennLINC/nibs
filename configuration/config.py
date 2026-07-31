"""Shared project configuration loader."""

from __future__ import annotations

import os


def load_config(config='pc') -> dict:
    """Load project path configuration from a YAML file.

    Parameters
    ----------
    config : str, optional
        Name of the config ('pc' or 'cubic').

    Returns
    -------
    config : dict
        Dictionary with resolved absolute paths. Keys include ``project_root``,
        ``bids_dir``, ``code_dir``, ``work_dir``, ``derivatives`` (dict),
        ``apptainer`` (dict), ``docker`` (dict), ``synthstrip_runtime`` (str),
        and ``freesurfer`` (dict).
    """
    import yaml

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'paths_{config}.yml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    root = raw['project_root']

    config = {
        'project_root': root,
        'bids_dir': os.path.join(root, raw['bids_dir']),
        'code_dir': os.path.join(root, raw['code_dir']),
        'work_dir': os.path.join(root, raw['work_dir']),
    }

    config['derivatives'] = {
        key: os.path.join(root, path) for key, path in raw['derivatives'].items()
    }

    if 'apptainer' in raw:
        config['apptainer'] = {
            key: os.path.join(root, path) for key, path in raw['apptainer'].items()
        }

    # Docker image references are tags (e.g. "freesurfer/synthstrip:1.7"), not
    # filesystem paths, so they must be kept verbatim (no project_root join).
    if 'docker' in raw:
        config['docker'] = dict(raw['docker'])

    # Container runtime used for SynthStrip skull-stripping ("apptainer" or "docker").
    config['synthstrip_runtime'] = raw.get('synthstrip_runtime', 'apptainer')

    if 'sourcedata' in raw:
        config['sourcedata'] = {
            key: os.path.join(root, path) for key, path in raw['sourcedata'].items()
        }

    config['freesurfer'] = {
        key: os.path.join(root, path) for key, path in raw['freesurfer'].items()
    }

    return config
