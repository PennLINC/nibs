"""Shared project configuration loader."""

from __future__ import annotations

import os


def load_config(config_path: str | None = None) -> dict:
    """Load project path configuration from a YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML config file. Defaults to ``paths.yaml`` in the
        same directory as this module (the repository root).

    Returns
    -------
    config : dict
        Dictionary with resolved absolute paths. Keys include ``project_root``,
        ``bids_dir``, ``code_dir``, ``work_dir``, ``derivatives`` (dict),
        ``apptainer`` (dict), and ``freesurfer`` (dict).
    """
    import yaml

    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paths.yaml')

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

    config['apptainer'] = {key: os.path.join(root, path) for key, path in raw['apptainer'].items()}

    if 'sourcedata' in raw:
        config['sourcedata'] = {
            key: os.path.join(root, path) for key, path in raw['sourcedata'].items()
        }

    config['freesurfer'] = {
        key: os.path.join(root, path) for key, path in raw['freesurfer'].items()
    }

    return config
