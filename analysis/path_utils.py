"""Shared path helpers for NIBS analysis scripts."""

from __future__ import annotations

from pathlib import Path


def find_project_roots() -> tuple[Path, Path]:
    """Return the derivatives project root and code root for local or cluster layouts."""

    path = Path(__file__).resolve()
    for parent in path.parents:
        if (
            parent.name == 'code'
            and (parent.parent / 'derivatives').exists()
            and (parent / 'configuration' / 'patterns.json').exists()
        ):
            return parent.parent, parent
        if (
            (parent / 'configuration' / 'patterns.json').exists()
            and (parent / 'analysis').exists()
        ):
            return parent, parent
    fallback = path.parents[1]
    return fallback, fallback


PROJECT_ROOT, CODE_ROOT = find_project_roots()
DERIVATIVES_ROOT = PROJECT_ROOT / 'derivatives'
