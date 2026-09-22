"""Tests for publication-facing metric labels."""

import pytest

from metric_registry import metric_plot_label


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        ('PAth', 'PAθ'),
        ('NG Parallel', 'NG ∥'),
        ('NG (Parallel)', 'NG ∥'),
        ('NG (Perpendicular)', 'NG⊥'),
    ],
)
def test_metric_plot_label_symbols(source, expected):
    assert metric_plot_label(source) == expected
