"""Plot source modalities against the scalar sets they produce.

Scalars are grouped into sets by their exact modality signature, read from
scalar_modalities.json: every scalar in a set depends on the same modalities, so
each set fails or survives manual QC as a unit. MP2RAGE therefore points at two
separate sets, R1 and R1-B1c, and B1+ points only at the second.

MPRAGE T1w is the sMRIPrep anatomical reference, so it feeds every set. Those
edges are drawn thin and pale to keep them from swamping the specific
dependencies; where MPRAGE T1w is also a direct input it gets a full-weight edge.
"""

import json
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from workflow_graph import (
    INK,
    INK_MUTED,
    PALETTE,
    draw_box,
    draw_edge,
    save,
    tint,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Every scalar map is normalized through the sMRIPrep anatomical reference,
# which is built from this modality.
UNIVERSAL = 'MPRAGE T1w'

# Left-column order, top to bottom.
MODALITY_ORDER = [
    'MPRAGE T1w',
    'SPACE T1w',
    'SPACE T2w',
    'MP2RAGE',
    'B1+',
    'ihMTRAGE',
    'dMRI',
    'MEGRE',
    'MESE',
]

# Right-column order, chosen so each modality's edges stay bundled.
SET_ORDER = [
    ('MPRAGE T1w', 'SPACE T2w'),
    ('SPACE T1w', 'SPACE T2w'),
    ('MP2RAGE',),
    ('MP2RAGE', 'B1+'),
    ('MP2RAGE', 'ihMTRAGE', 'B1+'),
    ('ihMTRAGE',),
    ('dMRI', 'ihMTRAGE'),
    ('MP2RAGE', 'dMRI', 'ihMTRAGE', 'B1+'),
    ('dMRI',),
    ('MEGRE',),
    ('MEGRE', 'MESE'),
]

# Sets too large to name scalar by scalar get a stand-in and a qualifier. The two
# QSM sets share a group name, so the qualifier is what tells them apart.
STAND_INS = {
    ('dMRI',): ('dMRI Scalars', 'DKI, DSIStudio, NODDI, TORTOISE'),
    ('MEGRE',): ('QSM Scalars', 'SEPIA and χ-separation from R₂* alone'),
    ('MEGRE', 'MESE'): ('QSM Scalars', "χ-separation using a measured R₂′"),
}

MAX_NAMED = 3

MOD_X = 13.0
SET_X = 78.0
MOD_W = 20.0
SET_W = 44.0
MOD_H = 6.6
GAP = 2.4
TOP = 96.0

FAINT = '#d5d2cc'


def load_sets():
    """Group scalars by their modality signature, in SET_ORDER."""
    with open(os.path.join(_SCRIPT_DIR, 'scalar_modalities.json'), 'r') as fo:
        scalar_modalities = json.load(fo)

    grouped = OrderedDict()
    for scalar, modalities in scalar_modalities.items():
        grouped.setdefault(tuple(modalities), []).append(scalar)

    if set(grouped) != set(SET_ORDER):
        raise ValueError(
            'SET_ORDER is out of date with scalar_modalities.json:\n'
            f'  missing: {sorted(set(grouped) - set(SET_ORDER))}\n'
            f'  stale:   {sorted(set(SET_ORDER) - set(grouped))}'
        )

    return OrderedDict((signature, grouped[signature]) for signature in SET_ORDER)


def set_text(signature, scalars):
    """Return ``(headline, detail)`` for one scalar set."""
    pretty = ' + '.join(signature).replace('B1+', 'B₁⁺').replace('T1w', 'T₁w')
    pretty = pretty.replace('T2w', 'T₂w')
    if signature in STAND_INS:
        headline, qualifier = STAND_INS[signature]
        return headline, f'{pretty}\n{len(scalars)} maps · {qualifier}'
    if len(scalars) > MAX_NAMED:
        return f'{len(scalars)} maps', pretty
    return '\n'.join(scalars), pretty


if __name__ == '__main__':
    sets = load_sets()

    set_pos = {}
    y = TOP
    for signature, scalars in sets.items():
        headline, detail = set_text(signature, scalars)
        n_lines = len(headline.splitlines()) + len(detail.splitlines())
        height = max(MOD_H, 2.4 + 2.3 * n_lines)
        set_pos[signature] = (y - height / 2, height)
        y -= height + GAP
    set_span = TOP - (y + GAP)
    bottom = TOP - set_span

    # Spread the modalities over the full height of the set column rather than
    # centring a short stack, which keeps the edges short and roughly parallel.
    step = (set_span - MOD_H) / (len(MODALITY_ORDER) - 1)
    mod_pos = {
        modality: TOP - MOD_H / 2 - i * step
        for i, modality in enumerate(MODALITY_ORDER)
    }

    # Fan each modality's edges across its own right edge so they do not all
    # leave from a single point.
    targets = {m: [] for m in MODALITY_ORDER}
    for signature in sets:
        for modality in signature:
            targets[modality].append(signature)
        if UNIVERSAL not in signature:
            targets[UNIVERSAL].append(signature)

    anchors = {}
    for modality, signatures in targets.items():
        n = len(signatures)
        for j, signature in enumerate(signatures):
            offset = 0.0 if n == 1 else (j / (n - 1) - 0.5) * MOD_H * 0.62
            anchors[(modality, signature)] = mod_pos[modality] - offset

    # Fan the incoming edges across each set box's left edge too, ordered by the
    # height they leave from so arriving lines stay in the same relative order
    # and do not cross each other at the landing.
    landings = {}
    for signature, (cy, height) in set_pos.items():
        incoming = list(signature)
        if UNIVERSAL not in signature:
            incoming.append(UNIVERSAL)
        incoming.sort(key=lambda m: anchors[(m, signature)], reverse=True)
        n = len(incoming)
        for j, modality in enumerate(incoming):
            offset = 0.0 if n == 1 else (j / (n - 1) - 0.5) * height * 0.58
            landings[(modality, signature)] = cy - offset
    x0, x1 = 0.0, SET_X + SET_W / 2 + 4
    y0, y1 = bottom - 22, TOP + 12
    fig, ax = plt.subplots(figsize=((x1 - x0) / 6.4, (y1 - y0) / 6.4))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis('off')

    ax.text(MOD_X, TOP + 7, 'SOURCE MODALITY', ha='center', va='center',
            fontsize=11, fontweight='bold', color=INK_MUTED)
    ax.text(SET_X, TOP + 7, 'SCALAR SETS', ha='center', va='center',
            fontsize=11, fontweight='bold', color=INK_MUTED)

    # Backbone edges first, so the specific dependencies draw over them.
    for signature in set_pos:
        if UNIVERSAL in signature:
            continue
        draw_edge(ax, (MOD_X + MOD_W / 2, anchors[(UNIVERSAL, signature)]),
                  (SET_X - SET_W / 2, landings[(UNIVERSAL, signature)]),
                  rad=0.12, color=FAINT, linewidth=0.7, head=(3.5, 2.2), zorder=1)

    for signature in set_pos:
        for modality in signature:
            draw_edge(ax, (MOD_X + MOD_W / 2, anchors[(modality, signature)]),
                      (SET_X - SET_W / 2, landings[(modality, signature)]), rad=0.07)

    for modality, cy in mod_pos.items():
        label = modality.replace('B1+', 'B₁⁺').replace('T1w', 'T₁w').replace('T2w', 'T₂w')
        draw_box(ax, MOD_X, cy, MOD_W, MOD_H, label, '', 'modality', fontsize=9.5)

    for signature, (cy, height) in set_pos.items():
        headline, detail = set_text(signature, sets[signature])
        draw_box(ax, SET_X, cy, SET_W, height, headline, detail, 'output', fontsize=9)

    # Legend: two node roles and two edge weights.
    lx, ly = 2.0, bottom - 6
    for i, (role, label) in enumerate((('modality', 'Source modality'),
                                       ('output', 'Scalar set (one modality signature)'))):
        ax.add_patch(
            FancyBboxPatch(
                (lx, ly - i * 4 - 0.8), 2.6, 1.6,
                boxstyle='round,pad=0,rounding_size=0.3',
                facecolor=tint(PALETTE[role]), edgecolor=PALETTE[role],
                linewidth=1.5, zorder=3,
            )
        )
        ax.text(lx + 3.6, ly - i * 4, label, ha='left', va='center',
                fontsize=9, color=INK_MUTED)

    draw_edge(ax, (lx + 0.2, ly - 8.6), (lx + 2.6, ly - 8.6))
    ax.text(lx + 3.6, ly - 8.6, 'Direct input to every scalar in the set',
            ha='left', va='center', fontsize=9, color=INK_MUTED)
    draw_edge(ax, (lx + 0.2, ly - 12.6), (lx + 2.6, ly - 12.6),
              color=FAINT, linewidth=0.7, head=(3.5, 2.2))
    ax.text(lx + 3.6, ly - 12.6,
            'Anatomical reference via sMRIPrep (cross-session, so not a per-session dependency)',
            ha='left', va='center', fontsize=9, color=INK_MUTED)

    ax.set_title('NIBS source modalities and the scalar sets they produce',
                 fontsize=15, fontweight='bold', color=INK, pad=16)

    save(fig, 'workflow_modality_sets')
    plt.close(fig)
