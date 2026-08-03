"""Plot missingness.

MPRAGE
T1w SPACE
T2w SPACE
B1+
MP2RAGE
ihMTRAGE
dMRI
MESE
MEGRE
"""

import pandas as pd
import seaborn as sns

from utils import convert_to_multindex, matrix


LABEL_REPLACEMENTS = {
    'T1w': 'T₁w',
    'T2w': 'T₂w',
    'B1+': 'B₁⁺',
}


def relabel(column):
    """Replace acquisition abbreviations with their subscripted forms."""
    return next(
        (column.replace(old, new) for old, new in LABEL_REPLACEMENTS.items() if old in column),
        column,
    )


if __name__ == '__main__':
    df = pd.read_table('../data/missingness_list.tsv', index_col='participant_id')
    df = df.fillna(0)
    df['Session 01--MP2RAGE'] = df[['Session 01--MP2RAGE', 'Session 01--MP2RAGE-P']].mean(axis=1)
    df['Session 02--MP2RAGE'] = df[['Session 02--MP2RAGE', 'Session 02--MP2RAGE-P']].mean(axis=1)
    columns = df.columns.tolist()
    columns = [c for c in columns if not c.endswith('MP2RAGE-P')]
    # Hardcode column order until I re-run build_missingness_list.py
    columns = [
        'Session 01--MPRAGE T1w',
        'Session 01--SPACE T1w',
        'Session 01--SPACE T2w',
        'Session 01--B1+',
        'Session 01--MP2RAGE',
        'Session 01--ihMTRAGE',
        'Session 01--dMRI',
        'Session 01--MESE',
        'Session 01--MEGRE',
        'Session 02--MPRAGE T1w',
        'Session 02--SPACE T1w',
        'Session 02--SPACE T2w',
        'Session 02--B1+',
        'Session 02--MP2RAGE',
        'Session 02--ihMTRAGE',
        'Session 02--dMRI',
        'Session 02--MESE',
        'Session 02--MEGRE',
    ]
    df = df[columns]

    # Ratings of 'n/a' parse as NaN and so compare False. Columns without a QC counterpart
    # (e.g., G-Ratio) and subjects absent from the QC table are never grayed out.
    qc_df = pd.read_table('../data/manual_qc_modality.tsv', index_col='participant_id')
    excluded = (qc_df == 0).reindex(index=df.index, columns=df.columns, fill_value=False)

    df = df.rename(columns=relabel)
    excluded = excluded.rename(columns=relabel)
    subjects = df.index.tolist()
    pilot_subjects = [subj for subj in subjects if subj.startswith('sub-PILOT')]
    other_subjects = [subj for subj in subjects if not subj.startswith('sub-PILOT')]
    subjects = pilot_subjects + other_subjects
    df = df.loc[subjects]
    excluded = excluded.loc[subjects]
    df = convert_to_multindex(df)
    excluded = convert_to_multindex(excluded)
    pal = sns.color_palette('husl', 9) * 2
    ax = matrix(df, palette=pal, excluded=excluded)
    ax.figure.savefig(
        '../figures/data_missingness.png',
        bbox_inches='tight',
        dpi=400,
    )
