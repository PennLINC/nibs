"""Apply hierarchical clustering to correlation matrices."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list


SELECTED_SCALARS = [
    'DKI MD',
    'DKI MKT',
    'DKI Tensor FA',
    'NODDI ICVF',
    'NODDI ICVF Modulated',
    'TORTOISE MAPMRI RTOP',
    'ihMTR',
    'ihMTsat-B1c',
    'R1',
    'R1-B1c',
    'MPRAGE-MyelinW',
    'SPACE-MyelinW',
    'G-ihMTsat',
    'G-ihMTR',
    'QSM-SEPIA-E5',
    "QSM-X-R2'-E5-X",
    "QSM-X-R2'-E5-Para",
    "QSM-X-R2'-E5-Dia",
]


def mirror_linkage(Z):
    """Reverse dendrogram orientation by swapping children at every merge."""
    Z_flipped = Z.copy()
    Z_flipped[:, [0, 1]] = Z_flipped[:, [1, 0]]
    return Z_flipped


def save_clustermap(df, linkage_matrix, title, out_file, color_mapper=None):
    n = df.shape[0]
    font_size = max(10, min(16, round(900 / n)))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ax = sns.clustermap(
        df,
        figsize=(20, 20),
        cmap=cmap,
        cbar_pos=None,
        dendrogram_ratio=(0.1, 0),
        row_colors=color_mapper,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        vmin=-1,
        vmax=1,
    )
    ticks = np.arange(0, n) + 0.5
    ax.ax_heatmap.set_xticks(ticks)
    ax.ax_heatmap.set_xticklabels(ax.data2d.index, fontsize=font_size, rotation=45, ha='right')
    ax.ax_heatmap.set_yticks(ticks)
    ax.ax_heatmap.set_yticklabels(ax.data2d.index, fontsize=font_size)
    ax.ax_heatmap.set_ylabel(None)
    if color_mapper is not None:
        ax.ax_row_colors.set_xticks([])
    ax.figure.suptitle(title, fontsize=36, y=1.02)
    ax.figure.savefig(out_file, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    grouping_df = pd.read_table(os.path.join(_script_dir, 'scalar_groups.tsv'), index_col='scalar')
    out_dir = os.path.join(_script_dir, '..', 'figures', 'correlation_matrices')
    os.makedirs(out_dir, exist_ok=True)

    names = {
        'Whole Brain': 'mean_wb_corr_mat.tsv',
        'Gray Matter': 'mean_gm_corr_mat.tsv',
        'White Matter': 'mean_wm_corr_mat.tsv',
        'Cortical Gray Matter': 'mean_cortical_gm_corr_mat.tsv',
        'Deep Gray Matter': 'mean_deep_gm_corr_mat.tsv',
    }

    modality_groups = list(grouping_df['group'].unique())
    pal = sns.color_palette('hls', len(modality_groups))
    color_mapper = pd.Series({
        scalar: pal[modality_groups.index(row['group'])]
        for scalar, row in grouping_df.iterrows()
    })

    groups = ['all', 'dMRI', 'QSM', 'selected']
    for group in groups:
        for i_name, (title, filename) in enumerate(names.items()):
            df = pd.read_table(
                os.path.join(_script_dir, '..', 'data', filename),
                index_col='Image',
            )
            df = df.apply(np.tanh)
            arr = df.values
            np.fill_diagonal(arr, 1)
            df.loc[:, :] = arr

            if group == 'all':
                df_reduced = df.copy()
            elif group in ('dMRI', 'QSM'):
                cols = grouping_df.loc[grouping_df['group'] == group].index.tolist()
                df_reduced = df.loc[cols, cols]
            else:
                df_reduced = df.loc[SELECTED_SCALARS, SELECTED_SCALARS]

            # linkage_matrix = linkage(np.abs(df_reduced.values), optimal_ordering=True)
            linkage_matrix = linkage(df_reduced.values, optimal_ordering=True)
            leaves = leaves_list(linkage_matrix)

            if i_name == 0:
                first_leaves = leaves
                reversed_order = False
            else:
                rv_corr = np.corrcoef(leaves[::-1], first_leaves)[0, 1]
                fwd_corr = np.corrcoef(leaves, first_leaves)[0, 1]
                reversed_order = rv_corr > fwd_corr
                if reversed_order:
                    print(f'{title} forward correlation: {fwd_corr}, reverse correlation: {rv_corr}')
                    print(f'{title} is reversed')

            stem = filename.split('.')[0]
            stem = stem.replace('_corr_mat', '').replace('mean_', '')
            out_file = os.path.join(out_dir, f'group-{group}_tissue-{stem}_corrmat.png')

            if reversed_order:
                linkage_matrix = mirror_linkage(linkage_matrix)

            row_colors = color_mapper if group not in ('dMRI', 'QSM') else None
            save_clustermap(df_reduced, linkage_matrix, title, out_file, color_mapper=row_colors)
