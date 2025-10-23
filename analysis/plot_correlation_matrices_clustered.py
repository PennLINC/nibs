"""Apply hierarchical clustering to correlation matrices."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list


def mirror_linkage(Z):
    """Return a mirrored version of linkage matrix Z by swapping children at every merge.

    This preserves validity while reversing the left-right orientation of the dendrogram.
    """
    Z_flipped = Z.copy()
    Z_flipped[:, [0, 1]] = Z_flipped[:, [1, 0]]
    return Z_flipped


if __name__ == "__main__":
    grouping_df = pd.read_table('scalar_groups.tsv', index_col='scalar')

    names = {
        'Whole Brain': 'mean_wb_corr_mat.tsv',
        'Gray Matter': 'mean_gm_corr_mat.tsv',
        'White Matter': 'mean_wm_corr_mat.tsv',
        'Cortical Gray Matter': 'mean_cortical_gm_corr_mat.tsv',
        'Deep Gray Matter': 'mean_deep_gm_corr_mat.tsv',
    }
    groups = list(grouping_df['group'].unique())
    pal = sns.color_palette("hls", len(groups))
    color_mapper = {}
    for i_row, row in grouping_df.iterrows():
        group_idx = groups.index(row['group'])
        color_mapper[i_row] = pal[group_idx]

    color_mapper = pd.Series(color_mapper)

    for i_name, (title, filename) in enumerate(names.items()):
        df = pd.read_table(
            os.path.join('../data', filename),
            index_col='Image',
        )
        df = df.apply(np.tanh)
        arr = df.values
        np.fill_diagonal(arr, 1)
        df.loc[:, :] = arr
        linkage_matrix = linkage(np.abs(arr), optimal_ordering=True)
        leaves = leaves_list(linkage_matrix)
        reverse_it = False
        if i_name == 0:
            first_leaves = leaves
        else:
            rv_corr = np.corrcoef(leaves[::-1], first_leaves)[0, 1]
            fwd_corr = np.corrcoef(leaves, first_leaves)[0, 1]
            print(f'{title} forward correlation: {fwd_corr}, reverse correlation: {rv_corr}')
            if rv_corr > fwd_corr:
                print(f'{title} is reversed')
                linkage_matrix_rv = mirror_linkage(linkage_matrix)
                reverse_it = True

        # Generate a custom diverging colormap
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
        )

        # A tick label rotation can be set using Axes.tick_params.
        xlabels = ax.ax_heatmap.get_xticklabels()
        ticks = np.arange(0, df.shape[0]) + 0.5
        ax.ax_heatmap.set_xticks(ticks)
        ax.ax_heatmap.set_xticklabels(ax.data2d.index, fontsize=6)
        ax.ax_heatmap.set_yticks(ticks)
        ax.ax_heatmap.set_yticklabels(ax.data2d.index, fontsize=6)
        ax.ax_heatmap.set_ylabel(None)
        #ax.ax_heatmap.set_title(title, fontsize=16)
        out_file = os.path.join('../figures', filename.split('.')[0] + '_clustered.png')
        ax.figure.suptitle(title, fontsize=36, y=1.02)
        ax.figure.savefig(out_file, bbox_inches='tight')
        plt.close()

        if reverse_it:
            ax = sns.clustermap(
                df,
                figsize=(20, 20),
                cmap=cmap,
                cbar_pos=None,
                dendrogram_ratio=(0.1, 0),
                row_colors=color_mapper,
                row_linkage=linkage_matrix_rv,
                col_linkage=linkage_matrix_rv,
            )

            # A tick label rotation can be set using Axes.tick_params.
            xlabels = ax.ax_heatmap.get_xticklabels()
            ticks = np.arange(0, df.shape[0]) + 0.5
            ax.ax_heatmap.set_xticks(ticks)
            ax.ax_heatmap.set_xticklabels(ax.data2d.index, fontsize=6)
            ax.ax_heatmap.set_yticks(ticks)
            ax.ax_heatmap.set_yticklabels(ax.data2d.index, fontsize=6)
            ax.ax_heatmap.set_ylabel(None)
            #ax.ax_heatmap.set_title(title, fontsize=16)
            out_file = os.path.join('../figures', filename.split('.')[0] + '_clustered_rv.png')
            ax.figure.suptitle(title, fontsize=36, y=1.02)
            ax.figure.savefig(out_file, bbox_inches='tight')
            plt.close()
