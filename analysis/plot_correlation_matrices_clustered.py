"""Apply hierarchical clustering to correlation matrices."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    with open("patterns.json", "r") as f:
        patterns = json.load(f)

    names = {
        'Whole Brain': 'mean_wb_corr_mat.tsv',
        'Gray Matter': 'mean_gm_corr_mat.tsv',
        'White Matter': 'mean_wm_corr_mat.tsv',
        'Cortical Gray Matter': 'mean_cortical_gm_corr_mat.tsv',
        'Deep Gray Matter': 'mean_deep_gm_corr_mat.tsv',
    }
    pal = sns.color_palette("hls", len(patterns.keys()))
    color_mapper = {}
    for i_mod, (mod, subdict) in enumerate(patterns.items()):
        for measure in subdict.keys():
            color_mapper[measure] = pal[i_mod]

    color_mapper = pd.Series(color_mapper)

    for title, filename in names.items():
        df = pd.read_table(
            os.path.join('../data', filename),
            index_col='Image',
        )
        df = df.apply(np.tanh)
        arr = df.values
        np.fill_diagonal(arr, 1)
        df.loc[:, :] = arr

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        ax = sns.clustermap(
            df,
            figsize=(20, 20),
            cmap=cmap,
            cbar_pos=None,
            dendrogram_ratio=(0.1, 0),
            row_colors=color_mapper,
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
