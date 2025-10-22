"""Plot differences between correlation matrices."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    with open("patterns.json", "r") as f:
        patterns = json.load(f)

    modalities = list(patterns.keys())
    modalities = modalities[::-1]
    breaks = [len(scalars) for scalars in patterns.values()]
    breaks = [0] + breaks
    break_idx = np.cumsum(np.array(breaks))

    names = {
        'Whole Brain - White Matter': ('mean_wb_corr_mat.tsv', 'mean_wm_corr_mat.tsv'),
        'Gray Matter - White Matter': ('mean_gm_corr_mat.tsv', 'mean_wm_corr_mat.tsv'),
        'White Matter - Gray Matter': ('mean_wm_corr_mat.tsv', 'mean_gm_corr_mat.tsv'),
    }
    for title, filenames in names.items():
        df1 = pd.read_table(
            os.path.join('../data', filenames[0]),
            index_col='Image',
        )
        df2 = pd.read_table(
            os.path.join('../data', filenames[1]),
            index_col='Image',
        )
        df = df1.copy()
        df.loc[:, :] = df1.values - df2.values
        df = df.apply(np.tanh)
        arr = df.values
        np.fill_diagonal(arr, 0)
        df.loc[:, :] = arr

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        fig, ax = plt.subplots(figsize=(16, 12))
        ax = sns.heatmap(df, cmap=cmap, vmin=-1, vmax=1, square=True, ax=ax)
        # A tick label rotation can be set using Axes.tick_params.
        xlabels = ax.get_xticklabels()
        ticks = np.arange(0, df.shape[0]) + 0.5
        ax.set_xticks(ticks)
        ax.set_xticklabels(df.index, fontsize=7, ha='center')
        ax.set_yticks(ticks)
        ax.set_yticklabels(df.index, fontsize=7)
        ax.set_ylabel(None)
        ax.set_title(title, fontsize=16)
        out_file = os.path.join(
            '../figures',
            f'{title.lower().replace(" ", "_").replace("-", "minus")}.png',
        )

        # Add lines separating networks
        for idx in break_idx[1:-1]:
            ax.axes.axvline(idx, color='black')
            ax.axes.axhline(idx, color='black')

        ax.figure.savefig(out_file, bbox_inches='tight')
        plt.close()
