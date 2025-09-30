import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    names = {
        'Whole Brain': 'mean_wb_corr_mat.tsv',
        'Gray Matter': 'mean_gm_corr_mat.tsv',
        'White Matter': 'mean_wm_corr_mat.tsv',
    }
    for title, filename in names.items():
        df = pd.read_table(
            os.path.join('../data', filename),
            index_col='Image',
        )
        df = df.apply(np.tanh)
        arr = df.values
        np.fill_diagonal(arr, 0)
        df.loc[:, :] = arr

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        ax = sns.heatmap(df, cmap=cmap, vmin=-1, vmax=1)
        # A tick label rotation can be set using Axes.tick_params.
        xlabels = ax.get_xticklabels()
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.set_ylabel(None)
        ax.set_title(title, fontsize=16)
        out_file = os.path.join('../figures', filename.split('.')[0] + '.png')
        ax.figure.savefig(out_file, bbox_inches='tight')
        plt.close()
