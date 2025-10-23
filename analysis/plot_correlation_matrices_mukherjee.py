"""Plot correlation matrices."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    grouping_df = pd.read_table('scalar_groups.tsv', index_col='scalar')

    to_use = [
        ['DSIStudio Tensor FA', 'FA', 'DTI', 'Anisotropy'],
        ['DSIStudio Tensor AD', 'AD', 'DTI', 'Diffusivity'],
        ['DSIStudio Tensor RD', 'RD', 'DTI', 'Diffusivity'],
        ['DSIStudio Tensor MD', 'MD', 'DTI', 'Diffusivity'],
        ['DKI Tensor FA', 'DKI-FA', 'DKI (DTI)', 'Anisotropy'],
        ['DKI AD', 'DKI-AD', 'DKI (DTI)', 'Complex Diffusivity'],
        ['DKI RD', 'DKI-RD', 'DKI (DTI)', 'Complex Diffusivity'],
        ['DKI MD', 'DKI-MD', 'DKI (DTI)', 'Complex Diffusivity'],
        ['DKI MK', 'MK', 'DKI', 'Kurtosis'],
        ['DKI AK', 'AK', 'DKI', 'Kurtosis'],
        ['DKI RK', 'RK', 'DKI', 'Kurtosis'],
        ['DKI KFA', 'KFA', 'DKI', 'Kurtosis'],
        ['DKI MKT', 'MKT', 'DKI', 'Kurtosis'],
        ['NODDI ICVF', 'ICVF', 'NODDI', 'Kurtosis'],
        ['NODDI OD', 'ODI', 'NODDI', 'Anisotropy'],
        ['NODDI ISOVF', 'ISOVF', 'NODDI', 'Diffusivity'],
        ['MSD', 'MSD', 'MAPMRI', 'Diffusivity'],  # DNE
        ['QIV', 'QIV', 'MAPMRI', 'Complex Diffusivity'],  # DNE
        ['TORTOISE MAPMRI RTOP', 'RTOP', 'MAPMRI', 'Diffusivity'],
        ['TORTOISE MAPMRI RTAP', 'RTAP', 'MAPMRI', 'Diffusivity'],
        ['TORTOISE MAPMRI RTPP', 'RTPP', 'MAPMRI', 'Diffusivity'],
    ]
    to_flip = [
        'NODDI OD',
        'TORTOISE MAPMRI RTAP',
        'TORTOISE MAPMRI RTOP',
        'TORTOISE MAPMRI RTPP',
    ]

    names = {
        'Whole Brain': 'mean_wb_corr_mat.tsv',
        'Gray Matter': 'mean_gm_corr_mat.tsv',
        'White Matter': 'mean_wm_corr_mat.tsv',
        'Cortical Gray Matter': 'mean_cortical_gm_corr_mat.tsv',
        'Deep Gray Matter': 'mean_deep_gm_corr_mat.tsv',
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

        # Select only the measures in to_use
        found_measures = [m[0] for m in to_use if m[0] in df.index]
        notfound_measures = [m[0] for m in to_use if m[0] not in df.index]
        df = df.loc[found_measures, found_measures]
        df.loc[:, notfound_measures] = np.nan
        df.loc[notfound_measures, :] = np.nan

        used_measures = [m[1] for m in to_use if m[1] in df.index]
        rename_dict = {m[0]: m[1] for m in to_use}
        df = df.rename(index=rename_dict, columns=rename_dict)
        df = df.loc[used_measures, used_measures]

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
        out_file = os.path.join('../figures', filename.split('.')[0] + '.png')

        # Add lines separating networks
        for idx in break_idx[1:-1]:
            ax.axes.axvline(idx, color='black')
            ax.axes.axhline(idx, color='black')

        ax.figure.savefig(out_file, bbox_inches='tight')
        plt.close()
