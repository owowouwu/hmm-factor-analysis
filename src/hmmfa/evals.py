import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

def simple_time_heatmap(mat : np.ndarray, xlabel = '', ylabel = ''):

    # Create a figure and axes for the faceted heatmaps
    T = mat.shape[0]
    fig, axes = plt.subplots(nrows=T, ncols=1, figsize=(10, 2 * T), sharex=True, sharey=True)

    # Create a color map
    cmap = 'viridis'
    norm = mcolors.Normalize(vmin=np.min(mat), vmax=np.max(mat))

    # Plot each heatmap
    for t in range(T):
        sns.heatmap(mat[t], ax=axes[t], cmap=cmap, norm = norm, cbar = False)
        axes[t].set_title(f'Time step {t}')
        axes[t].set_xlabel(xlabel)
        axes[t].set_ylabel(ylabel)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', pad=0.01, aspect=30, fraction=0.02, shrink=0.8)
    cbar.set_label('Value')

    # Adjust layout
    plt.tight_layout()
    plt.show()