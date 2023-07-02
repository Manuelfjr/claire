# basics
# viz
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class GeneratePlots:
    def __init__(self, pij=None, data=None):
        self.pij = pij
        self.data = data

    def scatterplot(self, nrows=2, ncols=2, figsize=(18, 14), plot_parameters=None):
        
        params_english = ["difficulties", "discrimination"]
        params_portugues = ["dificuldades", "discriminações"]
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        if np.array(axes).ndim in [1]:
            axes = np.matrix(axes)
            if axes.shape[0] == 1:
                axes = axes.T

        for args in plot_parameters:
            (name, ax, params) = args

            ax = axes[ax[0], ax[1]]
            if name != "uniform":
                if name in params_english + self.pij.columns.to_list() + params_portugues:
                    points = ax.scatter(**params)
                    cbar = ax.collections[0].colorbar
                    if name in params_english:
                        ax.set_title(f"{name}")
                    else:
                        ax.set_title(f"{name} [best]")

                    fig.colorbar(points, ax=ax)

                else:
                    params["ax"] = ax
                    sns.scatterplot(**params)  # scatterplot_diff_disc
                    ax.set_title(f"{name}")
                    ax.set_xlabel("pc1")
                    ax.set_ylabel("pc2")
                    cbar = ax.collections[0].colorbar
        return axes

    def scatterplot_diff_disc(self, nrows=5, ncols=2, figsize=(18, 14), plot_parameters=None):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for idxx1, _name in enumerate(self.data.keys()):
            for idxx2, args in enumerate(plot_parameters[_name]):
                (name, ax, _params) = args
                points = axes[ax[0], ax[1]].scatter(**_params)
                if idxx1 == 0:
                    axes[ax[0], ax[1]].set_title(f"{name}", fontsize=20)
                if idxx2 == 0:
                    axes[ax[0], ax[1]].set_ylabel(f"{_name}", fontsize=20)

                fig.colorbar(points, ax=axes[ax[0], ax[1]])

        return axes
