# viz
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class GeneratePlots:
    def __init__(self, pij=None, data=None):
        self.pij = pij
        self.data = data

    def _scatterplot(self,
                    nrows=1,
                    ncols=1,
                    figsize=(18, 18),
                    plot_parameters=None,
                    fontsize=20,
                    xlabel="$pca_{(1)}$", 
                    ylabel="$pca_{(2)}$"):
        _fig = {}
        for which_param in plot_parameters:
            (name, _, params) = which_param
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            points = axes.scatter(**params)
            # axes.set_title(name)
            axes.set_xlabel(xlabel, fontsize=fontsize)
            axes.set_ylabel(ylabel, fontsize=fontsize)
            fig.colorbar(points, ax=axes)
            fig.tight_layout()
            _fig[name] = fig
            plt.close()
        return _fig
        
    def scatterplot(self,
                    nrows=1,
                    ncols=1,
                    figsize=(18, 18),
                    plot_parameters=None,
                    fontsize=20,
                    xlabel="$pca_{(1)}$", 
                    ylabel="$pca_{(2)}$"):
        _fig = {}
        (name_diff, _, params_diff), (name_disc, _, params_disc)  = plot_parameters[0], plot_parameters[1] 
        params_diff["s"] = params_disc["c"].values*100
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        points = axes.scatter(**params_diff)
        axes.set_xlabel(xlabel, fontsize=fontsize)
        axes.set_ylabel(ylabel, fontsize=fontsize)
        fig.colorbar(points, ax=axes)
        fig.tight_layout()
        _fig[name_diff] = fig
        plt.close()
        return _fig


    def scatterplot_diff_disc(self, nrows=5, ncols=2, figsize=(18, 14), plot_parameters=None, fontsize=20):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if isinstance(axes, matplotlib.axes._axes.Axes):
            axes = np.array([[axes]])
        elif axes.shape == (2,):
            axes = np.array([axes])

        for idxx1, _name in enumerate(self.data.keys()):
            for idxx2, args in enumerate(plot_parameters[_name]):
                (name, ax, _params) = args
                points = axes[ax[0], ax[1]].scatter(**_params)
                if idxx1 == 0:
                    axes[ax[0], ax[1]].set_title(f"{name}", fontsize=fontsize)
                if idxx2 == 0:
                    axes[ax[0], ax[1]].set_ylabel(f"{_name}", fontsize=fontsize)

                fig.colorbar(points, ax=axes[ax[0], ax[1]])
        plt.close()
        return fig, axes

    def scatterplot_diff_disc_unique(self, nrows=5, ncols=2, figsize=(18, 14), plot_parameters=None, fontsize = 20):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if isinstance(axes, matplotlib.axes._axes.Axes):
            axes = np.array([[axes]])
        elif axes.shape == (2,):
            axes = np.array([axes]).T
        for idxx1, _name in enumerate(self.data.keys()):
            for idxx2, args in enumerate(plot_parameters[_name]):
                (name, ax, _params) = args
                points = axes[ax[1], ax[0]].scatter(**_params)
                axes[ax[1], ax[0]].set_title(f"{_name}", fontsize=fontsize)
                axes[ax[1], ax[0]].set_ylabel(f"{name}", fontsize=fontsize)

                fig.colorbar(points, ax=axes[ax[1], ax[0]])
        plt.close()
        return fig, axes
