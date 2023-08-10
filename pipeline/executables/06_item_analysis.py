#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


# utils
import os
import sys
from pathlib import Path

PROJECT_DIR = Path.cwd().parent
sys.path.append(str(PROJECT_DIR))

# viz
import matplotlib.pyplot as plt

# basics
import numpy as np
import pandas as pd

# metrics
from tqdm import tqdm

# metrics
from utils import config
from utils.reader import read_file_yaml
from utils.utils import get_last_modification_directory

# ## Parameters

# In[ ]:


path_outputs = PROJECT_DIR / "outputs"
path_data_url = PROJECT_DIR / "data"
file_path_parameters = PROJECT_DIR / "conf" / "parameters.yml"
parameters = read_file_yaml(file_path_parameters)
path_results = PROJECT_DIR / parameters["results"]["filepath"]
path_results_url = PROJECT_DIR / parameters["results"]["filepath"]
ext_type = parameters["outputs"]["extension_type"]
ext_local_img = parameters["outputs"]["extension_local_img"]
ext_best_img = parameters["outputs"]["extension_best_img"]

n_random = np.sort([int(i.replace("random_n", "")) for i in os.listdir(path_results_url) if ".placehold" not in i])
path_random = ["random_n" + str(i) for i in n_random]
path_results = [path_results_url / i for i in path_random]

_, path_random = get_last_modification_directory(path_results, path_random, parameters)

file_path_abi_diff = {
    i_name: {
        name: {_param: i / name / "params" / Path(_param + ext_type) for _param in ["abilities", "diff_disc"]}
        for name in config.file_names
    }
    for i, i_name in zip(path_results, path_random)
}

file_path_pij = {
    i_name: {
        name: {_param: i / name / "pij" / Path(_param + ext_type) for _param in ["pij_true", "pij_pred"]}
        for name in config.file_names
    }
    for i, i_name in zip(path_results, path_random)
}


# ## Read datasets

# In[ ]:


# params
data_params = {
    i_random: {
        j_name: {k_param: pd.read_csv(k_content, index_col=0) for k_param, k_content in j_content.items()}
        for j_name, j_content in i_content.items()
    }
    for i_random, i_content in file_path_abi_diff.items()
}


# In[ ]:


# pij
data_pij = {
    i_random: {
        j_name: {k_param: pd.read_csv(k_content, index_col=0) for k_param, k_content in j_content.items()}
        for j_name, j_content in i_content.items()
    }
    for i_random, i_content in file_path_pij.items()
}


# ## Plot

# In[ ]:


figs = {}
for i_random in tqdm(path_random):
    figs[i_random] = {}
    _fig, _axes = plt.subplots(len(config.file_names), 2, figsize=(14, 20), **parameters["outputs"]["args"])
    for idx, (i_name, i_content) in enumerate(list(data_pij[i_random].items())):
        pij_true = i_content["pij_true"].copy()
        abilities = data_params[i_random][i_name]["abilities"]
        diffs = data_params[i_random][i_name]["diff_disc"]

        _params_plot = [
            {
                "x": _param[0],
                "y": _param[1],
            }
            for _param in [
                [pij_true.mean().values, abilities["abilities"].values],
                [1 - pij_true.mean(axis=1).values, diffs["difficulty"].values],
            ]
        ]

        for idx_param, _param in enumerate(_params_plot):
            if idx_param == 1:
                _param = _param | {"c": diffs["discrimination"].values}
                scatter = _axes[idx, idx_param].scatter(**_param)
                cbar = plt.colorbar(scatter, ax=_axes[idx, idx_param])
                cbar.set_label("discrimination")
            else:
                scatter = _axes[idx, idx_param].scatter(**_param)
            _axes[idx, idx_param].grid(True)
            _axes[idx, idx_param].set_title(
                i_name + " | {:.5}".format(np.corrcoef(_param["x"], _param["y"])[0, 1]), loc="center"
            )

        _axes[idx, 0].set_ylabel("abilities")
        _axes[idx, 1].set_ylabel("difficulty")

    _axes[-1, 0].set_xlabel("average_response")
    _axes[-1, 1].set_xlabel("1 - average_item")

    plt.tight_layout()
    plt.ioff()
    figs[i_random] = {
        "figure": _fig,
        "file_path": (path_outputs / Path("_".join([i_random, "abi", "diff", "versus"]) + ext_local_img)),
    }


# ## Save

# In[ ]:


for contents in figs.values():
    _fig, _file_path = contents["figure"], contents["file_path"]
    _fig.savefig(_file_path, **parameters["outputs"]["args"])


# In[ ]:
