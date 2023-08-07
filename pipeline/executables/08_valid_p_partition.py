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
file_path_parameters = PROJECT_DIR / "conf" / "parameters.yml"

params = read_file_yaml(file_path_parameters)

path_results = PROJECT_DIR / params["results"]["filepath"]

ext_type = params["outputs"]["extension_type"]
ext_local_img = params["outputs"]["extension_local_img"]
ext_best_img = params["outputs"]["extension_best_img"]

file_path_abi_diff = {
    i: {
        name: {
            _param: path_results / i / name / "params" / Path(_param + ext_type)
            for _param in ["abilities", "diff_disc"]
        }
        for name in os.listdir(path_results / i)
    }
    for i in os.listdir(path_results)
}

file_path_pij = {
    i: {
        name: {_param: path_results / i / name / "pij" / Path(_param + ext_type) for _param in ["pij_true", "pij_pred"]}
        for name in os.listdir(path_results / i)
    }
    for i in os.listdir(path_results)
}

n_random = np.sort([int(i.replace("random_n", "")) for i in os.listdir(path_results) if ".placehold" not in i])
path_random = ["random_n" + str(i) for i in n_random]
path_results = [path_results / i for i in path_random]
path_results, path_random = get_last_modification_directory(path_results, path_random, params)


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


# ##  Methods

# In[ ]:


def equation_k(xi0: np.array, xi1: np.array, m_models: int, k_partitions: int) -> np.array:
    B = (m_models) / (m_models + k_partitions)
    _xi1_pred = xi0 * (B) + (1 / 2) * (1 - B)
    if np.sum(xi0 < xi1) >= int(np.ceil(len(xi0) / 2)):
        _xi1_pred = xi0 * (B) + (1 / 2) * (1 - B)
    else:
        _xi1_pred = xi0 * (B) - (1 / 2) * (1 - B)
    error = np.sqrt(np.sum((xi1 - _xi1_pred) ** (2)) / len(xi1))
    return _xi1_pred, error


# ##  Equation

# $$
# a_{ij} = a_{ij}\cdot B \pm \frac{1}{2}\cdot(1 - B)
# $$
#
# thus
#
# $$
# B = \frac{M}{M + K}
# $$

# ## Processing

# In[ ]:


error_partition = {}
partitions_compare = {}
for name in tqdm(config.file_names):
    error_partition[name] = {}
    partitions_compare[name] = {}
    for i0_random in path_random:
        error_partition[name][i0_random] = {}
        count = np.where(path_random[0] == np.array(path_random))[0][0] + 1
        for i1_random in path_random[(np.where(i0_random == np.array(path_random))[0][0] + 1) :]:
            m, k = len(np.unique(config.models_name_dataset[name])), int(i1_random.replace("random_", "")[1:])
            i0_text, i1_text = "$p_{" + str(k - 1) + "}$", "$p_{" + str(k) + "}$"
            i0_random_data = data_params[i0_random][name]["abilities"].copy()
            i1_random_data = data_params[i1_random][name]["abilities"].copy()
            #             i0_random_data  = data_pij[i0_random][name]["pij_true"].T.copy()
            #             i1_random_data  = data_pij[i1_random][name]["pij_pred"].T.copy()
            i0_random_data = (
                i0_random_data[~i0_random_data.index.str.startswith("random_model")]
                .reset_index()
                .rename(columns={"abilities": ""})
            )
            i1_random_data = (
                i1_random_data[~i1_random_data.index.str.startswith("random_model")]
                .reset_index()
                .rename(columns={"abilities": ""})
            )

            partitions_compare[name][i0_random + "_" + i1_random] = i0_random_data.merge(
                i1_random_data, on=["index"], suffixes=(i0_text, i1_text)
            )
            _i1_pred, _error = equation_k(
                partitions_compare[name][i0_random + "_" + i1_random].iloc[:, 1],
                partitions_compare[name][i0_random + "_" + i1_random].iloc[:, 2],
                m,
                k,
            )
            error_partition[name][i0_random][i1_random] = [_error]
            partitions_compare[name][i0_random + "_" + i1_random][i1_text + "_pred"] = _i1_pred


# ## Plot

# In[ ]:


figs = {}
for i_name, i_content in tqdm(error_partition.items()):
    figs[i_name] = {}
    for j_random, j_content_plot in i_content.items():
        if j_random == path_random[-1]:
            continue
        j_content_data = pd.DataFrame(j_content_plot).T
        j_content_data.index = j_content_data.index.str.replace("random_n", "$p_{(") + ")}$"
        _fig, _ax = plt.subplots(1, 1, figsize=(22, 10))
        _ax.plot(j_content_data.index, j_content_data.values)
        plt.ioff()
        figs[i_name][j_random] = _fig


# ## Save

# In[ ]:


# for contents in figs.values():
#     _fig, _file_path = contents["figure"], contents["file_path"]
#     _fig.savefig(_file_path)


# In[ ]:
