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
from tqdm import tqdm

# metrics
from utils import config
from utils.reader import read_file_yaml
from utils.utils import get_last_modification_directory

np.random.seed(0)


# ## Parameters

# In[ ]:


path_conf = PROJECT_DIR / "conf"
file_path_parameters = path_conf / "parameters.yml"
path_data = PROJECT_DIR / "data"
params = read_file_yaml(file_path_parameters)

path_root = PROJECT_DIR / params["results"]["filepath"]
path_outputs = PROJECT_DIR / "outputs"

if not os.path.exists(path_outputs):
    os.makedirs(path_outputs)

n_random = np.sort([int(i.replace("random_n", "")) for i in os.listdir(path_root) if ".placehold" not in i])
path_random = ["random_n" + str(i) for i in n_random]
path_results = [path_root / i for i in path_random]

path_results, path_random = get_last_modification_directory(path_results, path_random, params)

ext_type = params["outputs"]["extension_type"]
ext_local_img = params["outputs"]["extension_local_img"]
ext_best_img = params["outputs"]["extension_best_img"]

file_path_labels = {
    i_random: {i_name: i_content / i_name / "labels" / f"labels{ext_type}" for i_name in config.file_names}
    for i_random, i_content in zip(path_random, path_results)
}
file_path_data = {i_name: path_data / i_name / f"{i_name}{ext_type}" for i_name in config.file_names}


# In[ ]:


file_path_data = {
    i_name: (path_data / i_name / f"{i_name}_pca{ext_type}")
    if f"{i_name}_pca{ext_type}" in os.listdir(path_data / i_name)
    else (path_data / i_name / f"{i_name}{ext_type}")
    for i_name in config.file_names
}


# In[ ]:


under_line = "\n{}\n"
title_part_n2 = "PROJECT_DIR: [ {} ]".format(PROJECT_DIR)
title_part_n3 = under_line.format("".join(["-"] * len(title_part_n2)))
title_part_n1 = under_line.format("".join(["-"] * len(title_part_n2)))
print(title_part_n1 + title_part_n2 + title_part_n3)


# ## Read

# In[ ]:


parameters = read_file_yaml(file_path_parameters)


# In[ ]:


data_labels = {
    i_random: {i_name: pd.read_csv(i_path, index_col=0) for i_name, i_path in i_content.items()}
    for i_random, i_content in tqdm(list(file_path_labels.items()))
}


# In[ ]:


data = {i_name: pd.read_csv(i_path) for i_name, i_path in file_path_data.items()}


# In[ ]:


init = params["outputs"]["init_values"]
metrics = {}
for name, url in zip(path_random, path_results):
    metrics[name] = {}
    for dataset in os.listdir(url):
        metrics[name][dataset] = pd.read_csv(url / Path(dataset) / "metrics" / Path("metrics" + ext_type), index_col=0)


# ## Plot model output

# In[ ]:


figs = {}
for i_random, i_content in tqdm(list(data_labels.items())[:1]):
    figs[i_random] = {}
    for j_name, j_content in i_content.items():
        figs[i_random][j_name] = {}
        for i_model in j_content.columns:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            axes.scatter(
                x=data[j_name].iloc[:, 0], y=data[j_name].iloc[:, 1], c=data_labels[i_random][j_name][i_model].values
            )
            axes.set_title(i_model)
            fig.tight_layout()
            plt.close()
            figs[i_random][j_name][i_model] = {
                "figure": fig,
                "filepath": path_outputs / f"{i_random}_{j_name}_{i_model}{ext_local_img}",
            }


# ## Save

# In[ ]:


for content in figs.values():
    for i_name, i_content in tqdm(list(content.items())):
        for j_model, j_content in list(i_content.items()):
            j_content["figure"].savefig(
                j_content["filepath"], format=ext_local_img[1:], **parameters["outputs"]["args"]
            )


# In[ ]:
