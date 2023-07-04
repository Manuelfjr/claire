# basics
import sys
from pathlib import Path

PROJECT_DIR = Path.cwd().parent
sys.path.append(str(PROJECT_DIR))
import re

import numpy as np

# models
from sklearn.cluster import DBSCAN, OPTICS, KMeans, MeanShift, SpectralClustering

# metrics
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from tslearn.clustering import KernelKMeans

# utils
from utils.reader import read_file_yaml

##### parameters #################################
path_root = PROJECT_DIR / "data"
path_conf = PROJECT_DIR / "conf"
file_path_parameters = path_conf / "parameters.yml"

##### read       ##################################
parameters = read_file_yaml(file_path_parameters)
general_params = [
    parameters["general"]["min"],
    parameters["general"]["max"],
    parameters["general"]["step"],
]
model_params = parameters["models"]

##### names      ##################################
file_names = parameters["datasets"]

##### modelling objects ###########################
dir_result = "results"
_default_models = {
    "kmeans": KMeans,
    "dbscan": DBSCAN,
    "kernel_kmeans": KernelKMeans,
    "spectral_clustering": SpectralClustering,
    "mean_shift": MeanShift,
    "optics": OPTICS,
}

_default_params = {
    "kmeans": [{"n_clusters": i} for i in range(*general_params)],
    "dbscan": [
        {
            "eps": round(i, model_params["dbscan"]["eps"]["max"]),
            "min_samples": model_params["dbscan"]["min_samples"],
        }
        for i in np.arange(*list(model_params["dbscan"]["eps"].values()))
    ],
    "spectral_clustering": [{"n_clusters": i} | model_params["spectral_clustering"] for i in range(*general_params)],
    "mean_shift": model_params["mean_shift"],
    "kernel_kmeans": [{"n_clusters": i} | model_params["kernel_kmeans"] for i in np.arange(*general_params)],
    "optics": model_params["optics"],
}

models, params = {}, {}
for i_name in parameters["models_fit"]:
    if (i_name in _default_models.keys()) or (i_name in _default_params.keys()):
        models[i_name] = _default_models[i_name]
        params[i_name] = _default_params[i_name]

metrics = [
    ("v_measure", v_measure_score, True),
    ("mutual_info", mutual_info_score, True),
    ("adjusted_rand_score", adjusted_rand_score, True),
    ("calinski_harabasz", calinski_harabasz_score, False),
    ("davies_bouldin", davies_bouldin_score, False),
    ("silhouette", silhouette_score, False),
]

models_name = []
for name, model in params.items():
    for name_to_process in model:
        text = str(list(name_to_process.items()))
        text = re.sub(",+", "_", text)
        text = text.replace(".", "_")
        text = re.sub("[^0-9a-zA-Z_-]", "", text)
        text = str(name + "_" + text).replace(" ", "_")
        models_name.append(text)
