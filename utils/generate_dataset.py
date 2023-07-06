# basics
import sys
from pathlib import Path

PROJECT_DIR = Path.cwd()
sys.path.append(str(PROJECT_DIR))

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# utils
import argparse
import os

# basics

file_path_parameters = (
    PROJECT_DIR 
    / "conf" 
    / "parameters"
) 
np.random.seed(0)

# args
parser = argparse.ArgumentParser(description="Process config")
parser.add_argument(
    "integers",
    metavar="N_SAMPLE",
    type=int,
    nargs="?",
    default=[500],
    help="an integer for length of the dataset generated.",
)
parser.add_argument(
    "object",
    metavar="PATH",
    type=str,
    nargs="?",
    default=[file_path_parameters],
    help=f"file path to parameters configuration. (default: {file_path_parameters})",
)
args = parser.parse_args()

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
noisy_circles = datasets.make_circles(n_samples=args.integers, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=args.integers, noise=0.05)
blobs = datasets.make_blobs(n_samples=args.integers, random_state=8)
no_structure = np.random.rand(1000, 2), None

# Anisotropicly distributed data
random_state = 170
X, y, *others = datasets.make_blobs(n_samples=args.integers, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=args.integers, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

datasets = [
    (
        noisy_circles,
        "noisy_circles",
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        "noisy_moons",
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    (
        varied,
        "varied",
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        aniso,
        "aniso",
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, "blobs", {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    (no_structure, "no_structure", {}),
]


dataset_std = []
for i_dataset, (dataset, name, algo_params) in enumerate(datasets):
    URL = (
        PROJECT_DIR
        / "data" 
        / name
        )
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    if not os.path.exists(URL):
        os.makedirs(URL)
    data = pd.DataFrame(X)
    data["labels"] = y
    #data.to_csv(os.path.join(URL, name + ".csv"), index=False)
print(data)
