import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_DIR = Path.cwd()
sys.path.append(str(PROJECT_DIR))
warnings.filterwarnings("ignore")
print(PROJECT_DIR)
from src.claire import CLAIRE
from utils import config
from utils.reader import read_file_yaml

np.random.seed(0)

under_line = "\n{}\n"
title_part_n2 = "PROJECT_DIR: [ {} ]".format(PROJECT_DIR)
title_part_n3 = under_line.format("".join(["-"] * len(title_part_n2)))
title_part_n1 = under_line.format("".join(["-"] * len(title_part_n2)))
print(title_part_n1 + title_part_n2 + title_part_n3)

##### parameters #################################
path_root = PROJECT_DIR / "data"
path_conf = PROJECT_DIR / "conf"
file_path_parameters = path_conf / "parameters.yml"
path_data = [path_root / i for i in config.file_names]
parameters = read_file_yaml(file_path_parameters)
ext_type = parameters["outputs"]["extension_type"]
ext_local_img = parameters["outputs"]["extension_local_img"]
ext_best_img = parameters["outputs"]["extension_best_img"]
path_result = (
    PROJECT_DIR
    /  parameters["results"]["filepath"]
)
if not os.path.exists(path_result):
    os.makedirs(path_result)
    
#### read       ##################################
data_all = {i: pd.read_csv(path_data[idx] / Path(i + ext_type)) for idx, i in enumerate(config.file_names)}

#### set numer of random partitions ##############
models_name_dataset = {}
for dataset_name in config.file_names:
    models_name_dataset[dataset_name] = []
    for i_random in range(parameters["experiments"]["rp_init"], parameters["experiments"]["rp_final"]):
        models_name_dataset[dataset_name].append("random_partition_p{}".format(i_random))

#### running     #################################
_X, _Y = {}, {}
for i in config.file_names:
    _X[i] = data_all[i].drop("labels", axis=1).values
    _Y[i] = data_all[i]["labels"].values

init_generate = parameters["experiments"]["rp_init"]

if parameters["experiments"]["rp_final"] == "max":
    stop_generate = number_random_models
else:
    stop_generate = parameters["experiments"]["rp_final"]

# del config.params["optics"]
for k_random in tqdm(range(init_generate, stop_generate)):
    which_k_random = "n_random_model: [ {} ]".format(k_random + 1)
    print(title_part_n1 + which_k_random + title_part_n3)
    path_result_k_partition = path_result / Path(f"random_n{k_random+1}")
    if not os.path.exists(path_result_k_partition):
        os.makedirs(path_result_k_partition)

    for i in tqdm(config.file_names):
        models_params = config.params | {"optics": [config._optics_params[i]]}
        
        claire = CLAIRE(
            models_name=models_name_dataset[i],
            models={},
            params={},
            _X=_X,
            _Y=_Y,
            metrics=config.metrics,
            dir_result=path_result_k_partition,
            path_root=PROJECT_DIR,
        )

        if len(np.unique(_Y[i])) == 1:
            n_clusters = np.random.randint(1, 10)
        else:
            n_clusters = len(np.unique(_Y[i]))

        which_k_dataset = "dataset: [ {} ]".format(i)

        print(title_part_n1 + which_k_dataset + title_part_n3)
        
        pij = claire.generate_pij_matrix(
            pd.DataFrame(index = np.arange(0,_X[i].shape[0])),
            k_random + 1,
            n_clusters,
            True
        )

        # set beta4 params
        beta_params = parameters["beta_params"] | {"pij": pij, "n_respondents": pij.shape[1], "n_items": pij.shape[0]}

        # fit
        beta4_model = claire.fit_beta4(**beta_params)

        # metrics
        data_metrics = claire.calculate_metrics(claire._data_results, beta4_model, claire._X[i], claire._Y[i])

        # contents
        dir_contents = [
            (
                "metrics",
                (
                    "metrics" + ext_type,
                    data_metrics.sort_values("abilities", ascending=False),
                ),
                (None),
            ),
            (
                "pij",
                ("pij_true" + ext_type, pij),
                ("pij_pred" + ext_type, pd.DataFrame(claire.b4.pij, columns=pij.columns)),
            ),
            (
                "params",
                (
                    "abilities" + ext_type,
                    pd.DataFrame(claire.b4.abilities, index=pij.columns, columns=["abilities"]),
                ),
                (
                    "diff_disc" + ext_type,
                    pd.DataFrame(
                        {
                            "difficulty": claire.b4.difficulties,
                            "discrimination": claire.b4.discriminations,
                        }
                    ),
                ),
            ),
            ("labels", ("labels" + ext_type, claire._data_results), (None)),
        ]

        # save
        claire.save_results(i, dir_contents)
        