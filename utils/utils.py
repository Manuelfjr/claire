import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def set_params(content: dict, params: dict) -> dict:
    _ax = [[(idxx, 0), (idxx, 1)] for idxx, i in enumerate(range(len(content.keys())))]
    _base_list = {
        i: {
            "x": content[i].drop("labels", axis=1).values[:, 0],
            "y": content[i].drop("labels", axis=1).values[:, 1],
            "s": 20,
            "cmap": "Spectral_r",
        }
        for i in content.keys()
    }
    _plt_parameters = {
        i: [
            [
                "difficulties",  # "difficulties" or "dificuldades"
                ax[0],
                _base_list[i]
                | {
                    "c": params[i]["diff_disc"]["difficulty"],
                    "vmin": min(params[i]["diff_disc"]["difficulty"]),
                    "vmax": max(params[i]["diff_disc"]["difficulty"]),
                },
            ],
            [
                "discrimination",  # "discrimination" or "discriminações"
                ax[1],
                _base_list[i]
                | {
                    "c": params[i]["diff_disc"]["discrimination"],
                    "vmin": min(params[i]["diff_disc"]["discrimination"]),
                    "vmax": max(params[i]["diff_disc"]["discrimination"]),
                },
            ],
        ]
        for ax, i in zip(_ax, content.keys())
    }
    return _plt_parameters


def get_last_modification_time(directory_path: str) -> float:
    return max((entry.stat().st_mtime for entry in Path(directory_path).rglob("*")))


def format_human_readable_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def extract_number_from_directory_name(directory_name: str) -> int:
    match = re.search(r"random_.*?(\d+)$", directory_name)
    return int(match.group(1)) if match else 0


def get_last_modification_directory(
    path_results: List[Path], path_random: List[str], params: Optional[dict] = None
) -> List[str]:
    path_results_strings = [str(path) for path in path_results]

    if params and params.get("experiment_tests"):
        last_modify = extract_number_from_directory_name(max(path_results_strings, key=get_last_modification_time))
        if last_modify != params["experiments"]["rp_final"]:
            last_modify = last_modify - 1
        path_random = path_random[:last_modify]
        path_results = path_results[:last_modify]

    # most_recent_directory = max(path_results_strings, key=lambda p: get_last_modification_time(p))
    return path_results, path_random
