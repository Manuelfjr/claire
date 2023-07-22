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
