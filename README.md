## aaai-claire-clustering

*CLAIRE: Clustering Evaluation based on Model Agreement and Item Response Theory*


**link:** <a href=https://anonymous.4open.science/r/aaai-claire-clustering-6113/README.md>anonymous-repository</a>

<sub>note: The anonymous repository encompasses all the outputs generated within this study.</sub>


## Poetry installation

Run:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

```bash
pip install poetry

```

## Download the zip

Download this zip file. Thus:

```bash
cd aaai-claire-clustering/
```

## Install project dependencies

Run the code bellow for install all dependecies of the project.

```bash
poetry install
```

## Run the pipeline

Run the pipeline for reproduction of the results.

```bash
poetry run python3 pipeline/run.py
```


## Experiments


Upon the completion of the experiments, a directory must be created, as follows:

- **[results](/results/):** a standard directory, containing the outcomes for each random partition addition.


## Figures

For the generation of all graphics, simply execute:

First:

```bash
cd notebooks/
```

then:

```bash
poetry run python3 executables.py . ../pipeline/executables/
```

note that:

```bash
poetry run python3 executables.py --help
```

return:

```bash
usage: executables.py [-h] [input_directory] [output_directory] [works]

Convert and excute the notebooks.

positional arguments:
  input_directory   Dir for notebooks to generate the plots. (default:
                    {YOU_PROJECT_DIR}/aaai-claire-
                    clustering/notebooks)

  output_directory  Output directory for .pys. (default:
                    {YOU_PROJECT_DIR}/aaai-claire-
                    clustering/pipeline/excutables)
  works             Number of works. (default: 8)

options:
  -h, --help        show this help message and exit
```

Subsequently, an output directory should emerge at the project's root, containing various graphics.

<!--
-->