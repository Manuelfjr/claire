import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path.cwd().parent
sys.path.append(str(PROJECT_DIR))


def extract_numerical_prefix(notebook_name):
    match = re.match(r"^(\d+\.{0,}\d+)", notebook_name)
    if match:
        return match.group()
    return None


def convert_and_execute_notebooks(input_directory, output_directory):
    if not os.path.isdir(input_directory):
        print(f"The path {input_directory} doesnt exist.")
        return

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    pattern = r"^\d+\.{0,}\d+.*\.ipynb$"
    notebook_files = [
        file
        for file in os.listdir(input_directory)
        if (
            os.path.isfile(os.path.join(input_directory, file))
            and re.match(pattern, file)
            and not file.startswith("01")
            and not file.startswith("99")
        )
    ]

    notebook_files.sort(key=extract_numerical_prefix)

    for notebook_file in notebook_files:
        notebook_path = os.path.join(input_directory, notebook_file)
        subprocess.run(["jupyter", "nbconvert", "--to", "script", notebook_path])
        py_file = os.path.splitext(notebook_file)[0] + ".py"
        py_file_path = os.path.join(input_directory, py_file)
        subprocess.run(["black", py_file_path])
        shutil.move(os.path.join(input_directory, py_file), os.path.join(output_directory, py_file))
        subprocess.run(["python", os.path.join(output_directory, py_file)])
        subprocess.run(["poetry", "run", "pre-commit", "run", "--files", os.path.join(output_directory, py_file)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and excute the notebooks.")
    parser.add_argument(
        "input_directory",
        nargs="?",
        default="notebooks/",
        help="Dir for notebooks to generate the plots. (default: /notebooks)",
    )
    parser.add_argument(
        "output_directory",
        nargs="?",
        default="pipeline/executables/",
        help="Dir for notebooks to generate the plots, output directory for .py's. (default: pipeline/executables/)",
    )
    args = parser.parse_args()

    convert_and_execute_notebooks(args.input_directory, args.output_directory)
