import yaml
from typing import Any

def read_file_yaml(
    path: str
) -> Any:
    with open(path, 'r') as file:
        content_yaml = yaml.safe_load(file)
    return content_yaml
