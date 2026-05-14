import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
