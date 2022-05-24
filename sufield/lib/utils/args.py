import yaml

def get_args(file_path):
    with open(file_path) as f:
        args = yaml.load(f, yaml.Loader)
    return args