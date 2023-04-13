"""Utility functions."""
import dataclasses
import gin

@gin.configurable
@dataclasses.dataclass
class Config:
    name: str = 'default'

def load_config(args):
    # generate all gin bindings
    gin_bindings = []
    gin_bindings = gin_bindings + args.gin_params
    # parse config and gin bindings
    gin.parse_config_files_and_bindings([args.config], gin_bindings, skip_unknown=True)
    config = Config()
    return config

def load_gin_config(file, args=None):
    if args is not None:
        gin_bindings = []
        gin.parse_config_files_and_bindings([file], gin_bindings, skip_unknown=True)
    else: 
        gin.parse_config_file(file)
    config = Config()
    return config

