import os
from pathlib import Path
from typing import Dict, Optional
from nerfstudio.configs.base_config import InstantiateConfig



def populate_dict(d: Dict, default_d: Dict):
    """
    Populate `d` with values from `default_d` if they are not already present in `d`
    """
    for k, v in default_d.items():
        if k not in d:
            d[k] = v
        elif isinstance(v, dict):
            populate_dict(d[k], v)
    return d


def override_dict(d: Dict, overrides: Dict):
    """
    Recursively override values in `d` with values from `overrides`.
    """
    for k, v in overrides.items():
        if isinstance(v, dict):
            override_dict(d[k], v)
        else:
            d[k] = v
    return d


def prefix_dict(d: Dict, prefix: str):
    """
    Prefix keys in `d` with `prefix`.
    """
    return {f'{prefix}_{k}': v for k, v in d.items()}


def maybe_setup(config: InstantiateConfig):
    """
    Setup `config` if it is not None.
    """
    return config.setup() if config else None