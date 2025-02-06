import os
from pathlib import Path
from typing import Dict, Union, Tuple

import deeplake
import torch
from torchtyping import TensorType as TorchTensor


# STORAGE_DIR = Path(os.environ['STORAGE_DIR']) if 'STORAGE_DIR' in os.environ else None
STORAGE_DIR = Path('/home/user/zhang2/IML/data')

def load_dataset(name: str):
    """
    Load deeplake dataset from `name`.
    """
    return deeplake.load(STORAGE_DIR / name)


def dataset_name_from_path(path: Union[Path, str]) -> str:
    """ 
    Extract dataset name from `path` to deeplake dataset.
    """
    head = str(STORAGE_DIR)
    path = str(path)
    return path[len(head):].lstrip('/')


def metadata_prefix_from_tensor_name(tensor_name: str) -> str:
    """
    Extract metadata prefix from tensor name.
    """
    return '/'.join(tensor_name.split('/')[:-1])


def load_tensor_at(ds: deeplake.Dataset, tensor_name: str, index: int, format='torch') -> TorchTensor:
    """ 
    Load tensor from deeplake dataset into a torch tensor, at index.
    """
    assert format in ['torch', 'np']
    tensor = ds[tensor_name][index].numpy()
    if format == 'torch':
        tensor = torch.from_numpy(tensor)
    return tensor


def load_tensor(ds: deeplake.Dataset, tensor_name: str, tensor_slice: Tuple[int, int]=None, format='torch') -> TorchTensor:
    """ 
    Load tensor from deeplake dataset into a torch tensor, possibily a slice of tensor.
    """
    assert format in ['torch', 'np']
    tensor = ds[tensor_name].numpy()
    if tensor_slice:
        tensor = tensor[slice[0]:slice[1]]
    if format == 'torch':
        tensor = torch.from_numpy(tensor)
    return tensor
    

def load_tensor_group(ds: deeplake.Dataset, tensor_group_name: str, tensor_slice: Tuple[int, int]=None, format='torch') -> Dict:
    """
    Load tensor group from deeplake dataset into a dict of torch tensors, possibily slices of tensors.
    """
    batch = {}
    for k, v in ds.groups[tensor_group_name].tensors.items():
        if tensor_slice:
            tensor = tensor[slice[0]:slice[1]]
        if format == 'torch':
            tensor = torch.from_numpy(tensor)
        batch[k] = tensor
    return batch
