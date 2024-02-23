# This code gives a valid set of cuda, pytorch, torchvision and gcc version
# This program takes the desired cuda version, desired pytorch version, current nvidia driver cuda version as input
import argparse
import os
import re
import sys

from packaging.version import parse

CUDA_VERSIONS = ["11.3", "11.5", "11.6", "11.7", "11.8", "12.1"]
PYTORCH_VERSIONS = [
    "1.10.0", "1.10.1", "1.10.2", "1.11.0", "1.12.0", "1.12.1", "1.13.0",
    "1.13.1", "2.0.0", "2.0.1", "2.1.0"
]
CUDA_PYTORCH_COMPATIBILITY = {
    "11.3": [
        "1.10.0",  # need python 3.9
        "1.10.1",  # does not have mmcv-full
        "1.10.2",  # does not have mmcv-full
        "1.11.0",
        "1.12.0",
        "1.12.1",  # does not have mmcv-full
    ],
    "11.5": ["1.11.0"],
    "11.6": [
        "1.12.0",  # does not have mmcv-full
        "1.12.1",  # does not have mmcv-full
        "1.13.0",  # mmcv-full no 1.6.2
        "1.13.1",  # does not have mmcv-full
    ],
    "11.7": [
        "1.13.0",  # does not have mmcv-full
        "1.13.1",  # does not have mmcv-full
        "2.0.0",  # does not have mmcv-full
        "2.0.1",  # does not have mmcv-full
    ],
    "11.8": [
        "2.0.0",
        "2.0.1",
        "2.1.0",
    ],
    "12.1": ["2.1.0"],
}
PTTORCH_TORCHVISION_CORRESPONDENCE = {
    "2.1.0": "0.16.0",
    "2.0.1": "0.15.2",
    "2.0.0": "0.15.0",
    "1.13.1": "0.14.1",
    "1.13.0": "0.14.0",
    "1.12.1": "0.13.1",
    "1.12.0": "0.13.0",
    "1.11.0": "0.12.0",
    "1.10.2": "0.11.3",
    "1.10.1": "0.11.2",
    "1.10.0": "0.11.0",
}
CUDA_MAX_GCC_VERSION = {
    "11.3": "10.4.0",
    "11.5": "11.4.0",
    "11.6": "11.4.0",
    "11.7": "11.4.0",
    "11.8": "11.4.0",
    "12.1": "12.2.0",
}
CUDA_MIN_GCC_VERSION = "8.5.0"
CONDA_AVAIL_CUDA_MAPPING = { # use the higher version
    "11.3": "11.3.1",
    "11.5": "11.5.1",
    "11.6": "11.6.2",
    "11.7": "11.7.1",
    "11.8": "11.8.0",
    "12.1": "12.1.1",
}
CONDA_AVAIL_GCC_VERSION = [
    "12.2.0", "12.1.0", "11.4.0", "11.3.0", "11.2.0", "11.1.0", "10.4.0",
    "10.3.0", "9.5.0", "9.4.0", "8.5.0"
]
CONDA_AVAIL_OPENBLAS_VERSION = [
    "0.3.21", "0.3.20", "0.3.18", "0.3.17", "0.3.13", "0.3.10", "0.3.6",
    "0.3.3", "0.3.2", "0.2.20"
]

OPEN3D_URLS = {
    "3.6":
        "https://github.com/cvg/open3d-manylinux2014/releases/download/0.17.0/open3d_cpu-0.17.0-cp36-cp36m-manylinux_2_17_x86_64.whl",
    "3.7":
        "https://github.com/cvg/open3d-manylinux2014/releases/download/0.17.0/open3d_cpu-0.17.0-cp37-cp37m-manylinux_2_17_x86_64.whl",
    "3.8":
        "https://github.com/cvg/open3d-manylinux2014/releases/download/0.17.0/open3d_cpu-0.17.0-cp38-cp38-manylinux_2_17_x86_64.whl",
    "3.9":
        "https://github.com/cvg/open3d-manylinux2014/releases/download/0.17.0/open3d_cpu-0.17.0-cp39-cp39-manylinux_2_17_x86_64.whl",
    "3.10":
        "https://github.com/cvg/open3d-manylinux2014/releases/download/0.17.0/open3d_cpu-0.17.0-cp310-cp310-manylinux_2_17_x86_64.whl",
}

if __name__ == "__main__":

  try:
    output_stream = os.popen('nvidia-smi | grep "CUDA Version:"')
    driver_cuda_version = parse(
        re.search(r"CUDA Version:( )*[0-9]+\.[0-9]",
                  output_stream.read()).group().split(':')[-1].strip())
  except:
    driver_cuda_version = None

  print(f"Found nvidia driver's cuda version: {driver_cuda_version} .")

  parser = argparse.ArgumentParser()
  parser.add_argument("--target_cuda_version", type=str)
  parser.add_argument("--target_torch_version", type=str)
  parser.add_argument("--target_gcc_version", type=str)
  args = parser.parse_args()

  print(args)

  # check CUDA
  target_cuda_version: str = None
  if args.target_cuda_version != 'unset':
    try:
      parse(args.target_cuda_version)
    except:
      raise ValueError("The cuda version should be in format of x.x !")

    assert args.target_cuda_version in CUDA_VERSIONS, f"The specified cuda version {args.target_cuda_version} is not supported, please use CUDA: {', '.join(CUDA_VERSIONS)}"

    target_cuda_version = args.target_cuda_version

  else:
    if driver_cuda_version is None:
      raise ValueError(
          "No CUDA driver detected on your machine, and no target cuda toolkit specified!"
      )

    for ver in CUDA_VERSIONS[::-1]:
      if parse(ver) <= driver_cuda_version:
        print(f"CUDA version not specified, using highes possible cuda: {ver}")
        target_cuda_version = ver
        break

    if target_cuda_version is None:
      raise NotImplementedError(
          f"The cuda version ({driver_cuda_version}) of this machine is too old!"
      )

  # check pytorch
  target_torch_version: str = None
  if args.target_torch_version != 'unset':
    try:
      parse(args.target_torch_version)
    except:
      raise ValueError("The pytorch version should be in format of x.x !")

    assert args.target_torch_version in PYTORCH_VERSIONS, f"The specified torch version {args.target_torch_version} is not supported, please use PyTorch: {', '.join(PYTORCH_VERSIONS)}"

    assert args.target_torch_version in CUDA_PYTORCH_COMPATIBILITY[
        target_cuda_version], f"The specified torch version {args.target_torch_version} is not supported by the selected version of cuda {target_cuda_version}, please use PyTorch: {', '.join(CUDA_PYTORCH_COMPATIBILITY[target_cuda_version])}"

    target_torch_version = args.target_torch_version

  else:
    for ver in PYTORCH_VERSIONS[::-1]:
      if ver in CUDA_PYTORCH_COMPATIBILITY[target_cuda_version]:
        print(f"PyTorch version not specified, using highes possible: {ver}")
        target_torch_version = ver
        break

  # check gcc
  target_gcc_version: str = None
  if args.target_gcc_version != 'unset':
    try:
      parse(args.target_gcc_version)
    except:
      raise ValueError("The GCC version should be in format of x.x !")

    assert parse(args.target_gcc_version) >= parse(
        CUDA_MIN_GCC_VERSION
    ), f"The target GCC compiler version {args.target_gcc_version} should be higher than {CUDA_MIN_GCC_VERSION}"

    assert parse(args.target_gcc_version) <= parse(
        CUDA_MAX_GCC_VERSION[target_cuda_version]
    ), f"The target GCC compiler version {args.target_gcc_version} should be lower than {CUDA_MAX_GCC_VERSION[target_cuda_version]}"

    assert args.target_gcc_version in CONDA_AVAIL_GCC_VERSION, f"The target GCC compiler version {args.target_gcc_version} should be one of {CONDA_AVAIL_GCC_VERSION}"

    target_gcc_version = args.target_gcc_version

  else:
    # use the highest possible gcc compiler
    target_gcc_version = CUDA_MAX_GCC_VERSION[target_cuda_version]

  target_torchvision_version = PTTORCH_TORCHVISION_CORRESPONDENCE[
      target_torch_version]

  # detect python version
  python_version = str(sys.version_info.major) + '.' + str(
      sys.version_info.minor)
  target_open3d_url = OPEN3D_URLS[python_version]

  with open(os.path.join(os.path.dirname(__file__), 'INSTALLED_VERSIONS.sh'),
            'w') as f:
    f.write(
        f'export INSTALLED_CUDA_VERSION={CONDA_AVAIL_CUDA_MAPPING[target_cuda_version]}\n'
    )
    f.write(
        f'export INSTALLED_CUDA_ABBREV={"cu" + "".join(target_cuda_version.split("."))}\n'
    )
    f.write(f'export INSTALLED_PYTORCH_VERSION={target_torch_version}\n')
    f.write(f'export INSTALLED_GCC_VERSION={target_gcc_version}\n')
    f.write(
        f'export INSTALLED_TORCHVISION_VERSION={target_torchvision_version}\n')
    f.write(f'export INSTALLED_OPEN3D_URL={target_open3d_url}\n')
