# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import shutil
import sys
import warnings

from setuptools import find_packages, setup

setup(
    name='labelmaker',
    version='0.1',
    description='',
    packages=find_packages(include=['labelmaker*', 'scripts*']),
    install_requires=['numpy'],
    package_data={'': ['*.csv', '*.sh']},
    include_package_data=True,
)
