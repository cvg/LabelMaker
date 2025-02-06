from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        ["*.pyx"], compiler_directives={'language_level': '3'}, annotate=True
    ),
    include_dirs=[np.get_include()],
)

# python setup.py build_ext --inplace