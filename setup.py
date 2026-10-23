#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Build the Cython extension(s). All metadata lives in pyproject.toml.
"""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


# OpenMP is used by wpm.pyx (prange). MSVC and GCC/clang spell it differently;
# Apple clang needs libomp and extra flags, so we build without it there.
if sys.platform == "win32":
    compile_args = ["/O2", "/openmp"]
    link_args = []
elif sys.platform == "darwin":
    compile_args = ["-O2"]
    link_args = []
else:
    compile_args = ["-O2", "-fopenmp"]
    link_args = ["-fopenmp"]

extensions = [
    Extension("pyoptics.wpm", ["pyoptics/wpm.pyx"],
              include_dirs=[np.get_include()],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              extra_compile_args=compile_args,
              extra_link_args=link_args),
]

compiler_directives = {
    "language_level": 3,
    "profile": False,
    "cdivision": True,
    "wraparound": False,
    "nonecheck": False,
}

setup(ext_modules=cythonize(extensions, compiler_directives=compiler_directives))
