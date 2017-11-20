#!/usr/bin/env python
#-*- coding:utf-8 -*-

top = "."
out = "build"


def options(opt):
    opt.load("compiler_c")
    opt.load("python")
    opt.load("cython")


def configure(conf):
    conf.load("python")
    conf.check_python_version((2, 7))
    conf.check_python_module("numpy")
    conf.check_python_module("scipy", condition="ver>=num(0, 14, 0)")
    #conf.check_python_module("skimage")
    try:
        conf.check_python_module('matplotlib')
    except conf.errors.ConfigurationError:
        print('could not find matplotlib (ignored)')
    try:
        conf.check_python_module('pyfftw')
    except conf.errors.ConfigurationError:
        print('could not find pyfftw (ignored)')

    conf.load("compiler_c")
    conf.load("cython")
    conf.check_python_headers()

    # make sure the NumPy headers are in the include path:
    import numpy as np
    numpy_header_path = np.get_include()
    conf.env.append_value("INCLUDES", numpy_header_path)
    print("Using NumPy headers in %s"%numpy_header_path)  # TODO: use waf logging instead

    conf.env.CFLAGS = ["-O2", "-fPIC", "-fopenmp", "-march=native", "-ftree-vectorizer-verbose=1"]
    conf.env.LINKFLAGS = ["-fopenmp"]
    conf.env.CYTHONFLAGS = ["--directive", "profile=False",
                            "--directive", "cdivision=True",
                            #"--directive", "boundscheck=False",
                            "--directive", "wraparound=False",
                            "--directive", "nonecheck=False"]


def build(bld):
    bld.recurse("pyoptics")
