#!/usr/bin/env python
#-*- coding:utf-8 -*-

top = "."
out = "build"


def options(opt):
    opt.load("python")


def configure(conf):
    conf.load("python")
    conf.check_python_version((2, 6))
    conf.check_python_module("numpy")
    conf.check_python_module("scipy")
    conf.check_python_module("skimage")
    try:
        conf.check_python_module('matplotlib')
    except conf.errors.ConfigurationError:
        print('could not find matplotlib (ignored)')
    try:
        conf.check_python_module('pyfftw')
    except conf.errors.ConfigurationError:
        print('could not find pyfftw (ignored)')


def build(bld):
    bld.recurse("pyoptics")
