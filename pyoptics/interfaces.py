#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tools for dielectric interfaces.
"""

import numpy as np
from numpy import cos, sin, arcsin  # those allow complex arguments


def refracted_angle(phi_in, n_in, n_out):

    phi_out = arcsin(n_in*sin(phi_in)/n_out)

    return phi_out


def r_s(phi, n_1, n_2):
    pass


def r_p(phi, n_1, n_2):
    pass


def R_s(phi, n_1, n_2):
    pass


def R_p(phi, n_1, n_2):
    pass
