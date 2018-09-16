#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tools for dielectric interfaces.
"""

import numpy as np
from numpy import cos, sin, arcsin, arctan2  # those allow complex arguments
import numpy.lib.scimath as scimath

# TODO: can we include conductivity where applicable? or can we do that in utils (e.g. with n = n(epsilon, sigma)?


def refracted_angle(phi_in, n_in, n_out):
    """Implement Snell's law.

    Parameters
    ----------
    phi_in : double
        Incident angle.
    n_in, n_out : double
        (Complex) refractive index for incoming and outgoing medium.

    Returns
    -------
    phi_out : double
        Refracted angle. Complex phi_out indicates total internal reflection if
        all input arguments are real.
    """

    phi_out = scimath.arcsin(n_in*sin(phi_in)/n_out)

    return phi_out


def r_s(phi_1, n_1, n_2):
    """Reflection (wave amplitude) coefficient for s-polarized waves.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    r = (n_2*cos(phi_1) - n_1*cos(phi_2))/(n_2*cos(phi_1) + n_1*cos(phi_2))

    return r


def r_p(phi_1, n_1, n_2):
    """Reflection (wave amplitude) coefficient for p-polarized waves.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    r = (n_1*cos(phi_1) - n_2*cos(phi_2))/(n_1*cos(phi_1) + n_2*cos(phi_2))

    return r


def R_s(phi_1, n_1, n_2):
    """Energy reflection coefficient for s-polarized waves.
    """

    R = np.abs(r_s(phi_1, n_1, n_2))**2

    return R


def R_p(phi_1, n_1, n_2):
    """Energy reflection coefficient for s-polarized waves.
    """

    R = np.abs(r_p(phi_1, n_1, n_2))**2

    return R


def t_s(phi_1, n_1, n_2):
    """Transmission coefficient (wave amplitude) for s-polarized waves.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    t = (2*n_2*cos(phi_1))/(n_2*cos(phi_1) + n_1*cos(phi_2))

    return t


def t_p(phi_1, n_1, n_2):
    """Transmission coefficient (wave amplitude) for p-polarized waves.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    t = (2*n_2*cos(phi_1))/(n_1*cos(phi_1) + n_2*cos(phi_2))

    return t


def T_s(phi_1, n_1, n_2):
    """Energy transmission coefficient for s-polarized waves.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    T = (4*n_1*n_2*cos(phi_1)*cos(phi_2))/(n_2*cos(phi_1) + n_1*cos(phi_2))**2

    return T


def T_p(phi_1, n_1, n_2):
    """Energy transmission coefficient for s-polarized waves.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    T = (4*n_1*n_2*cos(phi_1)*cos(phi_2))/(n_1*cos(phi_1) + n_2*cos(phi_2))**2

    return T


def brewster_angle(n_1, n_2):
    """Return Brewster's angle (p-wave reflectivity = 0).
    """

    b = arctan2(n_2, n_1)

    return b


def critical_angle(n_1, n_2):
    """Return critical angle for total internal reflection.
    """

    phi_c = scimath.arcsin(n_2/n_1)
    # TODO: how to handle n_2 > n_1? does this treatment here make sense?

    return phi_c
