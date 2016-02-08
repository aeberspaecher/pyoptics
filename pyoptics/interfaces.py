#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tools for dielectric interfaces.
"""

import numpy as np
from numpy import cos, sin, arcsin, arctan2  # those allow complex arguments


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
        Refracted angle.
    """

    phi_out = arcsin(n_in*sin(phi_in)/n_out)
    # TODO: how to handle total internal reflection gracefully?
    # in that case n_in*sin(phi_in)/n_out > 1
    # ==> sin(phi_in) > n_out/n_in

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


def brewster(n_1, n_2):
    """Return Brewster's angle (p-wave reflectivity = 0).
    """

    b = arctan2(n_2, n_1)

    return b
