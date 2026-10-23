#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tools for dielectric interfaces.

Convention: fields vary as exp(i(k.r - omega*t)), i.e. a wave travelling in
+z is exp(+i*k_z*z) and absorbing media have refractive indices n + i*kappa
with kappa > 0. Accordingly, the refracted angle is chosen such that
Im(n_out*cos(phi_out)) >= 0: transmitted waves (evanescent ones under total
internal reflection, or waves in absorbing media) decay away from the
interface.
"""

from math import pi

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
        all input arguments are real. Of the two solutions of Snell's law, the
        one with Im(n_out*cos(phi_out)) >= 0 is returned, i.e. the transmitted
        wave exp(i*n_out*k_0*cos(phi_out)*z) decays (exp(-i*omega*t)
        convention).
    """

    phi_out = scimath.arcsin(n_in*sin(phi_in)/n_out)

    # arcsin() picks the growing evanescent wave under total internal
    # reflection; pi - phi_out is the other solution of Snell's law:
    growing = np.imag(n_out*cos(phi_out)) < 0
    phi_out = np.where(growing, pi - phi_out, phi_out)[()]

    return phi_out


# Sign convention for the amplitude coefficients as in Hecht, Optics, eqs.
# (4.42) - (4.45): at normal incidence r_s = (n_1 - n_2)/(n_1 + n_2) and
# r_p = (n_2 - n_1)/(n_1 + n_2).

def r_s(phi_1, n_1, n_2):
    """Reflection (wave amplitude) coefficient for s-polarized waves.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    r = (n_1*cos(phi_1) - n_2*cos(phi_2))/(n_1*cos(phi_1) + n_2*cos(phi_2))

    return r


def r_p(phi_1, n_1, n_2):
    """Reflection (wave amplitude) coefficient for p-polarized waves.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    r = (n_2*cos(phi_1) - n_1*cos(phi_2))/(n_2*cos(phi_1) + n_1*cos(phi_2))

    return r


def R_s(phi_1, n_1, n_2):
    """Energy reflection coefficient for s-polarized waves.
    """

    R = np.abs(r_s(phi_1, n_1, n_2))**2

    return R


def R_p(phi_1, n_1, n_2):
    """Energy reflection coefficient for p-polarized waves.
    """

    R = np.abs(r_p(phi_1, n_1, n_2))**2

    return R


def t_s(phi_1, n_1, n_2):
    """Transmission coefficient (wave amplitude) for s-polarized waves.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    t = (2*n_1*cos(phi_1))/(n_1*cos(phi_1) + n_2*cos(phi_2))

    return t


def t_p(phi_1, n_1, n_2):
    """Transmission coefficient (wave amplitude) for p-polarized waves.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    t = (2*n_1*cos(phi_1))/(n_2*cos(phi_1) + n_1*cos(phi_2))

    return t


def T_s(phi_1, n_1, n_2):
    """Energy transmission coefficient for s-polarized waves.

    Ratio of the z-components of the time averaged Poynting vectors. T_s = 0
    under total internal reflection.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    T = (np.real(n_2*cos(phi_2))/np.real(n_1*cos(phi_1))
         *np.abs(t_s(phi_1, n_1, n_2))**2)

    return T


def T_p(phi_1, n_1, n_2):
    """Energy transmission coefficient for p-polarized waves.

    Ratio of the z-components of the time averaged Poynting vectors. T_p = 0
    under total internal reflection.
    """

    phi_2 = refracted_angle(phi_1, n_1, n_2)
    T = (np.real(np.conj(n_2)*cos(phi_2))/np.real(np.conj(n_1)*cos(phi_1))
         *np.abs(t_p(phi_1, n_1, n_2))**2)

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
