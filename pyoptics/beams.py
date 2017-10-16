#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Analytical beam definitions.

Includes
- Gaussian beams
- plane waves
"""

from itertools import product
from math import factorial, pi, sin, sqrt

import numpy as np
from scipy.special import eval_genlaguerre, eval_hermite
from utils import sgn, wavenumber


# plane waves:
def plane_wave(A, dir_vec, x, y, z, wl, n=1.0):
    """Return plane wave defined by direction vector.

    Parameters
    ----------
    A : number
        Amplitude (scalar or vector-valued).
    dir_vec : array
        Dimensionless direction vector of unit length. Relates to wave vector
        by k_vector = n*2*pi/lambda dircetion_vector.
    x, y : arrays
        x, y coordinate arrays.
    z : number
        z position.
    wl : number
        Vacuum wavelength.
    n : number, optional
        Refractive index.

    Returns
    -------
    E : array
        Sampled electric field E(x, y, z; k).
    """

    XX, YY = np.meshgrid(x, y)
    E = A*np.exp(1j*2*n*pi/wl*(dir_vec[0]*XX + dir_vec[1]*YY + dir_vec[2]*z))

    return E


def spherical_wave(x, y, z, wavelength, x0, y0, z0, U0=1.0, n=1.0):
    """Return spherical wave

    Parameters
    ----------
    x, y : arrays
    z : number
    wl : number
    x0, y0, z0 : numbers

    Returns
    -------
    U : array
    """

    # TODO: note on convergent/diverging spherical wave?

    X, Y = np.meshgrid(x, y)
    k = wavenumber(wavelength, n)
    R = np.sqrt((X-x0)**2 + (Y-y0)**2 + (z-z0)**2)
    sign = sgn(z-z0)
    vals = U0*(z-z0)/R*np.exp(1j*sign*k*R)

    return vals


# Gaussian beams:
def z_r_from_divergence_angle(w0, theta, n, wl):
    return w0/(n*sin(theta))


def gauss_laguerre(p, l, x, y, z, w_0, z_r, wl):
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    k = wavenumber(wl)  # TODO: n dependency

    mode_number = abs(l) + 2*p
    phi_z = (mode_number + 1)*np.arctan2(z, z_r)  # Guoy phase
    z_over_z_r = z/z_r
    w_z = w_0*sqrt(1 + z_over_z_r**2)

    const = sqrt(2*factorial(p)/(pi*factorial(abs(l) + p)))
    L = eval_genlaguerre(p, abs(l), 2*r**2/w_z**2)

    if z != 0.0:
        R_z = z*(1 + (z_r/z)**2)
        field = (const/w_z*(sqrt(2)*r/w_z)**abs(l)*np.exp(-r**2/w_z**2)*L
                 *np.exp(-1j*k*r**2/(2*R_z))*np.exp(-1j*(l*phi - k*z + phi_z))
                )
    else:
        field = (const/w_z*(sqrt(2)*r/w_z)**abs(l)*np.exp(-r**2/w_z**2)*L
                 *np.exp(-1j*(l*phi + phi_z))
                )

    return field


def gauss_hermite(l, m, x, y, z, w_0, z_r, wl):
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    k = wavenumber(wl)  # TODO: n dependency

    mode_number = l + m
    phi_z = (mode_number + 1)*np.arctan2(z, z_r)  # Guoy phase
    z_over_z_r = z/z_r
    w_z = w_0*sqrt(1 + z/z_r**2)

    Hx = eval_hermite(l, sqrt(2)*X/w_z)
    Hy = eval_hermite(m, sqrt(2)*Y/w_z)
    E0 = sqrt( sqrt(2/pi)/(2**l*factorial(l)*w_0) ) * sqrt( sqrt(2/pi)/(2**m*factorial(m)*w_0) )

    if np.abs(z) >= 1E-14:  # TODO: do not hardcode an epsilon
        R_z = z*(1.0 + z_r/z**2)
        field = E0*w_0/w_z*Hx*Hy*np.exp(-r**2/w_z**2)*np.exp(-1j*k*r**2/(2*R_z))*np.exp(-1j*(-k*z + phi_z))
    else:
        field = E0*w_0/w_z*Hx*Hy*np.exp(-r**2/w_z**2)*np.exp(-1j*(-k*z + phi_z))

    return field

