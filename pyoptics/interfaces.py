#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Fresnel coefficients of a single planar interface between two media.

All functions take the angle of incidence ``phi_1`` (in radians, measured
from the surface normal) and the refractive indices ``n_1`` of the incident
and ``n_2`` of the second medium. Every argument may be a number or an array;
arrays broadcast against each other with the usual NumPy rules, so a sweep
over angles and wavelengths is a single call.

Conventions
-----------
Fields vary as exp(i(k.r - omega*t)): a wave travelling in +z is
exp(+i*k_z*z), and absorbing media have refractive indices n + i*kappa with
kappa > 0. The normal component of the transmitted wave vector,
k_0*n_2*cos(phi_2), is chosen with a non-negative imaginary part, so that
transmitted waves (evanescent ones under total internal reflection, or waves
in absorbing media) decay away from the interface.

The amplitude coefficients are ratios of the full electric field amplitudes
with the sign convention of Hecht, Optics, eqs. (4.42) - (4.45): at normal
incidence r_s = (n_1 - n_2)/(n_1 + n_2) and r_p = (n_2 - n_1)/(n_1 + n_2).
The p-wave reference directions are such that r_p = tan(phi_1 - phi_2)/
tan(phi_1 + phi_2) for dielectrics, which vanishes at Brewster's angle.

The energy coefficients R and T are ratios of the normal components of the
time averaged Poynting vectors. They are well defined for a lossless incident
medium (real ``n_1``) only; then R + T = 1 also for an absorbing second
medium, T being the power that enters it.

References
----------
M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), section 1.5.

E. Hecht, "Optics", 5th ed., Pearson (2017), section 4.6.

A. Zangwill, "Modern Electrodynamics", Cambridge University Press (2013).

R. M. A. Azzam and N. M. Bashara, "Ellipsometry and Polarized Light",
North-Holland (1977).
"""

from math import pi

import numpy as np
import numpy.lib.scimath as scimath

__all__ = ["refracted_angle", "r_s", "r_p", "t_s", "t_p", "R_s", "R_p", "T_s",
           "T_p", "brewster_angle", "critical_angle"]


def refracted_angle(phi_1, n_1, n_2):
    """Angle of the refracted wave from Snell's law.

    Parameters
    ----------
    phi_1 : number or array
        Angle of incidence in radians.
    n_1, n_2 : numbers or arrays
        (Complex) refractive indices of the incident and the second medium.

    Returns
    -------
    phi_2 : number or array
        Refracted angle in radians, solving n_1*sin(phi_1) = n_2*sin(phi_2).
        Real for propagating waves between lossless media, complex under total
        internal reflection or for absorbing media. Of the two solutions of
        Snell's law, the one with Im(n_2*cos(phi_2)) >= 0 is returned, i.e. the
        transmitted wave exp(i*n_2*k_0*cos(phi_2)*z) decays (exp(-i*omega*t)
        convention).
    """

    phi_2 = scimath.arcsin(n_1*np.sin(phi_1)/n_2)

    # arcsin() picks the growing evanescent wave under total internal
    # reflection; pi - phi_2 is the other solution of Snell's law:
    growing = np.imag(n_2*np.cos(phi_2)) < 0
    phi_2 = np.where(growing, pi - phi_2, phi_2)[()]

    return phi_2


def _normal_indices(phi_1, n_1, n_2):
    """Normal components q_j = n_j*cos(phi_j) of the wave vectors over k_0.

    q_2 = sqrt(n_2**2 - (n_1*sin(phi_1))**2) is computed directly instead of
    via the (possibly complex) refracted angle, on the branch Im(q_2) >= 0
    chosen by refracted_angle(). It stays real for propagating waves between
    lossless media, so that the Fresnel coefficients are real there.
    """

    phi_1, n_1, n_2 = np.asarray(phi_1), np.asarray(n_1), np.asarray(n_2)

    q_1 = n_1*np.cos(phi_1)
    q_2 = scimath.sqrt(n_2**2 - (n_1*np.sin(phi_1))**2)
    q_2 = np.where(np.imag(q_2) < 0, -q_2, q_2)

    return q_1, q_2, n_1, n_2


def r_s(phi_1, n_1, n_2):
    """Reflection coefficient (field amplitude) for s-polarized waves.

    Parameters
    ----------
    phi_1 : number or array
        Angle of incidence in radians.
    n_1, n_2 : numbers or arrays
        (Complex) refractive indices of the incident and the second medium.

    Returns
    -------
    r : number or array
        (n_1*cos(phi_1) - n_2*cos(phi_2))/(n_1*cos(phi_1) + n_2*cos(phi_2)).
        Real for propagating waves between lossless media, complex otherwise.
    """

    q_1, q_2, _, _ = _normal_indices(phi_1, n_1, n_2)

    return ((q_1 - q_2)/(q_1 + q_2))[()]


def r_p(phi_1, n_1, n_2):
    """Reflection coefficient (field amplitude) for p-polarized waves.

    Parameters
    ----------
    phi_1 : number or array
        Angle of incidence in radians.
    n_1, n_2 : numbers or arrays
        (Complex) refractive indices of the incident and the second medium.

    Returns
    -------
    r : number or array
        (n_2*cos(phi_1) - n_1*cos(phi_2))/(n_2*cos(phi_1) + n_1*cos(phi_2)).
        Real for propagating waves between lossless media, complex otherwise.
    """

    q_1, q_2, n_1, n_2 = _normal_indices(phi_1, n_1, n_2)
    # multiplied by n_1*n_2 to avoid cos(phi_2):
    a, b = n_2**2*q_1, n_1**2*q_2

    return ((a - b)/(a + b))[()]


def t_s(phi_1, n_1, n_2):
    """Transmission coefficient (field amplitude) for s-polarized waves.

    Parameters
    ----------
    phi_1 : number or array
        Angle of incidence in radians.
    n_1, n_2 : numbers or arrays
        (Complex) refractive indices of the incident and the second medium.

    Returns
    -------
    t : number or array
        2*n_1*cos(phi_1)/(n_1*cos(phi_1) + n_2*cos(phi_2)). Note t_s = 1 + r_s.
    """

    q_1, q_2, _, _ = _normal_indices(phi_1, n_1, n_2)

    return (2*q_1/(q_1 + q_2))[()]


def t_p(phi_1, n_1, n_2):
    """Transmission coefficient (field amplitude) for p-polarized waves.

    Parameters
    ----------
    phi_1 : number or array
        Angle of incidence in radians.
    n_1, n_2 : numbers or arrays
        (Complex) refractive indices of the incident and the second medium.

    Returns
    -------
    t : number or array
        2*n_1*cos(phi_1)/(n_2*cos(phi_1) + n_1*cos(phi_2)). Note
        n_2*t_p = n_1*(1 + r_p).
    """

    q_1, q_2, n_1, n_2 = _normal_indices(phi_1, n_1, n_2)

    return (2*n_1*n_2*q_1/(n_2**2*q_1 + n_1**2*q_2))[()]


def R_s(phi_1, n_1, n_2):
    """Reflectance (power reflection coefficient) for s-polarized waves.

    Parameters
    ----------
    phi_1 : number or array
        Angle of incidence in radians.
    n_1, n_2 : numbers or arrays
        (Complex) refractive indices of the incident and the second medium.

    Returns
    -------
    R : number or array
        ``|r_s|**2``.
    """

    return np.abs(r_s(phi_1, n_1, n_2))**2


def R_p(phi_1, n_1, n_2):
    """Reflectance (power reflection coefficient) for p-polarized waves.

    Parameters
    ----------
    phi_1 : number or array
        Angle of incidence in radians.
    n_1, n_2 : numbers or arrays
        (Complex) refractive indices of the incident and the second medium.

    Returns
    -------
    R : number or array
        ``|r_p|**2``.
    """

    return np.abs(r_p(phi_1, n_1, n_2))**2


def T_s(phi_1, n_1, n_2):
    """Transmittance (power transmission coefficient) for s-polarized waves.

    Ratio of the normal components of the time averaged Poynting vectors of
    the transmitted and the incident wave. T_s = 0 under total internal
    reflection, and R_s + T_s = 1 for a lossless incident medium.

    Parameters
    ----------
    phi_1 : number or array
        Angle of incidence in radians.
    n_1, n_2 : numbers or arrays
        Refractive indices of the incident (real) and the second (complex)
        medium.

    Returns
    -------
    T : number or array
        Re(n_2*cos(phi_2))/Re(n_1*cos(phi_1))*|t_s|**2.
    """

    q_1, q_2, n_1, n_2 = _normal_indices(phi_1, n_1, n_2)
    t = 2*q_1/(q_1 + q_2)

    return (np.real(q_2)/np.real(q_1)*np.abs(t)**2)[()]


def T_p(phi_1, n_1, n_2):
    """Transmittance (power transmission coefficient) for p-polarized waves.

    Ratio of the normal components of the time averaged Poynting vectors of
    the transmitted and the incident wave. T_p = 0 under total internal
    reflection, and R_p + T_p = 1 for a lossless incident medium.

    Parameters
    ----------
    phi_1 : number or array
        Angle of incidence in radians.
    n_1, n_2 : numbers or arrays
        Refractive indices of the incident (real) and the second (complex)
        medium.

    Returns
    -------
    T : number or array
        Re(conj(n_2)*cos(phi_2))/Re(conj(n_1)*cos(phi_1))*|t_p|**2.
    """

    q_1, q_2, n_1, n_2 = _normal_indices(phi_1, n_1, n_2)
    t = 2*n_1*n_2*q_1/(n_2**2*q_1 + n_1**2*q_2)
    # conj(n_j)*cos(phi_j) = conj(n_j)*q_j/n_j:
    eta_1 = np.real(np.conj(n_1)*q_1/n_1)
    eta_2 = np.real(np.conj(n_2)*q_2/n_2)

    return (eta_2/eta_1*np.abs(t)**2)[()]


def brewster_angle(n_1, n_2):
    """Brewster's angle, at which the p-wave reflectance vanishes.

    Parameters
    ----------
    n_1, n_2 : numbers or arrays
        Real refractive indices of the incident and the second medium.

    Returns
    -------
    phi_B : number or array
        arctan(n_2/n_1) in radians. Brewster's angle exists for both
        directions of incidence; going from the denser medium, it lies below
        the critical angle.
    """

    return np.arctan2(n_2, n_1)


def critical_angle(n_1, n_2):
    """Critical angle for total internal reflection.

    Parameters
    ----------
    n_1, n_2 : numbers or arrays
        Real refractive indices of the incident and the second medium.

    Returns
    -------
    phi_c : number or array
        arcsin(n_2/n_1) in radians where n_1 >= n_2, NaN where n_1 < n_2 (no
        total internal reflection when going into a denser medium).
    """

    ratio = np.asarray(n_2/np.asarray(n_1, dtype=float), dtype=float)
    with np.errstate(invalid="ignore"):
        phi_c = np.where(ratio <= 1, np.arcsin(np.minimum(ratio, 1)), np.nan)

    return phi_c[()]
