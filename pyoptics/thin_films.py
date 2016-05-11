#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Thin film reflection and transmission coefficients.
"""

# Literature:
# Zangwill, Modern Electrodynamics, CUP, 2013
# Brooker, Modern Classical Optics, OUP, 2002
# McLeod, Thin Film Optical Filters, Institute of Physics Publishing, 2001

from copy import copy
from functools import reduce  # Python 3 compatibility

import numpy as np
from numpy.lib.scimath import sqrt as complex_sqrt
from numpy import sin, cos

from utils import Z_0 as Z_vac, wavenumber


def _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k):
    """Extend n and k arrays by embedding medium and substrate values.
    """

    n_extended = np.asarray([embedding_n] + list(n) + [substrate_n])
    k_extended = np.asarray([embedding_k] + list(k) + [substrate_k])

    return n_extended, k_extended


def _impedances(complex_n):
    """Return impedances for stack defined by (complex) refractive indices.
    """

    return Z_vac/complex_n


def _cos_theta(theta_0, complex_n):
    """Compute cosines of complex propagation angles in all media of a given
    stack.

    Parameters
    ----------
    theta_0 : double
        Angle in embedding medium.
    complex_n : array
        Refractive indices for whole stack.

    Returns
    -------
    angles : array
        Array cos(theta_i).
    """

    # cos(theta) = sqrt(1 - sin(theta)); sin(theta) is found from Snell's law:
    cos_theta = complex_sqrt(1.0-(complex_n[0]/complex_n*sin(theta_0))**2)

    # NOTE: n_i sin(theta_i) = n_j sin(theta_j) for any j, i! thus, we can use
    # any layer (here: the embedding medium) to compute angles in any other
    # layer.

    return cos_theta


def _tilted_Z(Z_in, cos_theta, pol):
    """Compute 'tilted admitances' for a stack.

    Parameters
    ----------
    Z_in, cos_theta : arrays
        (Complex) impedances of layer stack and cosines of propagation angles.
    pol : string
        One of {'s','TE'; 'p', 'TM'}.

    Returns
    -------
    Z_tilted : array
        Polarization dependent tilted impedances.
    """

    Z = copy(Z_in)
    # compute 'tilted' addmittances:
    if pol in ("s", "TE"):
        Z *= cos_theta  # McLeod, eq. (8.5)
    elif pol in ("p", "TM"):
        Z /= cos_theta  # McLeod, eq. (8.6)
    else:
        raise ValueError("'pol' must be one of 's'/'TE', 'p'/'TM'")

    return Z


def r_t_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
               substrate_k, pol, wavelength, incoherent):
    """Reflection and transmission coefficients for a thin film stack.

    Parameters
    ----------
    angle : double
        Incident angle in embedding medium.
    n , k : arrays
        Real and imaginary parts of the refractive indices of the thin film stack.
    t : arrays
        Thicknesses of the layers.
    embedding_n, embedding_k, substrate_n, substrate_k : double
        Real and imaginary parts of the embedding medium's and the substrate's
        refractive index. Both the embedding medium and the substrate are taken
        be infinitely extend.
    pol : string
        "s" or "p" polarization. Also accepted: "TE" or "TM".

    Returns
    -------
    r, t : double
        Reflection and transmission amplitude of the stack.
    """

    # TODO: make checks more verbose and raise sensible exceptions
    assert len(n) >= 1
    assert len(k) >= 1
    assert len(t) == len(n) == len(k)

    if(incoherent is None):
        incoherent = np.zeros(len(n)+2, dtype=np.bool)  # array of all False
    else:
        incoherent = np.array([False] + incoherent + [False])

    n, k = _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k)  # extend n, k arrays by embedding medium and substrate
    complex_n = n + 1j*k
    Z = _impedances(complex_n)  # complex impedance for all media
    cos_theta = _cos_theta(angle, complex_n)  # complex propagation angles in all media
    Z = _tilted_Z(Z, cos_theta, pol)  # McLeod's 'tilted admittances'

    # construct Zangwill's transfer matrices for all layers:
    M = []
    for j in range(1, len(n) - 1):
        phi_j = t[j-1]*wavenumber(wavelength, complex_n[j])*cos_theta[j]  # 'phase' accumulated in layer j

        if(incoherent[j]):  # inchoherent layers lose phase information
            phi_j = np.imag(phi_j)  # only damping remains

        M_curr = np.array([[cos(phi_j), -1j*Z[j]*sin(phi_j)],
                           [-1j/Z[j]*sin(phi_j), cos(phi_j)]
                          ]
                         )

        #if(incoherent[j]):  # inchoherent layers lose phase information
            #M_curr = np.abs(M_curr)  # only damping remains
            # TODO: how to kill a phase in the incoherent case? build a test case! single film over wavelength or such...

        M.append(M_curr)

    M_total = reduce(np.dot, M)  # multiply all M matrices

    A, B, C, D = M_total.flatten()  # unpack layer stack matrix

    # solve Zangwill's eq. (17.92) for r and t:
    r = (A*Z[-1] + B - Z[0]*(C*Z[-1] + D))/(A*Z[-1] + B + Z[0]*(C*Z[-1] + D))
    t = 2*Z[-1]/(A*Z[-1] + B + Z[0]*(C*Z[-1] + D))

    return r, t


def R_T_A_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
                 substrate_k, pol, wavelength, incoherent=None):
    """Same as r_t_coeffs, but for intensity reflectance and transmittance.
    Also return 'absorbtance' such that R + T + A = 1.

    Note
    ----
    A = 0 can only be expected for losless media under the condition that the
    embeddig medium is equal to the substrate.
    """

    r, t = r_t_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
                      substrate_k, pol, wavelength, incoherent)

    # prepare data for T correction:
    n_extended, k_extended = _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k)
    complex_n = n_extended + 1j*k_extended
    Z = _impedances(complex_n)  # complex impedance for all media
    cos_theta = _cos_theta(angle, complex_n)
    Z = _tilted_Z(Z, cos_theta, pol)

    R = np.abs(r)**2
    T = np.real(Z[-1])/np.real(Z[0])*np.abs(t)**2
    A = 1.0 - R - T

    return R, T, A
