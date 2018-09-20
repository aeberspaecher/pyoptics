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
from itertools import product

import numpy as np
from numpy.lib.scimath import sqrt as complex_sqrt
from numpy import sin, cos

from .utils import Z_0 as Z_vac, wavenumber, complex_average


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
    """Compute 'tilted impedances' for a stack.

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

    # NOTE: source is McLeod's Thin Film Optical Filters, 3rd edition. McLeod
    # uses admittances, whereas our code uses impedances (cmp. Zangwil).
    # Admittance is the reciprocal value of impedance.
    # Using admittance y: H = Y*E
    # Using impedance: E = Z*H = 1/Y*H

    Z = copy(Z_in)
    # compute 'tilted' impedance from McLeod's tilted admittances:
    if pol in ("s", "TE"):
        Z /= cos_theta  # McLeod 3rd ed, eq. (8.5)
    elif pol in ("p", "TM"):
        Z *= cos_theta  # McLeod 3rd ed, eq. (8.6)
    else:
        raise ValueError("'pol' must be one of 's'/'TE', 'p'/'TM'")

    return Z


def r_t_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
               substrate_k, pol, wavelength, phi0=None):
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

    n, k = _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k)  # extend n, k arrays by embedding medium and substrate
    complex_n = n + 1j*k
    Z = _impedances(complex_n)  # complex impedance for all media
    cos_theta = _cos_theta(angle, complex_n)  # complex propagation angles in all media
    Z = _tilted_Z(Z, cos_theta, pol)  # McLeod's 'tilted admittances'

    if phi0 is None:  # if no additional phases are given, assume zero additional phases
        phi0 = np.zeros(len(n))
    else:
        phi0 = np.array([0.] + list(phi0) + [0.])  # additional zeros for substrate and embedding medium

    # construct Zangwill's transfer matrices for all layers:
    M = []
    for j in range(1, len(n) - 1):
        phi_j = t[j-1]*wavenumber(wavelength, complex_n[j])*cos_theta[j] + phi0[j]  # 'phase' accumulated in layer j

        M_curr = np.array([[cos(phi_j), -1j*Z[j]*sin(phi_j)],
                           [-1j/Z[j]*sin(phi_j), cos(phi_j)]
                          ]
                         )
        M.append(M_curr)

    M_total = reduce(np.dot, M)  # multiply all M matrices

    A, B, C, D = M_total.flatten()  # unpack layer stack matrix

    # solve Zangwill's eq. (17.92) for r and t:
    r = (A*Z[-1] + B - Z[0]*(C*Z[-1] + D))/(A*Z[-1] + B + Z[0]*(C*Z[-1] + D))
    t = 2*Z[-1]/(A*Z[-1] + B + Z[0]*(C*Z[-1] + D))

    return r, t


def R_T_A_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
                 substrate_k, pol, wavelength, incoherent=None, N_phases=None):
    """Same as r_t_coeffs, but for intensity reflectance and transmittance.
    Also return 'absorbtance' such that R + T + A = 1.

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
    wavelength : number
        Wavelength of incident light.
    incoherent : array, boolean, optional
        Element i is True if the i-th layer is "incoherent" (i.e. such that
        phase information is lost). Defaults to None which is interpreted as full
        coherence in all layers.
    N_phases : int, optional
        Use N_phases phase offsets for incoherent layers in averaging. Only
        used if at least one incoherent layer is present.

    Note
    ----
    A = 0 can only be expected for losless media under the condition that the
    embeddig medium is equal to the substrate.
    """

    # TODO: iterate over all permutations of incoherent layer additional phases
    #       compute r, t, a coherently
    #       average over all realizations

    if (incoherent is not None) and N_phases is None:
        raise ValueError("N_phases must be a number")
 
    # generate ensemble of phases to add in each layer. if a layer is coherent,
    # zero phase is added beyond the phase acquired by propagation through the
    # layer. if the layer is incoherent, linearly distructed values in [0, 2pi)
    # are added for each realization of the ensemble.
    if incoherent is not None:
        add_phases = [ np.linspace(0, 2*np.pi, N_phases, endpoint=False) if is_incoherent else [0.0]
                       for is_incoherent in incoherent
                     ]
        # iterate over realizations of the phase ensemble
        r_ensemble, t_ensemble = [], []
        for phi0 in product(*add_phases):
            curr_r, curr_t = r_t_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
                                        substrate_k, pol, wavelength, phi0)
            r_ensemble.append(curr_r)
            t_ensemble.append(curr_t)
        r, t = complex_average(r_ensemble), complex_average(t_ensemble)
    else:
        # add_phases = len(n)*[0.0]
        r, t = r_t_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
                          substrate_k, pol, wavelength, phi0=None)

    # prepare data for T correction:
    n_extended, k_extended = _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k)
    complex_n = n_extended + 1j*k_extended
    Z = _impedances(complex_n)  # complex impedance for all media
    cos_theta = _cos_theta(angle, complex_n)
    Z = _tilted_Z(Z, cos_theta, pol)

    R = np.abs(r)**2
    T = np.real(Z[0])/np.real(Z[-1])*np.abs(t)**2  # FIXME: order of Z
    # if pol.upper() in ("TE", "S"):
    T = np.abs(t)**2*np.real(cos_theta[-1]*complex_n[-1])/np.real(cos_theta[0]*complex_n[0])
    # else:
    #     T = np.abs(t)**2*np.real(np.conj(cos_theta[-1])*complex_n[-1])/np.real(np.conj(cos_theta[0])*complex_n[0])
    A = 1.0 - R - T

    return R, T, A
