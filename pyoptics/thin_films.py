#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Thin film reflection and transmission coefficients.
"""

from functools import reduce  # Python 3 compatibility

import numpy as np
from numpy.lib.scimath import sqrt as complex_sqrt
from numpy import sin, cos

from utils import Z_0 as Z_vac, wavenumber
from interfaces import refracted_angle


def r_t_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
               substrate_k, pol, wavelength):
    """Reflection and transmission coefficients for a thin film stack.

    Parameters
    ----------
    n , k : arrays
        Real and imaginary parts of the refractive indices of the thin film stack.
    t : arrays
        Thicknesses of the layers. The length of T needs to be size(n) - 2.
    embedding_n, embedding_k, substrate_n, substrate_k : double
        Real and imaginary parts of the embedding medium's and the substrate's
        refractive index. Both the embedding medium and the substrate are taken
        be infinitely extend.
    pol : string
        "s" or "p" polarization.

    Returns
    -------
    r, t : double
        Reflection and transmission amplitude of the stack.
    """

    # TODO: make checks more verbose and raise sensible exceptions
    assert(len(n) >= 1)
    assert(len(k) >= 1)
    assert(len(t) == len(n) == len(k))

    n = np.asarray( [embedding_n] + list(n) + [substrate_n] )
    k = np.asarray( [embedding_k] + list(k) + [substrate_k] )

    complex_n = n + 1j*k
    Z = Z_vac/complex_n  # complex impedance for all media

    # build A B C D matrices for each layer:
    N_max = len(n) - 1

    ##note: cos(theta) = sqrt(1 - sin(theta)); sin(theta) is found from Snell's law:
    ##TODO: why complex_n[0]/complex_n[i] and not complex_n[i-1]/complex_n[i]?
    #cos_theta = complex_sqrt(1.0-(complex_n[0])/complex_n*sin(angle))**2

    # FIXME: which angle thingy is correct?

    # compute angles in layers using Snell's law:
    angles = [ angle, ]
    for i in range(1, N_max+1):
        angles.append( refracted_angle(angles[i-1], complex_n[i-1], complex_n[i]) )
    #angles = np.asarray(angles)
    cos_theta = np.cos(angles)

    # FIXME: which cos_theta is to use?

    # compute 'tilted' addmittances:
    if(pol in ("s", "TE")):
        Z *= cos_theta
    elif(pol in ("p", "TM")):
        Z /= cos_theta
    else:
        raise ValueError("'pol' must be on of 's'/'TE', 'p'/'TM'")

    # construct transfer matrices for all layers:
    M = []
    for j in range(1, N_max):
        phi_j = t[j-1]*wavenumber(wavelength, complex_n[j])*cos_theta[j]  # 'phase' accumulated in layer j

        M_curr = np.array( [ [cos(phi_j), -1j*Z[j]*sin(phi_j)], [-1j/Z[j]*sin(phi_j), cos(phi_j)] ] )
        M.append(M_curr)

    M_total = reduce(np.dot, M)  # multiply all M matrices
    # TODO: order of multiplication?

    A, B, C, D = M_total.flatten()  # unpack layer stack matrix

    # solve Zangwill's eq. (17.92) for r and t:
    r = (A*Z[-1] + B - Z[0]*(C*Z[-1] + D))/(A*Z[-1] + B + Z[0]*(C*Z[-1] + D))
    t = 2*Z[-1]/(A*Z[-1] + B + Z[0]*(C*Z[-1] + D))

    return r, t


def R_T_A_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
                 substrate_k, pol, wavelength):
    """Same as r_t_coeffs, but for intensity reflectance and transmittance. Also
    return 'absorptance' such that R + T + A = 1.

    Note
    ----
    A = 0 can only be expected for losless media under the condition that the
    embeddig medium is equal to the substrate.
    """

    r, t = r_t_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
                      substrate_k, pol, wavelength)

    R = np.abs(r)**2
    T = substrate_n/embedding_n*np.abs(t)**2  # TODO: can we interpret the prefactor as a radiometric correction?
    A = 1.0 - R - T

    return R, T, A


# test code:
if(__name__ == '__main__'):
    from utils import deg_to_rad
    import matplotlib.pyplot as plt

    AlOx_n = 1.635
    AlOx_k = 0.0

    Al_n = 1.6024
    Al_k = 7.4395

    Ag_n = 0.076220
    Ag_k = 6.5084

    wl = 0.905

    d = np.array( [0.135, 0.005, 0.200] )
    n = np.array([ AlOx_n, Al_n, Ag_n ] )
    k = np.array([ AlOx_k, Al_k, Ag_k ] )

    AOI_vals = np.linspace(0, 90, 200)

    R_p_vals, R_s_vals, T_s_vals, T_p_vals = [], [], [], []
    for AOI in AOI_vals:
        R_s, T_s, _ = R_T_A_coeffs(deg_to_rad(AOI), n, k, d, embedding_n=1.0,
                                   embedding_k=0.0, substrate_n=1.0, substrate_k=0.0,
                                   pol="s", wavelength=wl)
        R_p, T_p, _ = R_T_A_coeffs(deg_to_rad(AOI), n, k, d, embedding_n=1.0,
                                   embedding_k=0.0, substrate_n=1.0, substrate_k=0.0,
                                   pol="p", wavelength=wl)

        R_p_vals.append(R_p)
        R_s_vals.append(R_s)

        T_p_vals.append(T_p)
        T_s_vals.append(T_s)

    #print(np.array(R_s_vals) + np.array(T_s_vals))
    #print(np.array(R_p_vals) + np.array(T_p_vals))

    plt.plot(AOI_vals, R_s_vals, ls="-", color="blue", lw=1.5, label="$R_s$")
    plt.plot(AOI_vals, R_p_vals, ls="-", color="red", lw=1.5, label="$R_p$")
    plt.legend()
    plt.show()

    plt.plot(AOI_vals, T_s_vals, ls="--", color="blue", lw=1.5, label="$T_s$")
    plt.plot(AOI_vals, T_p_vals, ls="--", color="red", lw=1.5, label="$T_p$")
    plt.legend()
    plt.show()

    #print("R = {:.3f}%".format(100*R))
    #print("T = {:.3f}%".format(100*T))
    #print("R+T = {}".format(R + T))
