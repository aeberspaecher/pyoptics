#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Thin film reflection and transmission coefficients.
"""

import numpy as np
from numpy import sin, cos

from utils import Z_0 as Z_vac, Z, wavenumber
from interfaces import refracted_angle


def r_t_coeffs(angle, n, k, t, pol, wavelength):
    """Reflection and transmission coefficients for a thin film stack.

    Parameters
    ----------
    n , k : arrays
        Real and imaginary parts of the refractive indices in the order [layer
        0 = source_medium, layer 1, layer 2, ..., layer N, layer N+1 =
        substrate]. The substrate and the source medium are taken to be
        infinitely extended: layer 0 extends to -oo, layer N+1 to +oo.
    t : arrays
        Thicknesses of the layers. The length of T needs to be size(n) - 2.
    pol : string
        "s" or "p" polarization.

    Returns
    -------
    r, t : double
        Reflection and transmission amplitude of the stack.
    """

    # TODO: make checks more verbose and raise sensible exceptions
    assert(len(n) >= 3)
    assert(len(k) >= 3)
    assert(len(t) == len(k) - 2)
    assert(pol in ("s", "p"))

    n = n + 1j*k

    # assign impedances of source and substrate media:
    Z_0 = Z(n[0])  # embedding medium impedance
    Z_N = Z(n[-1])  # substrate impedance

    # build A B C D matrices for each layer:
    N_max = len(n) - 1
    M = []

    # compute angles in layers using Snell's law:
    angles = [angle,]
    for i in range(N_max):
        angles.append( refracted_angle(angles[i], n[i], n[i+1]) )
    angles = np.asarray(angles, dtype=np.complex)

    # modify n by polarization dependent term as in Brooker's excercise:
    # NOTE: impedances are constructed with the unmodified n values. all code
    # below these lines uses n only in the transfer matrices. in that case,
    # it's okay to use modified n values.
    n_pol_modified = n*cos(angles) if pol == "s" else n/cos(angles)

    # construct transfer matrices for all layers:
    for j in range(1, N_max):
        phi_j = t[j-1]*wavenumber(wavelength, n_pol_modified[j])  # 'phase' accumulated in layer j

        M_curr = np.array( [ [cos(phi_j), -1j/n_pol_modified[j]*sin(phi_j)], [-1j*n_pol_modified[j]*sin(phi_j), cos(phi_j)] ] )
        M.append(M_curr)

    M_total = reduce(np.dot, M)  # multiply all M matrices
    # TODO: order of multiplication?

    A, B, C, D = M_total.flatten()  # unpack layer stack matrix

    # Brooker, eq (6.7) solved for r and t:
    # different notation: we use Z_vac for vacuum impedance, Brooker uses Z_0.
    # we use Z_0 for the embedding medium impedance, Brooker uses Z_1.
    # Brooker in his example uses Z_4 for the substrate impedance, we use Z_N.
    r = (-Z_0*(C*Z_N + D*Z_vac) + Z_vac*(A*Z_N + B*Z_vac))/(Z_0*(C*Z_N + D*Z_vac) + Z_vac*(A*Z_N + B*Z_vac))
    t = 2*Z_N*Z_vac/(Z_0*(C*Z_N + D*Z_vac) + Z_vac*(A*Z_N + B*Z_vac))

    return r, t


def R_T_coeffs(angle, n, k, t, pol, wavelength):
    """Same as r_t_coeffs, but for intensity reflectivity and transmittivity.
    """

    r, t = r_t_coeffs(angle, n, k, t, pol, wavelength)
    R, T = np.abs(r)**2, np.abs(t)**2

    return R, T



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

    d = np.array( [0.135, 0.200] )
    n = np.array([ 1.0, AlOx_n, Ag_n, 1.0 ] )
    k = np.array([ 0.0, AlOx_k, Ag_k, 0.0 ] )

    AOI_vals = np.linspace(0, 60, 200)

    R_p_vals, R_s_vals, T_s_vals, T_p_vals = [], [], [], []
    for AOI in AOI_vals:
        R_s, T_s = R_T_coeffs(deg_to_rad(AOI), n, k, d, "s", wl)
        R_p, T_p = R_T_coeffs(deg_to_rad(AOI), n, k, d, "p", wl)

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
