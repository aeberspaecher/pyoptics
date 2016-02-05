#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Thin film reflection and transmission coefficients.
"""

import numpy as np
from numpy import sin, cos

from utils import Z_0 as Z_vac, Z, wavenumber


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

    assert(len(n) >= 3)
    assert(len(k) >= 3)
    assert(len(t) == len(k) - 2)
    assert(pol in ("s", "p"))

    n = n + 1j*k

    print(angle)

    # assign impedances of source and substrate media:
    Z_0 = Z(n[0])
    Z_N = Z(n[-1])

    # build A B C D matrices for each layer:
    N_max = len(n) - 1
    M = []
    for j in range(1, N_max):
        phi_j = t[j-1]*wavenumber(wavelength, n[j])  # 'phase' accumulated in layer j
        Z_j = Z(n[j])
        if(pol == "s"):
            # n replaced by n*cos theta
            M_curr = np.array( [ [cos(phi_j), -1j*Z_j*(cos(angle))*sin(phi_j)], [-1j/(Z_j/cos(angle))*sin(phi_j), cos(phi_j)] ] )
        else:
            # n replaced by n/cos(theta)

            # *Zj is like /n --> *cos(theta)

            M_curr = np.array( [ [cos(phi_j), -1j*Z_j/cos(angle)*sin(phi_j)], [-1j/(Z_j*cos(angle))*sin(phi_j), cos(phi_j)] ] )
            # TODO: fix oblique incidence and polarisation

        M.append(M_curr)

    M_total = reduce(np.dot, M[::-1])  # multiply all M matrices - from "right" to "left"
    # TODO: order?


    #M_total = np.identity(2)
    #for i in range(len(M)):
        #M_total = np.dot(M_total, M[-i])

    A, B, C, D = M_total.flatten()  # unpack layer stack matrix

    # Brooker, eq (6.7) solved for r and t:
    r = (A*Z_N + B - Z_0*(C*Z_N + D))/(A*Z_N + B + Z_0*(C*Z_N + D))
    t = 2*Z_N/(A*Z_N + B + Z_0*(C*Z_N + D))

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

    AlOx_n = 1.63
    AlOx_k = 0.0

    Al_n = 1.9941
    Al_k = 8.2549

    Ag_n = 0.20821
    Ag_k = 5.7129

    wl = 0.905

    d = np.array( [wl/4, 0.005, 0.200] )
    n = np.array([ 1, AlOx_n, Al_n, Ag_n, 1 ] )
    k = np.array([ 0, AlOx_k, Al_k, Ag_k, 0 ] )


    AOI_vals = np.linspace(0, 80, 100)

    R_p_vals, R_s_vals = [], []
    for AOI in AOI_vals:
        R_s, T_s = R_T_coeffs(deg_to_rad(AOI), n, k, d, "s", wl)
        R_p, T_p = R_T_coeffs(deg_to_rad(AOI), n, k, d, "p", wl)
        R_p_vals.append(R_p)
        R_s_vals.append(R_s)

    plt.plot(AOI_vals, R_s_vals, ls="-", color="blue", lw=1.5, label="s")
    plt.plot(AOI_vals, R_p_vals, ls="-", color="red", lw=1.5, label="p")
    plt.legend()
    plt.show()


    print("R = {:.3f}%".format(100*R))
    print("T = {:.3f}%".format(100*T))
    print("R+T = {}".format(R + T))
