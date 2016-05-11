#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
Propagators for electromagnetic fields.

The following propagators are available:

- rayleigh_sommerfeld_I_IR()
- rayleigh_sommerfeld_I_TF()
- fresnel_TF()
- fresnel_IR()
- fresnel_rescaling()
- fraunhofer()

The propagators share a common interface:

propagated_field, x, y = propagator_func(field, x_prime, y_prime, z, wavelength),

but individual propagators may accept further arguments.
"""

# TODO: can from_spectrum switches be implemented? this might save Fourier transforms.
# TODO: implement shifted windows
# TODO: implement Rayleigh-Sommerfeld direct integration

from math import pi, exp, sqrt
from itertools import product
import warnings

import matplotlib.pyplot as plt
import numpy as np

try:
    __numexpr_available = True
    import numexpr as ne
except ImportError:
    __numexpr_available = False


from fft import fft2, ifft2, fftshift, ifftshift, FT, inv_FT
from utils import freq_grid, wavenumber, k_z, TWOPI, simpson_weights, weight_grid


# TODO: account for n properly: lambda -> lambda/n


def rayleigh_sommerfeld_I_IR(field, x_prime, y_prime, z, wavelength, n=1.0):
    k = wavenumber(wavelength, n)
    X_prime, Y_prime = np.meshgrid(x_prime, y_prime)
    r = np.sqrt(z**2 + X_prime**2 + Y_prime**2)

    IR = z/(1j*wavelength)*np.exp(1j*k*r)/r**2

    field_propagated = inv_FT(FT(field)*FT(IR))

    return field_propagated, x_prime, y_prime


def rayleigh_sommerfeld_I_TF(field, x_prime, y_prime, z, wavelength, n=1.0):
    k = wavenumber(wavelength, n)
    K_X, K_Y = freq_grid(x_prime, y_prime, wavenumbers=True, normal_order=True)
    KZ = k_z(k, K_X, K_Y)

    TF = np.exp(1j*KZ*z)

    field_propagated = inv_FT(FT(field)*TF)

    return field_propagated, x_prime, y_prime


def _rs_di_g_np(x, y, z, k):
        print("Using g with numpy!")
        r = np.sqrt(x**2 + y**2 + z**2)
        val = 1/TWOPI*np.exp(1j*k*r)/r**2*z*(1/r - 1j*k)   # TODO: two pi? last terms in parens?

        return val

def _rs_di_g_ne(x, y, z, k):
        print("Using g with numexpr!")
        r = ne.evaluate("sqrt(x**2 + y**2 + z**2)")
        val = ne.evaluate("1/TWOPI*exp(1j*k*r)/r**2*z*(1/r - 1j*k)")
        print("... done: g with numexpr!")

        return val


def rayleigh_sommerfeld_I_DI(field, x_prime, y_prime, z, wavelength, n=1.0, use_simpson=True):
    """Implement the Rayleigh-Sommerfeld direct integration method of Shen and
    Wang.

    This method expressed a numerical integration of the Rayleigh-Sommerfeld
    diffraction formula using Simpon's rule as a convolution product which can
    be computed by means of FFTs.
    """

    # TODO: take out g(x,y,z) and decide whether to numexpress it on import...

    N = np.shape(field)
    assert(N[0] == N[1])  # TODO: raise more expressive exception
    N = N[0]

    g = _rs_di_g_ne if __numexpr_available else _rs_di_g_np

    k = wavenumber(wavelength, n)
    dx = x_prime[1] - x_prime[0]
    dy = y_prime[1] - y_prime[0]

    if(use_simpson):
        weights = weight_grid(simpson_weights, N, N)
    else:
        weights = np.ones([N, N])

    U = np.asarray(np.bmat([ [weights*field, np.zeros([N, N-1])],
                             [np.zeros([N-1, N]), np.zeros([N-1, N-1])]
                           ])
                  )

    # adapt Shen/Wang's indexing to zero-based indexing:
    x_vec = np.array( [x_prime[0] - x_prime[N - j] for j in range(1, N) ] +  [x_prime[j - N] - x_prime[0] for j in range(N, 2*N) ] )
    y_vec = np.array( [y_prime[0] - y_prime[N - j] for j in range(1, N) ] +  [y_prime[j - N] - y_prime[0] for j in range(N, 2*N) ] )
    X, Y = np.meshgrid(x_vec, y_vec)

    H = g(X, Y, z, k)

    val = ifft2(fft2(U)*fft2(H))*dx*dy
    field_propagated = val[-N:, -N:]  # lower right sub-matrix

    return field_propagated, x_prime, y_prime


def fresnel_IR(field, x_prime, y_prime, z, wavelength, n=1.0):
    k = wavenumber(wavelength, n)
    X_prime, Y_prime = np.meshgrid(x_prime, y_prime)

    IR = np.exp(1j*k*z)/(1j*wavelength*z)*np.exp(1j*k/(2.0*z)*(X_prime**2 + Y_prime**2))

    field_propagated = inv_FT(FT(field)*FT(IR))

    return field_propagated, x_prime, y_prime


def fresnel_TF(field, x_prime, y_prime, z, wavelength, n=1.0):
    k = wavenumber(wavelength, n)
    F_x, F_y = freq_grid(x_prime, y_prime, wavenumbers=False, normal_order=True)

    TF = np.exp(1j*k*z)*np.exp(-1j*pi*wavelength*z*(F_x**2 + F_y**2))

    field_propagated = inv_FT(FT(field)*TF)

    return field_propagated, x_prime, y_prime


def fresnel_rescaling(field, x_prime, y_prime, z, wavelength):
    X_prime, Y_prime, x_new, y_new, X_new, Y_new = _new_coordinates_mesh(x_prime, y_prime, z, wavelength)

    prefactor = np.exp(1j*k*z)/(1j*k*z)*np.exp(1j*k/(2*z)*(X_new**2 + Y_new**2))
    dx = x_prime[1] - x_prime[0]
    dy = y_prime[1] - y_prime[0]

    field_propagated = prefactor*FT(field*np.exp(1j/(2*z)*(X_prime**2 + Y_prime**2)))*dx*dy

    return field_propagated, x_new, y_new


def fraunhofer(field, x_prime, y_prime, z, wavelength):
    X_prime, Y_prime, x_new, y_new, X_new, Y_new = _new_coordinates_mesh(x_prime, y_prime, z, wavelength)

    prefactor = np.exp(1j*k*z)/(1j*k*z)*np.exp(1j*k/(2*z)*(X_new**2 + Y_new**2))
    dx = x_prime[1] - x_prime[0]
    dy = y_prime[1] - y_prime[0]

    field_propagated = prefactor*FT(field)*dx*dy

    return field_propagated, x_new, y_new


def _new_coordinates_mesh(x_prime, y_prime, z, wavelength):
    # create coordinates meshes for both "object" (primed coordinates) and
    # "image" space ("new" coordinates):
    X_prime, Y_prime = np.meshgrid(x_prime, y_prime)
    x_new, y_new = _fraunhofer_coord_scale(x_prime, y_prime, z, wavelength)
    X_new, Y_new = np.meshgrid(x_new, y_new)

    return X_prime, Y_prime, x_new, y_new, X_new, Y_new


def _fraunhofer_coord_scale(x, y, z, wavelength):
    """Scaled coordinates for Fraunhofer & coordinate scaling Fresnel propagator.

    Uses the fact that x = f_x*lambda*z with f_x as usual in FFT computations.
    """

    f_x, f_y = map(lambda item: fftshift(fftfreq(len(item), dx=(item[1] - item[0]))), (x, y))
    x_new, y_new = wavelength*z*f_x, wavelength*z*f_y

    return x_new, y_new


# TODO: also provide error term for Fresnel approxmiation?
def fresnel_number(aperture_diameter, z, wavelength):
    return aperture_diameter**2/(wavelength*z)


def z_from_fresnel_number(F, aperture_diameter, wavelength):
    z = aperture_diameter**2/(wavelength*F)

    return z
