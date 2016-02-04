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

x, y, field = propagator_func(field, x_prime, y_prime, z, wavelength),

but individual propagators may accept further arguments.
"""

from math import pi
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy.lib.scimath import sqrt as complex_sqrt

from utils import freq_grid, wavenumber, k_z

from fft import FT, inv_FT

# TODO: account for n properly: lambda -> lambda/n


def rayleigh_sommerfeld_I_IR(field, x_prime, y_prime, z, wavelength, n=1.0):
    k = wavenumber(wavelength, n)
    X_prime, Y_prime = np.meshgrid(x_prime, y_prime)
    r = np.sqrt(z**2 + X_prime**2, Y_prime**2)

    IR = z/(1j*wavelength)*np.exp(1j*k*r)/r**2

    field_propagated = inv_FT(FT(field)*FT(IR))

    return field_propagated


def rayleigh_sommerfeld_I_TF(field, x_prime, y_prime, z, wavelength, n=1.0):
    k = wavenumber(wavelength, n)
    K_X, K_Y = freq_grid(x_prime, y_prime, wavenumbers=True)
    KZ = k_z(k, K_X, K_Y)

    TF = np.exp(1j*KZ*z)

    field_propagated = inv_FT(FT(field)*TF)

    return field_propagated


def fresnel_IR(field, x_prime, y_prime, z, wavelength, n=1.0):
    k = wavenumber(wavelength, n)
    X_prime, Y_prime = np.meshgrid(x_prime, y_prime)

    IR = np.exp(1j*k*z)/(1j*wavelength*z)*np.exp(1j*k/(2*z)*(X_prime**2 + Y_prime**2))

    field_propagated = inv_FT(FT(field)*FT(IR))

    return field_propagated


def fresnel_TF(field, x_prime, y_prime, z, wavelength, n=1.0):
    k = wavenumber(wavelength, n)
    F_x, F_y = freq_grid(x_prime, y_prime, wavenumbers=False)

    TF = np.exp(1j*k*z)*np.exp(1j*pi*wavelength*z*(F_x**2 + F_y**2))

    field_propagated = inv_FT(FT(field)*TF)

    return field_propagated


def fresnel_rescaling(field, x_prime, y_prime, z, wavelength):
    # TODO: implement
    pass


def fraunhofer(field, x_prime, y_prime, z, wavelength):
    pass


def _fraunhofer_coord_scale(x, y):
    # TODO: implement
    pass


def prop_free_space(u0, x, y, delta_z, k, from_spectrum=False):
    """Propagate a field defined in the z=0 plane to a different z plane.

    Parameters
    ----------
    u0 : array
        Initial field.
    x, y : arrays
        x and y coordinates at which the initial field is sampled.
    z : double
        The z-coordinate to propagate the field to.
    k : double
        Wavenumber (may contain a factor from an immersion medium, in that case
        k=n*k_0).
    from_spectrum : boolean, optional
        If True, u0 is the spectrum of the field to propagate. This saves one
        Fourier transform. The spectrum is assumed to be in 'normal' order
        (zero frequency component at index (0, 0)).

    Returns
    -------
    u : array
        Propagated field.
    """

    # TODO: think about maximum safe propagation distance (Nyquist!)
    # TODO: explain use of scimath's sqrt function here
    # (evanescent field components!)
    # TODO: implement a switch that discards evanescent components

    # checks:
    check_free_space_sampling(x, y, k)

    if(from_spectrum):
        U0 = u0
    else:
        U0 = fftshift(fft2(ifftshift(u0)))

    K_X, K_Y = freq_grid(x, y, wavenumbers=True, normal_order=False)

    k_z = complex_sqrt(k**2 - K_X**2 - K_Y**2)
    transfer_function = np.exp(1j*k_z*delta_z)

    print("max(|transfer func| = %s"%np.max(np.abs(transfer_function)))
    print("min(|transfer func| = %s"%np.min(np.abs(transfer_function)))
    plt.imshow(np.log10(np.abs((transfer_function))))
    plt.title('log |transfer function|')
    plt.colorbar()
    plt.show()
    plt.imshow(np.angle((transfer_function)))
    plt.show()

    out = fftshift(ifft2(ifftshift(U0*transfer_function)))

    return out


def check_free_space_sampling(x, y, k):
    """Check whether the x,y sampling is such that all frequencies (that may or
    may not be present) that propagate non-evanescently are present in the
    given position space sampling.

    Note that this is not necessarily a problem as the actual frequency
    spectrum may be narrower.
    """

    K_X, K_Y = freq_grid(x, y)
    if(np.max(K_X**2 + K_Y**2) < k**2):
        warnings.warn("Not all non-evanescent parts of k-space are sampled!\nMaybe increase number of samples or decrease x,y extent.")


def prop_free_space_paraxial(u0, x, y, z, k, from_spectrum=False):

    # TODO: implement

    pass
