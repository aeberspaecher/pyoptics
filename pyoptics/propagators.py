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

from math import pi
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.scimath import sqrt as complex_sqrt

from fft import fft2, ifft2, fftshift, ifftshift, FT, inv_FT
from utils import freq_grid, wavenumber, k_z


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

    #X_prime, Y_prime, x_new, y_new, X_new, Y_new = _new_coordinates_mesh(x_prime, y_prime, z, wavelength)

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


# TODO: also provide error term for Fresnel approxmiation?
def fresnel_number(aperture_diameter, z, wavelength):
    return aperture_diameter**2/(wavelength*z)


def z_from_fresnel_number(F, aperture_diameter, wavelength):
    z = aperture_diameter**2/(wavelength*F)

    return z



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
