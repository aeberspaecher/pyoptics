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

from math import pi, ceil
from itertools import product

from numpy.lib.scimath import sqrt as complex_sqrt
import numpy as np

from pyoptics.fft import fft2, ifft2, fftshift, ifftshift, FT, inv_FT, FT_unitary, inv_FT_unitary
from pyoptics.utils import freq_grid, wavenumber, k_z, TWOPI, simpson_weights, weight_grid


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

    k = wavenumber(wavelength, n)
    dx = x_prime[1] - x_prime[0]
    dy = y_prime[1] - y_prime[0]

    if use_simpson:
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

    r = np.sqrt(X**2 + Y**2 + z**2)
    H = 1/TWOPI*np.exp(1j*k*r)/r**2*z*(1/r - 1j*k)   # TODO: two pi? last terms in parens?

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


def vectorial_fresnel(field, x_prime, y_prime, z, wavelength, n=1.0):
    """Implement Mansuripur's formulas for vectorial diffraction.

    Parameters
    ----------
    field : array
        Input field of dimension [Ny, Nx, 2]. The last index corresponds
        to x and y polarization components.
    x_prime, y_prime
    z
    wavelength

    Sources
    -------
    M. Mansuripur, "Certain computational aspects of vector diffraction problems,",
    J. Opt. Soc. Am. A 6, 786-805 (1989)
    """

    # TODO: Fresnel? Document Fresnel nature?

    # Mansuripur's approach is to identify each point in the aperture with a
    # single ray and use a Fourier transform aproach still
    # TODO: conditions for FT approach? should be in the book

    # use Mansuripur's notation inside the function:
    # T - spectrum of field
    # psi_alpha_beta - amplitude weight for the beta-polarization component
    #                  stemming from alpha component of light in aperture
    #

    field_propagated = np.zeros([np.size(field, 0), np.size(field, 1), 3],
                                dtype=np.complex)  # three polarization components

    # TODO: use scaled x, y instead of unsacled ones?
    k = wavenumber(wavelength, n)
    K_x, K_y = freq_grid(x_prime, y_prime, wavenumbers=True, normal_order=True)
    Sigma_x, Sigma_y = K_x/k, K_y/k
    Sigma_z = complex_sqrt(1.0 - Sigma_x**2 - Sigma_y**2)

    alpha = (0, 1)  # x & y polarization components in aperture
    beta = (0, 1, 2)  # x, y & z polarization for propgated field

    T = np.zeros_like(field, dtype=np.complex)
    for a in alpha:
        T[:, :, a] = FT_unitary(field[:, :, a])  # TODO: does fft2 have an axis argument or similar? or np.applyaxis?

    for a, b in product(alpha, beta):
        field_propagated[:, :, b] += inv_FT_unitary(
                    T[:, :, a]
                    *_psi_alpha_beta(a, b, Sigma_x, Sigma_y, Sigma_z)
                    *np.exp(1j*2*pi*z/wavelength*Sigma_z)
                                                   )

    return field_propagated, x_prime, y_prime


def vectorial_fresnel_extended(field, x_prime, y_prime, z, wavelength,
                               eta=1.0, n=1.0):
    """Implement Mansuripur's formula for extended Fresnel vectorial
    diffraction.

    Parameters
    ----------
    field : array
        Input field of dimension [Ny, Nx, 2]. The last index corresponds
        to x and y polarization components.
    x_prime, y_prime
    z
    wavelength

    Sources
    -------
    M. Mansuripur, "Certain computational aspects of vector diffraction problems,",
    J. Opt. Soc. Am. A 6, 786-805 (1989)
    """

    ## TODO: Fresnel? Document Fresnel nature?

    ## Mansuripur's approach is to identify each point in the aperture with a
    ## single ray and use a Fourier transform aproach still
    ## TODO: conditions for FT approach? should be in the book

    ## use Mansuripur's notation inside the function:
    ## T - spectrum of field
    ## psi_alpha_beta - amplitude weight for the beta-polarization component
    ##                  stemming from alpha component of light in aperture
    ##

    #field_propagated = np.zeros([np.size(field, 0), np.size(field, 1), 3],
                                #dtype=np.complex)  # three polarization components

    #k = wavenumber(wavelength, n)
    #K_x, K_y = freq_grid(x_prime, y_prime, wavenumbers=True, normal_order=True)
    #Sigma_x, Sigma_y = K_x/k, K_y/k
    #Sigma_z = complex_sqrt(1.0 - Sigma_x**2 - Sigma_y**2)

    #alpha = (0, 1)  # x & y polarization components in aperture
    #beta = (0, 1, 2)  # x, y & z polarization for propgated field

    #T = np.zeros_like(field, dtype=np.complex)
    #for a in alpha:
        #T[:, :, a] = FT(field[:, :, a])  # TODO: does fft2 have an axis argument or similar? or np.applyaxis?

    #for a, b in product(alpha, beta):
        #field_propagated[:, :, b] += inv_FT(
                #T[:, :, a]
                #*_psi_alpha_beta(a, b, Sigma_x, Sigma_y, Sigma_z)
                #*np.exp(1j*2*pi*z/wavelength*Sigma_z)
                                           #)
    #return field_propagated, x_prime, y_prime
    pass


def _psi_alpha_beta(alpha, beta, sigma_x, sigma_y, sigma_z):
    # TODO: can this be written in a compact way?
    if (alpha == 0) and (beta == 0):  # x to x
        psi = 1 - sigma_x**2/(1 + sigma_z)
    elif(alpha == 0) and (beta == 1):  # x to y
        psi = -sigma_x*sigma_y/(1+sigma_z)
    elif(alpha == 0) and (beta == 2):  # x to z
        psi = -sigma_x
    elif (alpha == 1) and (beta == 0):  # y to x
        psi = -sigma_x*sigma_y/(1+sigma_z)
    elif(alpha == 1) and (beta == 1):  # y to y
        psi = 1 - sigma_y**2/(1 + sigma_z)
    elif(alpha == 1) and (beta == 2):  # y to z
        psi = -sigma_y
    else:
        raise ValueError("Invalid values alpha, beta")

    psi /= complex_sqrt(sigma_z)  # see erratum to Mansuripur's 1989 paper

    return psi


# TODO: also provide error term for Fresnel approxmiation?
def fresnel_number(aperture_diameter, z, wavelength):
    return aperture_diameter**2/(wavelength*z)


def z_from_fresnel_number(F, aperture_diameter, wavelength):
    z = aperture_diameter**2/(wavelength*F)

    return z
