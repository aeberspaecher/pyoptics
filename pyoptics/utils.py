#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Utils for pyoptics.
"""

from math import pi

import numpy as np
from numpy.fft import fftfreq, fftshift
from numpy.lib.scimath import sqrt as complex_sqrt
from scipy.signal import savgol_coeffs

# TODO: add convenience routines the create complete grids? like a routine
# that returns the unit square from [-1, +1, -1, +1] input

TWOPI = 2*pi
epsilon_0 = 8.854187817620E-12  #  A^2 s^4 kg^−1 m^−3
mu_0 = 4*pi*1E-7  # V s / A m
Z_0 =  np.sqrt(mu_0/epsilon_0)  # vacuum impedance

deg_to_rad = lambda d: d/180.0*pi


rad_to_deg = lambda r: r/pi*180.0


def Z(n):
    """Impedance of medium of (possbily complex) refractive index.
    """

    impedance = Z_0/n   # TODO: is it really abs(n) or rather complex_sqrt(epsilon)?

    return impedance


def I(E, n=1.0):
    """Get intensity from field amplitude. Returns correct result as obtained by time-avergaing the Poynting vector.

    Parameters
    ----------
    E : array
        Field amplitude
    n : double, optional
        Refractive index of medium

    Returns
    -------
    I : array
        Field amplitude E**2*n/(2*Z_0)
    """

    intensity = E**2*n/(2*Z_0)

    return intensity


def wavenumber(wavelength, n=1.0):
    return 2*pi*n/wavelength


def k_z(k, k_x, k_y):
    return complex_sqrt(k**2 - k_x**2 - k_y**2)


def new_2d_grid(x_min, x_max, y_min, y_max, Nx, Ny=None,
                assume_periodicity=True):
    """Generate a two-dimensional grid...
    """

    if(Ny is None):
        Ny = Nx

    x = grid1d((x_min, x_max), Nx, endpoint=~assume_periodicity)
    y = grid1d((y_min, x_max), Ny, endpoint=~assume_periodicity)

    X, Y = grid2d(x, y)

    return x, y, X, Y


def grid1d(extent, N, x_min=0.0, assume_periodicity=True):
    x = np.linspace(extent[0] + x_min, extent[1] + x_min,
                    endpoint=~assume_periodicity)

    return x


def grid2d(x, y):
    [X, Y] = np.meshgrid(x, y)

    return X, Y


def freq_grid(x, y, wavenumbers=True, normal_order=True):
    """Compute a Fourier frequency grid corresponding to given x, y.

    Parameters
    ----------
    x,y : arrays
        x and y values to use.
    wavenumbers, boolean, optional
        If True, wavenumbers instead of spatial frequencies will be returned.
    normal_order : bool, optional
        If True, the grid is  returned in 'normal' order with zero frequency
        components centered and increasing frequencies. If False, the zero
        frequency component is to the 'left'.

    Returns
    -------
    freq_x, freq_y : arrays
        Meshed spatial frequencies/wavenumbers.
    """

    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    N_x, N_y = len(x), len(y)
    freq_x, freq_y = fftfreq(N_x, dx), fftfreq(N_y, dy)

    if(wavenumbers):
        freq_x, freq_y = 2*np.pi*freq_x, 2*np.pi*freq_y

    if(normal_order):
        freq_x, freq_y = fftshift(freq_x), fftshift(freq_y)

    out = np.meshgrid(freq_x, freq_y)

    return out


def get_length_scales(wavelength, NA, n=1.0, use_small_NA=False):
    """Return two length scales for an imaging system: the resolution
    limit/Airy diameter and depth of focus (DOF, for small NA known as the
    Rayleigh unit).

    Parameters
    ----------
    wavelength : double
        The vacuum wavelength.
    NA : double
    n : double, optional
        Refractive index of the medium. Defaults to 1 (air/vacuum).
    use_small_NA : boolean, optional
        If True, use the small NA approximation commonly used in microscopy for
        the DOF.

    Returns
    -------
    airy : double
        The Airy *diameter* (not radius!).
    DOF : double
        The Rayleigh unit.
    """

    airy_diameter = 1.22*n*wavelength/NA
    DOF = n*wavelength/NA**2  # TODO: better formula for DOF, case distinction

    return airy_diameter, DOF


def get_z_derivative(stack, dz, order):
    r"""Get the derivative \partial_z I for an image stack. This quantity may
    be used in transport of intensity computations.

    Finite difference quotients are used in the computation. Finite differences
    amplfiy noise. To avoid this, a Savitzky-Golay filter is used here for a
    noise smoothing derivative. The idea is to approximate the the derivative
    by a low order polynomial and use a larger number of samples (here: images)
    to define the polynomial in a least squares sense.

    Parameters
    ----------
    stack : array
        Intensity stack. An odd number of equidistant image plane centered
        around the plane of interest is expected.
    dz : double
        Distance of individual image planes.
    order : int
        Order of approximating polynomial. Must be smaller than the number of
        images.

    Returns
    -------
    dIdZ : array
        z-derivative in the central z-plane.
    """

    # TODO: implement default behaviour for order

    num_images = np.size(stack, 2)
    deriv_coeffs = savgol_coeffs(num_images, order, 1, delta=dz, use="dot")

    stack_work = stack.copy()

    # TODO: can this be replaced by einsum()?
    for i in range(num_images):
        stack_work[:, :, i] = deriv_coeffs[i]*stack_work[:, :, i]
        # TODO: write more elegantly, check what coeffs*stack_work does

    return np.sum(stack_work)


def add_camera_noise():
    """Apply a simple noise model to an image stack.
    """

    # TODO: implement a noise model with Poissonian photon noise, and Gaussian
    # other noise
    pass


def scalar_product(field1, field2, x, y):
    """Compute the scalar product <field_1|field_2>.
    """

    # use the simplest integration scheme possible
    prod = np.sum(np.conj(field1)*field2)*x*y/np.prod(np.shape(field1) - 1)

    # TODO: implement more precise schemes (2d Simpson?)

    return prod


def local_momentum(field, wavelength):
    # TODO: implement a sensible notion of local momentum / direction cosines
    pass


def simpson_weights(N):
    w = np.zeros(N)

    w[0], w[-1] = 2, 2
    w[1::2] = 8
    w[2:-1:2] = 4

    if(N % 2 == 0):
        w[-3:] += np.array([-1.0, 8, 5])/2
        # TODO: also treat "left" interval similarly?
        # or return average of left and right treatment?

    w /= 6.0

    return w


def weight_grid(func, Nx, Ny):
    wx = func(Nx)
    wy = func(Ny)

    w = np.einsum("i,j -> ij", wy, wx)
    # TODO: np.outer() seems faster from some N on...

    return w


#def trapz_weights(N):


if(__name__ == '__main__'):
    w = weight_grid(simpson_weights, 128, 128)
    import matplotlib.pyplot as plt
    plt.imshow(w, origin="lower", interpolation="none"); plt.show()
