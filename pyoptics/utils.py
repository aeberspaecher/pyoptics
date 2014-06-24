#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Utils for pyoptics.
"""

import numpy as np
from numpy.fft import fftfreq, fftshift
from scipy.signal import savgol_coeffs


def freq_grid(x, y, wavenumbers=True, normal_order=True):
    """Compute a Fourier frequency grid corresponding to given x, y.

    Parameters
    ----------
    x,y : arrays
        x and y values to use.
    wavenumbers, boolean, optional
        If True, wavenumbers instead of spatial frequencies will be returned.
    normal_order : bool, optional
        If False, the grid is not returned in 'normal' order, but rather in
        'real' order (with increasing frequencies/wavenumbers; zero-frequency
        in the middle).

    Returns
    -------
    freq_x, freq_y : arrays
        Meshed spatial frequencies/wavenumbers.
    """

    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])

    freq_x, freq_y = fftfreq(len(x), dx), fftfreq(len(y), dy)

    if(wavenumbers):
        freq_x, freq_y = 2*np.pi*freq_x, 2*np.pi*freq_y

    if(not normal_order):
        freq_x, freq_y = fftshift(freq_x), fftshift(freq_y)

    out = np.meshgrid(freq_x, freq_y)

    return out


def get_length_scales(wavelength, NA, n=1.0):
    """Return two length scales for an imaging system: the Airy diameter and
    the Rayleigh unit.

    Parameters
    ----------
    wavelength : double
        The vacuum wavelength.
    NA : double
    n : double, optional
        Refractive index of the medium. Defaults to 1 (air/vacuum).

    Returns
    -------
    airy : double
        The Airy *diameter* (not radius!).
    RU : double
        The Rayleigh unit.
    """

    return 1.22*n*wavelength/NA, n*wavelength/NA**2  # TODO: check


def get_z_derivative(stack, dz, order):
    r"""Get the derivative \partial_z I for an image stack. This quantity may
    be used in transport of intensity computations.

    Finite difference quotients are used in the computation. Finite difference
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

    num_images = np.size(stack, 2)
    deriv_coeffs = savgol_coeffs(num_images, order, 1, delta=dz, use="dot")

    stack_work = stack.copy()

    for i in range(num_images):
        stack_work[:, :, i] = deriv_coeffs[i]*stack_work[:, :, i]
        # TODO: write more elegantly, check what coeffs*stack_work does

    return np.sum(stack_work)


def add_camera_noise():
    """Apply a simple noise model to an image stack.
    """

    # TODO: implement a noise model Poissonian photon noise, and Gaussian
    # other noise
    pass


def scalar_product(field1, field2, x, y):
    pass
