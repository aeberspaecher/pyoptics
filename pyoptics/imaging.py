#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Simple imaging.
"""

import numpy as np
try:
    from transparent_pyfftw import fft2, ifft2, fftshift, ifftshift, fftfreq, save_wisdom
except ImportError:
    from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from numpy import pi
from math import sqrt
from numpy.lib.scimath import sqrt as cmpl_sqrt
from .utils import freq_grid


def image(obj, x, y, NA, k):
    """Take an image in focus plane (may be complex!) and take it through a 4f
    system.
    """

    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=False)

    obj_spectrum = fftshift(fft2(ifftshift(obj)))

    # NA truncation:
    mask = np.ones(np.shape(obj))
    mask[K_x**2 + K_y**2 > k**2*NA**2] = 0.0

    #plt.imshow(mask)
    #plt.show()

    #obj_spectrum *= mask
    obj_spectrum[K_x**2 + K_y**2 > k**2*NA**2] = 0.0

    #obj_spectrum *= cmpl_sqrt(1 - K_x**2/k**2 - K_y**2/k**2)  # is this the obliquity factor rewritten using the sine condition?
    # TODO: cosine factor?

    image = fftshift(ifft2(ifftshift(obj_spectrum)))

    return image


def get_psf(x, y, NA, k, pupil_mask=None, phase=None):
    dx, dy = x[1] - x[0], y[1] - y[0]
    Nx, Ny = len(x), len(y)

    K_X, K_Y = freq_grid(x, y, wavenumbers=True, normal_order=False)

    # TODO: implement masks
    mask = K_X**2 + K_Y**2 <= k**2*NA**2

    #plt.imshow(mask)
    #plt.title("YO MOMMA!")
    #plt.show()

    pupil_func = np.zeros([Nx, Ny])
    pupil_func[mask] = 1.0

    psf = fftshift(ifft2(ifftshift(pupil_func)))
    #psf /= sqrt(Nx*Ny)

    return psf
