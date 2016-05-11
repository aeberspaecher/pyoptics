#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Wave Propagtion method of Singer / Brenner.
"""

import numpy as np
cimport numpy as np
from libcpp.complex import sqrt as complex_sqrt
#from numpy.lib.scimath import sqrt as complex_sqrt
import matplotlib.pyplot as plt

import numexpr as ne  # TODO: make optional?

from .utils import freq_grid, wavenumber, k_z
from .fft import FT


def wpm_propagate(n_func, field_z, x, y, z, delta_z, wl):
    """Propagate a field using WPM.

    Parameters
    ----------
    n_func : callable
        Refractive index n = n(x, y, z). Needs to work with vector-valued x, y.

    Returns
    -------
    field_propagted : array
        Field propagated to z_new = z + delta_z.
    """

    cdef Py_ssize_t i, j, m, n

    k0 = wavenumber(wl)
    X, Y = np.meshgrid(x, y)
    n_sampled = n_func(X, Y, z)

    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)

    field_freq_space = FT(field_z)

    print(n_sampled[:,:,np.newaxis, np.newaxis])
    print(np.shape(n_sampled[:,:,np.newaxis, np.newaxis]))
    print(n_sampled[:,:,np.newaxis, np.newaxis] - K_x[np.newaxis, np.newaxis, :, :] - K_y[np.newaxis, np.newaxis, :, :])

    #phase_factor = np.exp(1j*delta_z*kz)  # phase k_z*delta_z where k_z is express locally (as a function of x,y)
    phase_factor = np.exp(1j*delta_z*complex_sqrt(n_sampled[:,:,np.newaxis, np.newaxis]**2*k0**2 - K_x[np.newaxis, np.newaxis, :, :]**2 - K_y[np.newaxis, np.newaxis, :, :]**2))
    print(np.shape(phase_factor))


    # TODO: move code to Cython! Typical matrices obtained from broadcasting are too big...


    # TODO: how can the phase_factor be made four dimensional using broadcasting? two axes shall correspond to space, two axes to frequencies

    # perform summation:
    # for each pixel in the propagated field, sum over all k_x, k_y components weighted by a local phase factor
    # p = p(k_x, k_y, x, y)

    # TODO: so far, the algorithm assumes n changes only with x and y, but is
    # constant in z. can this condition be levered by a x,y-pixel-wise
    # integration of n over z?

    return field_z
