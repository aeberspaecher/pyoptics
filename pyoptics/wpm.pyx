#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Wave Propagtion method of Singer / Brenner.
"""

import numpy as np
cimport numpy as np
#from libcpp.complex cimport sqrt as complex_sqrt, exp as complex_exp

cdef extern from "complex.h":
    double complex csqrt(double complex) nogil
    double complex cexp(double complex) nogil

from cython.parallel import parallel, prange

import matplotlib.pyplot as plt

import numexpr as ne  # TODO: make optional?

from .utils import freq_grid, wavenumber, k_z, TWOPI
from .fft import FT, FT_unitary


cpdef wpm_propagate(n_func, field_z, x, y, z, delta_z, wl):
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
    cdef np.ndarray[ndim=2, dtype=double complex] field_propagted = \
        np.zeros(np.shape(field_z), dtype=np.complex128)
    cdef np.ndarray[ndim=2, dtype=double complex] n_sampled, field_freq_space
    cdef np.ndarray[ndim=2, dtype=double] K_x, K_y, X, Y
    cdef double complex kz
    cdef double dz, k0, curr_x, curr_y, curr_kx, curr_ky
    cdef double complex imag_unit = complex(0, 1)

    dz = delta_z

    k0 = wavenumber(wl)
    X, Y = np.meshgrid(x, y)

    n_sampled = np.asarray(n_func(X, Y, z), dtype=complex)

    K_x, K_y = freq_grid(x, y, wavenumbers=True, normal_order=True)

    field_freq_space = FT_unitary(field_z)

    # iterate all over spatial pixels, perform k summation for each pixel:
    for i in prange(n_sampled.shape[0], nogil=True, schedule="static", num_threads=2):
        for j in range(n_sampled.shape[1]):
            curr_x, curr_y = X[i, j], Y[i, j]
            for m in range(K_x.shape[0]):
                for n in range(K_x.shape[1]):
                    curr_kx, curr_ky = K_x[m, n], K_y[m, n]
                    kz = csqrt((n_sampled[i, j]*k0)**2 - curr_kx**2  - curr_ky**2)
                    field_propagted[i, j] += (field_freq_space[m, n]  # plane wave amplitude
                                              * cexp(imag_unit*(curr_kx*curr_x + curr_ky*curr_y))  # transversal part
                                              * cexp(imag_unit*dz*kz)  # propagating component
                                              )

    # TODO: truncate loop over frequencies if non-propagating frequencies occur?
    # TODO: can we use simpson weights to improve accuracy?
    # TODO: normalize appropriately

    # TODO: so far, the algorithm assumes n changes only with x and y, but is
    # constant in z. can this condition be levered by a x,y-pixel-wise
    # integration of n over z?

    return field_propagted/np.sqrt(np.prod(np.shape(field_z)))
