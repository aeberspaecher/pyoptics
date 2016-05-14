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

from .utils import frequencies, wavenumber, k_z, TWOPI
from .fft import FT, FT_unitary


cpdef wpm_propagate(n_func, field_z, double[:] x, double[:] y, double z, double delta_z, double wl):
    """Propagate a field using WPM.

    Parameters
    ----------
    n_func : callable
        Refractive index n = n(x, y, z). Needs to work with vector-valued x, y.

    Returns
    -------
    field_propagted : array
        Field propagated to z_new = z + delta_z.

    Note
    ----
    For N x-samples and M y-samples, the WPM scales with O(N^2*M^2).
    """

    cdef Py_ssize_t i, j, m, n
    cdef np.ndarray[ndim=2, dtype=double complex] field_propagted = \
        np.zeros(np.shape(field_z), dtype=np.complex128)
    cdef np.ndarray[ndim=2, dtype=double complex] n_sampled, field_freq_space
    cdef np.ndarray[ndim=2, dtype=double] X, Y
    cdef double[:] k_x, k_y
    cdef double complex kz
    cdef double k0, curr_x, curr_y, curr_kx, curr_ky
    cdef double complex imag_unit = complex(0.0, 1.0)

    k0 = wavenumber(wl)

    X, Y = np.meshgrid(x, y)  # TODO: make n_sampled local in x, y (attention! works nicely with cpdef'ed funcs)
    n_sampled = np.asarray(n_func(X, Y, z), dtype=complex)

    k_x, k_y = frequencies(x, wavenumbers=True, normal_order=True), frequencies(y, wavenumbers=True, normal_order=True)

    field_freq_space = FT_unitary(field_z)

    # iterate all over spatial pixels, perform k summation for each pixel:
    for i in prange(n_sampled.shape[0], nogil=True, schedule="dynamic"):
        for j in range(n_sampled.shape[1]):
            curr_x, curr_y = x[j], y[i]
            for m in range(k_x.shape[0]):
                for n in range(k_y.shape[0]):
                    curr_kx, curr_ky = k_x[n], k_y[m]
                    kz = csqrt((n_sampled[i, j]*k0)**2 - curr_kx**2  - curr_ky**2)
                    field_propagted[i, j] += (field_freq_space[m, n]  # plane wave amplitude
                                              # transversal part and propagating part:
                                              *cexp(imag_unit*((curr_kx*curr_x + curr_ky*curr_y) + delta_z*kz))
                                              )

    # TODO: truncate loop over frequencies if non-propagating frequencies occur?
    # TODO: can we use simpson weights to improve accuracy?
    # TODO: normalize appropriately

    # TODO: so far, the algorithm assumes n changes only with x and y, but is
    # constant in z. can this condition be levered by a x,y-pixel-wise
    # integration of n over z?

    return field_propagted/np.sqrt(np.prod(np.shape(field_z)))
