#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Wave Propagtion method of Singer / Brenner.
"""


cdef extern from "complex.h":
    double complex csqrt(double complex) nogil
    double complex cexp(double complex) nogil
    double cabs(double complex) nogil


cimport cython
import numpy as np
cimport numpy as np

from numpy.polynomial.legendre import leggauss
from cython.parallel import parallel, prange

from .utils import frequencies, wavenumber, k_z
from .fft import FT_unitary, inv_FT_unitary


def averaging_z_sampling(func, x, y, z1, z2, N):
    """Average func(x, y, z) between z1 and z2 by performing the integral
    1/(z2-z1) \int_{x1}^{x2} func(x, y, z) dz
    numerically.

    Parameters
    ----------
    func : callable
        f(x, y, z)
    z1, z2 : double
    N : int
        Number of samples to use in Gaussian quadrature.

    Returns
    -------
    avg : same type as f(x, y, z)
        Averaged integral over z
    """

    z_support_unshifted, weights = leggauss(N)
    # shift and scale [-1, +1] to [x, x+delta_x]
    z_shifted = (z2-z1)/2.*z_support_unshifted + (z1+z2)/2.

    func_sampled = np.array([func(x, y, z_val) for z_val in z_shifted])

    func_averaged = 1/2.*np.einsum("ijk,i->jk", func_sampled, weights)

    return func_averaged


@cython.boundscheck(False)
cpdef wpm_propagate(n_func, field_z, double[:] x, double[:] y, double z,
                    double delta_z, double wl):
    """Propagate a field using WPM.

    Parameters
    ----------
    n_func : callable
        Refractive index n = n(x, y, z). Needs to work with vector-valued x, y.
    field_z : array
        Sampled field at z = z_curr.
    x, y : arrays
        Coordinate arrays.
    z : number
        Current z position.
    delta_z : number
        Distance to propagte.
    wl : number
        Wavelength.

    Returns
    -------
    field_propagted : array
        Field propagated to z_new = z + delta_z.

    Note
    ----
    For N x-samples and M y-samples, the WPM scales with O(N^2*M^2). This fact
    should render it close to unusable for large problems.

    n_func() is called only once per routine. It is supposed to accept gridded
    x and y data. This choice may seem more specific than calling a function
    that evaluates n(x, y) for a single x, y tuple. However, the latter
    approach implied N*M function calls, which is slow if n_func() is a Python
    function. The approach we've chosen shifts the responsibility of being fast
    to n_func() and doesn't care whether n_func() achieves being fast through
    clever vectorization, Cythonizing or any other solution.
    """

    # NOTE: all array-like objects that enter into array-formulas with operations
    # like squaring are defined as NumPy arrays, everything else is a MemoryView

    cdef Py_ssize_t i, j, l, m
    cdef np.ndarray[ndim=2, dtype=double complex] field_propagted = \
        np.zeros(np.shape(field_z), dtype=np.complex128)
    cdef np.ndarray[ndim=2, dtype=double complex] n_sampled, field_freq_space
    cdef np.ndarray[ndim=2, dtype=double] X, Y
    cdef double[::1] k_x, k_y
    cdef double complex kz, n_curr
    cdef double k0, x_curr, y_curr, kx_curr, ky_curr
    cdef double complex imag_unit = complex(0.0, 1.0)

    k0 = wavenumber(wl)

    X, Y = np.meshgrid(x, y)

    n_sampled = np.asarray(n_func(X, Y, z), dtype=complex)

    k_x, k_y = (frequencies(x, wavenumbers=True, normal_order=True),
                frequencies(y, wavenumbers=True, normal_order=True) )

    field_freq_space = inv_FT_unitary(field_z)  # TODO: inv_FT or FT?

    # iterate all over spatial pixels, perform k summation for each pixel:
    for i in prange(n_sampled.shape[0], nogil=True, schedule="dynamic"):
        for j in range(n_sampled.shape[1]):
            x_curr, y_curr = x[j], y[i]
            n_curr = n_sampled[i, j]
            # find k summation limits for this iteration's n - choose such that
            # only propagating frequencies are chosen:
            # TODO: implement
            for l in range(k_y.shape[0]):
                for m in range(k_x.shape[0]):
                    kx_curr, ky_curr = k_x[m], k_y[l]
                    kz = csqrt((n_curr*k0)**2 - kx_curr**2  - ky_curr**2)
                    if(cabs(kz) < 0.0):  # TODO: leave in?
                        kz = 0.0
                    field_propagted[i, j] += (field_freq_space[l, m]  # plane wave amplitude
                                              # transversal part and propagating part:
                                              *cexp(imag_unit*((kx_curr*x_curr
                                                                + ky_curr*y_curr)
                                                               + delta_z*kz))
                                              )

    # TODO: truncate loop over frequencies if non-propagating frequencies occur?
    # FIXME: normalize appropriately
    # TODO: check if sampling is appropriate - all propagating frequencies covered?
    # TODO: add note on suggested sampling - all propagating frequencies shall be covered

    return field_propagted/np.sqrt(np.prod(np.array(np.shape(field_z))))
