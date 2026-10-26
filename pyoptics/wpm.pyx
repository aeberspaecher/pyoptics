#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Wave propagation method (WPM) of Brenner and Singer.

The WPM propagates a scalar field through a medium with a spatially varying
refractive index n(x, y, z) in thin slices delta_z. In each slice, the field
is decomposed into plane waves, and every plane wave is propagated at every
location (x, y) with the local k_z(x, y) = sqrt((n(x, y) k_0)^2 - k_x^2 - k_y^2)::

    U(x, y, z + delta_z) = sum_{k_x, k_y} A(k_x, k_y) exp(i (k_x x + k_y y))
                                          exp(i k_z(x, y) delta_z).

K.-H. Brenner and W. Singer, Appl. Opt. 32, 4984 (1993).

Two implementations of the same sum are available:

- "direct": the double sum for every pixel, O(N_x^2 N_y^2) per step,
  parallelized with OpenMP.
- "levels": for each distinct index value n_q in the slice, the whole
  spectrum is propagated with the homogeneous transfer function of n_q by
  FFT; each pixel takes the result of its own index value. This is the same
  sum, rearranged, and costs O(Q N_x N_y log(N_x N_y)) for Q distinct index
  values. Media made of a few materials (lenses, waveguides, gratings) are
  fast with it. This variant is known as the "fast WPM".

The default method="auto" chooses the cheaper one.

Convention: exp(i(k.r - omega*t)), i.e. propagation by delta_z multiplies a
plane wave by exp(+i*k_z*delta_z). Absorbing media have n + i*kappa with
kappa > 0. Evanescent waves decay (Im(k_z) >= 0).

Sampling: the plane wave decomposition is periodic in the window, and only
plane waves up to abs(k_x), abs(k_y) <= pi/dx are represented. To cover all
propagating waves in the densest medium, use dx <= wl/(2 n_max), see
max_sampling_distance(). Like all FFT propagators, the WPM needs a guard band
(or an absorbing index n + i*kappa at the window edges) so that light leaving
the window does not re-enter on the other side.

Limitations: the WPM is a scalar method and neglects reflections; at index
steps, the transmitted amplitude is not corrected by Fresnel coefficients.
"""


cdef extern from "complex.h":
    double complex cexp(double complex) nogil
    double complex csqrt(double complex) nogil
    double cimag(double complex) nogil


cimport cython
import numpy as np

from numpy.polynomial.legendre import leggauss
from cython.parallel import prange

from .utils import frequencies, wavenumber, k_z
from .fft import FT, inv_FT


# contributions damped by more than exp(-_MAX_DECAY) are skipped in the
# direct sum (exp(-46) ~ 1E-20):
cdef double _MAX_DECAY = 46.0


def averaging_z_sampling(func, x, y, z1, z2, N):
    """Average func(x, y, z) between z1 and z2 by computing the integral

        1/(z2 - z1) int_{z1}^{z2} func(x, y, z) dz

    with an N point Gauss-Legendre quadrature.

    Parameters
    ----------
    func : callable
        func(x, y, z), returns an array.
    x, y : arrays
        Passed on to func.
    z1, z2 : number
        Integration limits.
    N : int
        Number of quadrature nodes. N = 1 is the midpoint rule.

    Returns
    -------
    avg : array
        func averaged over z, same shape as func(x, y, z).
    """

    z_support_unshifted, weights = leggauss(N)
    # shift and scale [-1, +1] to [z1, z2]:
    z_shifted = (z2 - z1)/2.*z_support_unshifted + (z1 + z2)/2.

    func_sampled = np.array([func(x, y, z_val) for z_val in z_shifted])

    # the weights sum up to 2 (the length of [-1, +1]):
    return np.tensordot(weights, func_sampled, axes=1)/2.


def max_sampling_distance(wl, n_max):
    """Return the largest sampling distance dx for which the plane wave
    decomposition contains all propagating waves in a medium with the
    refractive index n_max: dx = wl/(2 n_max).

    Parameters
    ----------
    wl : number
        Vacuum wavelength.
    n_max : number
        Largest (real part of the) refractive index in the problem.

    Returns
    -------
    dx : number
    """

    return wl/(2.0*np.real(n_max))


def _sample_index(n_func, X, Y, z, delta_z, n_sampling):
    """Sample n_func in the slice [z, z + delta_z]."""

    if isinstance(n_sampling, str):
        if n_sampling == "mid":
            n = n_func(X, Y, z + delta_z/2.)
        elif n_sampling == "start":
            n = n_func(X, Y, z)
        else:
            raise ValueError(f"Unknown n_sampling '{n_sampling}', use 'mid', 'start' or an int.")
    else:
        n = averaging_z_sampling(n_func, X, Y, z, z + delta_z, int(n_sampling))

    n = np.asarray(n, dtype=np.complex128)
    if n.shape != X.shape:
        n = np.broadcast_to(n, X.shape)

    return n


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _direct_sum(double complex[:, ::1] n_sampled,
                      double complex[:, ::1] spectrum,
                      double complex[:, ::1] phase_x,  # exp(i k_x x), [j, m]
                      double complex[:, ::1] phase_y,  # exp(i k_y y), [i, l]
                      double[::1] k_x, double[::1] k_y,
                      double k0, double delta_z,
                      double complex[:, ::1] out) noexcept nogil:
    """out[i, j] = sum_{l, m} spectrum[l, m] exp(i(k_x x_j + k_y y_i))
                   exp(i k_z(n[i, j]) delta_z)"""

    cdef Py_ssize_t i, j, l, m
    cdef double complex kz, k_sq, acc, acc_y
    cdef double complex imag_unit = 1j

    for i in prange(n_sampled.shape[0], schedule="dynamic"):
        for j in range(n_sampled.shape[1]):
            k_sq = (n_sampled[i, j]*k0)**2
            acc = 0.0
            for l in range(k_y.shape[0]):
                acc_y = 0.0
                for m in range(k_x.shape[0]):
                    kz = csqrt(k_sq - k_x[m]**2 - k_y[l]**2)
                    # branch with Im(k_z) >= 0: evanescent waves decay (csqrt
                    # returns Im < 0 for arguments on the negative zero side
                    # of the branch cut):
                    if cimag(kz) < 0.0:
                        kz = -kz
                    if cimag(kz)*delta_z > _MAX_DECAY:
                        continue
                    acc_y = acc_y + spectrum[l, m]*phase_x[j, m]*cexp(imag_unit*kz*delta_z)
                acc = acc + acc_y*phase_y[i, l]
            out[i, j] = acc


def _cost_direct(Ny, Nx):
    # rough run time in units of 100 ns (from timings with 4 threads): one
    # term of the direct sum, a complex sqrt and exp, takes about 150 ns
    return 1.5*float(Nx*Ny)**2


def _cost_levels(Ny, Nx, num_levels):
    # per level: an FFT pair plus a constant Python overhead of about 80 us
    N = Nx*Ny
    return num_levels*(800.0 + 0.15*N*np.log2(N + 1))


def wpm_propagate(n_func, field_z, x, y, z, delta_z, wl, n_sampling="mid",
                  method="auto"):
    """Propagate a field through one slice with the WPM.

    Parameters
    ----------
    n_func : callable
        Refractive index n = n_func(X, Y, z) with gridded X, Y (as from
        np.meshgrid(x, y)). May be complex (n + i*kappa, kappa > 0 absorbs).
        n_func() is called once per slice (more often for n_sampling=int)
        with the whole grid, so that it can be vectorized.
    field_z : array
        Sampled field at z, shape (len(y), len(x)). A one-dimensional field
        of shape (len(x),) describes a problem invariant in y (pass y=None);
        n_func then receives one-dimensional X and Y = 0.
    x, y : arrays
        Equidistant coordinate arrays. y is ignored for one-dimensional
        fields.
    z : number
        Current z position.
    delta_z : number
        Distance to propagate.
    wl : number
        Vacuum wavelength.
    n_sampling : "mid", "start" or int, optional
        Where in the slice [z, z + delta_z] the index is sampled: in the
        middle (default), at the start, or averaged with an n_sampling point
        Gauss-Legendre quadrature.
    method : "auto", "direct" or "levels", optional
        Implementation of the WPM sum, see the module docstring. Both give the
        same result; "auto" picks the faster one.

    Returns
    -------
    field_propagated : array
        Field at z + delta_z, same shape as field_z.

    Notes
    -----
    The plane wave decomposition uses the centered DFT of pyoptics.fft, i.e.
    phases refer to the grid point x[N//2], y[M//2]. Any equidistant grid
    works, centered or not.
    """

    field_z = np.asarray(field_z)
    one_dim = (field_z.ndim == 1)
    x = np.asarray(x, dtype=np.float64)

    if one_dim:
        if field_z.shape != x.shape:
            raise ValueError("A one-dimensional field needs the shape of x.")
        X = x
        Y = np.zeros_like(x)
        field = field_z.reshape(1, -1)
        y = np.zeros(1)
        k_y = np.zeros(1)
    else:
        y = np.asarray(y, dtype=np.float64)
        if field_z.shape != (len(y), len(x)):
            raise ValueError(f"The field must have the shape (len(y), len(x)) = {(len(y), len(x))}, "
                             f"got {field_z.shape}.")
        X, Y = np.meshgrid(x, y)
        field = field_z
        k_y = frequencies(y, wavenumbers=True, normal_order=True)
    k_x = frequencies(x, wavenumbers=True, normal_order=True)

    Ny, Nx = field.shape
    k0 = wavenumber(wl)

    n_sampled = _sample_index(n_func, X, Y, z, delta_z, n_sampling).reshape(Ny, Nx)

    # plane wave amplitudes A(k_x, k_y) for exp(+i(k_x x + k_y y)), with x and
    # y relative to the grid points at index N//2 (the origin of FT()):
    spectrum = np.ascontiguousarray(FT(field.astype(np.complex128))/(Nx*Ny))

    if method == "auto":
        levels = np.unique(n_sampled)
        if _cost_levels(Ny, Nx, len(levels)) < _cost_direct(Ny, Nx):
            method = "levels"
        else:
            method = "direct"
    elif method == "levels":
        levels = np.unique(n_sampled)
    elif method != "direct":
        raise ValueError(f"Unknown method '{method}', use 'auto', 'direct' or 'levels'.")

    if method == "levels":
        out = np.empty((Ny, Nx), dtype=np.complex128)
        K_X, K_Y = np.meshgrid(k_x, k_y)
        for n_q in levels:
            KZ = k_z(n_q*k0, K_X, K_Y)
            field_q = inv_FT(spectrum*np.exp(1j*KZ*delta_z))*(Nx*Ny)
            mask = (n_sampled == n_q)
            out[mask] = field_q[mask]
    else:
        x_rel = x - x[Nx//2]
        y_rel = y - y[Ny//2]
        out = np.zeros((Ny, Nx), dtype=np.complex128)
        _direct_sum(np.ascontiguousarray(n_sampled), spectrum,
                    np.ascontiguousarray(np.exp(1j*np.outer(x_rel, k_x))),
                    np.ascontiguousarray(np.exp(1j*np.outer(y_rel, k_y))),
                    np.ascontiguousarray(k_x), np.ascontiguousarray(k_y),
                    k0, delta_z, out)

    if one_dim:
        out = out[0]

    return out


def wpm_propagate_z(n_func, field, x, y, z_vals, wl, n_sampling="mid",
                    method="auto"):
    """Propagate a field through the slices between consecutive z_vals and
    return the field at all z_vals.

    Parameters
    ----------
    n_func : callable
        n_func(X, Y, z), see wpm_propagate().
    field : array
        Field at z_vals[0], shape (len(y), len(x)), or (len(x),) for a problem
        invariant in y (pass y=None).
    x, y : arrays
        Coordinate arrays.
    z_vals : array
        Increasing (or decreasing) z positions. Need not be equidistant.
    wl : number
        Vacuum wavelength.
    n_sampling, method : optional
        See wpm_propagate().

    Returns
    -------
    fields : array
        Fields at z_vals, shape (len(z_vals),) + field.shape. fields[0] is the
        input field.
    """

    z_vals = np.asarray(z_vals, dtype=np.float64)
    field = np.asarray(field, dtype=np.complex128)

    fields = np.empty((len(z_vals),) + field.shape, dtype=np.complex128)
    fields[0] = field
    for k in range(1, len(z_vals)):
        fields[k] = wpm_propagate(n_func, fields[k-1], x, y, z_vals[k-1],
                                  z_vals[k] - z_vals[k-1], wl,
                                  n_sampling=n_sampling, method=method)

    return fields
