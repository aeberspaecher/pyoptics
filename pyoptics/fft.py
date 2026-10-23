#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Helper module to deal with FFTs.

Convention: fields vary as exp(i(k.r - omega*t)). A field is a superposition
of plane waves exp(+i*(k_x*x + k_y*y)); the forward transform FT() has the
kernel exp(-i*(k_x*x + k_y*y)) and yields their amplitudes, the inverse
transform inv_FT() sums them up again:

    FT(f)[m, n] = sum_jl f[j, l]*exp(-i*(k_x[n]*x[l] + k_y[m]*y[j])),
    inv_FT(F)[j, l] = 1/(Nx*Ny) sum_mn F[m, n]*exp(+i*(k_x[n]*x[l] + k_y[m]*y[j])).

Both take and return centered arrays: x = 0 at index N//2 in position space
and k = 0 at index N//2 in frequency space, the latter matching
utils.frequencies() and utils.freq_grid() with normal_order=True. For even N,
utils.grid1d(-L/2, L/2, N) gives such a centered x. For odd N use e.g.
dx*(np.arange(N) - N//2). (If x is not centered, FT() equals the sum above up
to the constant phase factor exp(+i*k*x[N//2]), which inv_FT() removes again.)

FT() and inv_FT() are unnormalized like numpy.fft.fft2() and ifft2(); multiply
FT() by dx*dy to approximate the continuous Fourier integral. FT_unitary()
and inv_FT_unitary() are scaled by 1/sqrt(Nx*Ny) and sqrt(Nx*Ny) such that
FT_unitary() conserves the sum of |f|**2.

rFT() and inv_rFT() are the versions for real input. Their spectra only hold
the non-negative k_x (the columns N//2, N//2 + 1, ... of the FT() spectrum,
for even N followed by the Nyquist frequency); only the k_y axis is centered.

If installed, pyFFTW is used (via transparent_pyfftw or pyfftw's NumPy
interface); its transforms use the same sign convention as numpy.fft.
"""

import numpy as np
from numpy.fft import fftfreq, fftshift, ifftshift

try:
    from transparent_pyfftw.numpy_fft import fft, ifft, fft2, ifft2, rfft2, irfft2
    from transparent_pyfftw import save_wisdom
except ImportError:
    try:  # the 'fftw' extra installs pyfftw itself
        from pyfftw.interfaces.numpy_fft import fft, ifft, fft2, ifft2, rfft2, irfft2
    except ImportError:  # no FFTW at all: use NumPy's FFTs
        from numpy.fft import fft, ifft, fft2, ifft2, rfft2, irfft2

    def save_wisdom():
        """No-op: there is no FFTW wisdom to save without transparent_pyfftw."""
        pass


FT = lambda x: fftshift(fft2(ifftshift(x)))
inv_FT = lambda x: fftshift(ifft2(ifftshift(x)))

FT_unitary = lambda x: FT(x)/np.sqrt(np.prod(np.shape(x)))
inv_FT_unitary = lambda x: inv_FT(x)*np.sqrt(np.prod(np.shape(x)))


def rFT(x):
    """FT() for real x. Returns the spectrum for k_x >= 0 only (shape
    (Ny, Nx//2 + 1)); k_y is centered.
    """

    return fftshift(rfft2(ifftshift(x)), axes=0)


def inv_rFT(X, shape=None):
    """Inverse of rFT(). shape is the shape of the real array and needs to be
    given for an odd number of x samples.
    """

    if shape is None:
        shape = (np.shape(X)[0], 2*(np.shape(X)[1] - 1))

    return fftshift(irfft2(ifftshift(X, axes=0), s=shape))


def rFT_unitary(x):
    return rFT(x)/np.sqrt(np.prod(np.shape(x)))


def inv_rFT_unitary(X, shape=None):
    x = inv_rFT(X, shape)

    return x*np.sqrt(np.prod(np.shape(x)))


# TODO: add more routines that avoid fftshift?
