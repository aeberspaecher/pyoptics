#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Helper module to deal with FFTs.
"""

import numpy as np

try:
    from transparent_pyfftw.numpy_fft import fft, ifft, fft2, ifft2, fftfreq, fftshift, ifftshift, rfft2, irfft2
    from transparent_pyfftw import save_wisdom
except ImportError:  # transparent_pyfftw not installed: use NumPy's FFTs
    from numpy.fft import fft, ifft, fft2, ifft2, fftfreq, fftshift, ifftshift, rfft2, irfft2

    def save_wisdom():
        """No-op: there is no FFTW wisdom to save without transparent_pyfftw."""
        pass


FT = lambda x: fftshift(fft2(ifftshift(x)))
inv_FT = lambda x: fftshift(ifft2(ifftshift(x)))

FT_unitary = lambda x: FT(x)/np.sqrt(np.prod(np.shape(x)))
inv_FT_unitary = lambda x: inv_FT(x)*np.sqrt(np.prod(np.shape(x)))

rFT = lambda x: fftshift(rfft2(ifftshift(x)))
inv_rFT = lambda x: fftshift(irfft2(ifftshift(x)))

rFT_unitary = lambda x: rFT(x)/np.sqrt(np.prod(np.shape(x)))
inv_rFT_unitary = lambda x: inv_rFT(x)*np.sqrt(np.prod(np.shape(x)))


# TODO: add more routines that avoid fftshift?
