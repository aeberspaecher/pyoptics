#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Helper module to deal with FFTs.
"""

# TODO: add more routines that avoid fftshift?

try:
    from transparent_pyfftw.numpy_fft import fft, ifft, fft2, ifft2, fftfreq, fftshift, ifftshift
except ImportError:
    from numpy.fft import fft, ifft, fft2, ifft2, fftfreq, fftshift, ifftshift


FT = lambda x: fftshift(fft2(ifftshift(x)))
inv_FT = lambda x: ifftshift(ifft2(fftshift(x)))
