#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the simple imaging module.
"""

import numpy as np

from pyoptics.fft import FT
from pyoptics.imaging import get_psf, image
from pyoptics.utils import freq_grid, grid1d


WL = 0.5
K = 2*np.pi/WL
NA = 0.5


def test_band_limited_object_is_imaged_unchanged():
    x = grid1d(-10, 10, 128)
    X, Y = np.meshgrid(x, x)
    obj = np.exp(-(X**2 + Y**2)/2.0**2)  # spectrum far inside the NA circle

    assert np.allclose(image(obj, x, x, NA, K), obj, atol=1E-10)


def test_spectrum_is_cut_at_NA():
    x = grid1d(-10, 10, 128)
    X, Y = np.meshgrid(x, x)
    rng = np.random.default_rng(1)
    obj = rng.standard_normal(X.shape)

    K_X, K_Y = freq_grid(x, x, wavenumbers=True, normal_order=True)
    outside = K_X**2 + K_Y**2 > (K*NA)**2
    spectrum = FT(image(obj, x, x, NA, K))

    assert np.allclose(spectrum[outside], 0.0, atol=1E-8)
    assert not np.allclose(spectrum[~outside], 0.0)


def test_psf():
    x = grid1d(-10, 10, 128)
    y = grid1d(-10, 10, 100)
    psf = get_psf(x, y, NA, K)

    assert psf.shape == (len(y), len(x))
    # an unaberrated, symmetric pupil gives a real PSF peaked in the center:
    assert np.allclose(psf.imag, 0.0, atol=1E-12)
    assert np.unravel_index(np.argmax(psf.real), psf.shape) == (len(y)//2, len(x)//2)
    assert np.all(psf.real[len(y)//2, len(x)//2 - 3:len(x)//2 + 4] > 0)
