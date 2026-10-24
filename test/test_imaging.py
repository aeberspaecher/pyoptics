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


# PSF, OTF and Strehl ratio against analytical results:

from pyoptics.analytical import airy_pattern, airy_encircled_energy, circular_otf
from pyoptics.bases import FringeZernikes
from pyoptics.imaging import (get_mtf, get_otf, marechal_strehl, pupil_function,
                              rms_phase, strehl_ratio)

import pytest


@pytest.fixture(scope="module")
def fine_grid():
    # 0.2 lambda pixels, pupil radius L*NA/lambda = 50 samples
    L, N = 100*WL, 512
    x = grid1d(-L/2, L/2, N)

    return x


def test_psf_is_airy_pattern(fine_grid):
    x = fine_grid
    psf = get_psf(x, x, NA, K)
    I = np.abs(psf[len(x)//2, :])**2

    assert np.isclose(I[len(x)//2], 1.0)
    assert np.allclose(I, airy_pattern(np.abs(x), NA, WL), atol=5E-3)


def test_psf_energy_normalization(fine_grid):
    x = fine_grid
    dx = x[1] - x[0]
    psf = get_psf(x, x, NA, K, normalization="energy")

    assert np.isclose(np.sum(np.abs(psf)**2)*dx*dx, 1.0)


def test_encircled_energy_of_airy_pattern():
    # 83.8 % of the energy within the first dark ring at 0.61 lambda/NA:
    assert np.isclose(airy_encircled_energy(0.6098*WL/NA, NA, WL), 0.838, atol=1E-3)
    assert np.isclose(airy_encircled_energy(1E6*WL, NA, WL), 1.0, atol=1E-5)


def test_otf_of_circular_pupil(fine_grid):
    x = fine_grid
    otf = get_otf(x, x, NA, K)
    f = freq_grid(x, x, wavenumbers=False, normal_order=True)[0][len(x)//2]

    assert np.allclose(otf.imag, 0.0, atol=1E-10)
    assert np.allclose(otf[len(x)//2, :].real, circular_otf(f, NA, WL), atol=1E-2)
    assert np.allclose(get_mtf(x, x, NA, K), np.abs(otf))


def test_circular_otf_values():
    f_c = 2*NA/WL
    assert np.isclose(circular_otf(0.0, NA, WL), 1.0)
    assert np.isclose(circular_otf(f_c/2, NA, WL), 2/np.pi*(np.pi/3 - np.sqrt(3)/4))
    assert circular_otf(1.01*f_c, NA, WL) == 0.0


def test_mask_and_phase_as_arrays_or_callables(fine_grid):
    x = fine_grid
    P_x, P_y = np.meshgrid(*[freq_grid(x, x, normal_order=True)[0][0]/(K*NA)]*2)
    obscuration = lambda px, py: (px**2 + py**2 >= 0.3**2).astype(float)
    tilt = lambda px, py: 2.0*px

    P_callable = pupil_function(x, x, NA, K, obscuration, tilt)
    P_array = pupil_function(x, x, NA, K, obscuration(P_x, P_y), tilt(P_x, P_y))
    assert np.allclose(P_callable, P_array)

    with pytest.raises(ValueError):
        pupil_function(x, x, NA, K, np.ones((3, 3)))


def test_obscured_pupil_peak_normalization(fine_grid):
    """An annular pupil gives the PSF (2J_1(v)/v - eps^2 2J_1(eps v)/(eps v))^2/(1 - eps^2)^2."""

    from scipy.special import j1
    x = fine_grid
    eps = 0.4
    psf = get_psf(x, x, NA, K, pupil_mask=lambda px, py: px**2 + py**2 >= eps**2)
    v = 2*np.pi*NA*np.abs(x)/WL
    v[v == 0] = 1E-12
    I_analytic = ((2*j1(v)/v - eps**2*2*j1(eps*v)/(eps*v))/(1 - eps**2))**2

    assert np.allclose(np.abs(psf[len(x)//2, :])**2, I_analytic, atol=1E-2)


def test_strehl_ratio_and_marechal(fine_grid):
    x = fine_grid
    Z = FringeZernikes(x, x)  # used for scattered evaluation in pupil coordinates only
    a = 0.05  # Fringe Z9 (spherical) coefficient in waves
    phase = lambda px, py: 2*np.pi*a*Z.eval_single_scattered(px, py, 9)

    S = strehl_ratio(x, x, NA, K, phase=phase)
    psf = get_psf(x, x, NA, K, phase=phase)
    sigma = rms_phase(x, x, NA, K, phase=phase)

    # the PSF peak (normalization="peak") is the Strehl ratio:
    assert np.isclose(np.abs(psf[len(x)//2, len(x)//2])**2, S)
    # Fringe Z9 = 6 rho^4 - 6 rho^2 + 1 has the rms value 1/sqrt(5):
    assert np.isclose(sigma, 2*np.pi*a/np.sqrt(5), rtol=1E-2)
    # Marechal holds well for small aberrations:
    assert np.isclose(S, marechal_strehl(sigma), atol=5E-3)
    assert marechal_strehl(sigma, "quadratic") < marechal_strehl(sigma, "marechal") < 1.0


def test_tilt_does_not_change_psf_shape(fine_grid):
    """A pupil tilt shifts the PSF: the phase k_x*s = k*NA*s*P_x moves it to
    x = -s (the plane waves exp(i*k_x*x) of the pupil then add up in phase at
    x = -s)."""

    x = fine_grid
    dx = x[1] - x[0]
    shift = 10*dx
    tilt = lambda px, py: K*NA*shift*px
    I_tilted = np.abs(get_psf(x, x, NA, K, phase=tilt))**2
    I = np.abs(get_psf(x, x, NA, K))**2

    assert np.allclose(np.roll(I, -10, axis=1), I_tilted, atol=1E-10)


def test_coarse_grid_warns():
    x = grid1d(-10, 10, 32)  # dx = 0.625 > wavelength/(4 NA)
    with pytest.warns(UserWarning):
        get_otf(x, x, NA, K)
