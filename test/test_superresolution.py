#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.superresolution."""

import numpy as np
import pytest

from pyoptics import superresolution as sr
from pyoptics.fft import FT
from pyoptics.imaging import get_otf
from pyoptics.utils import grid2d, freq_grid


NA, WL = 1.2, 0.5


@pytest.fixture
def setup():
    N, dx = 128, 0.04
    L = N*dx
    x, y, X, Y = grid2d(-L/2, L/2, -L/2, L/2, N)
    otf = np.real(get_otf(x, y, NA, 2*np.pi/WL))
    k_c = 2*NA*2*np.pi/WL
    rng = np.random.default_rng(0)
    sample = rng.random((N, N))*np.exp(-(np.hypot(X, Y)/1.8)**8)
    return x, y, X, Y, otf, k_c, sample


def test_pattern_mean_and_modulation(setup):
    x, y, X, Y, otf, k_c, _ = setup
    p = sr.sim_pattern(x, y, 10.0, 0.3, 1.0, modulation=0.7)
    assert np.isclose(p.max(), 1.7, atol=0.01) and np.isclose(p.min(), 0.3, atol=0.01)


def test_band_separation_exact(setup):
    x, y, X, Y, otf, k_c, sample = setup
    k_p, angles, phases = 0.9*k_c, [0.2], np.arange(3)*2*np.pi/3
    imgs = sr.sim_images(sample, x, y, otf, k_p, angles, phases, modulation=0.8)
    bands = sr.sim_bands(imgs, phases, modulation=0.8)
    for i, j in enumerate((-1, 0, 1)):
        ref = otf*FT(sample*np.exp(1j*j*k_p*(np.cos(0.2)*X + np.sin(0.2)*Y)))
        assert np.abs(bands[0, i] - ref).max() < 1e-10*np.abs(ref).max()


def test_shift_spectrum_moves_peak(setup):
    x, y, X, Y, otf, k_c, _ = setup
    K_x, K_y = freq_grid(x, y)
    dk = K_x[0, 1] - K_x[0, 0]
    spec = FT(np.exp(1j*5*dk*X))           # a peak at k_x = 5 dk
    shifted = sr.shift_spectrum(spec, x, y, 5*dk, 0.0)
    iy, ix = np.unravel_index(np.argmax(np.abs(shifted)), shifted.shape)
    assert K_x[iy, ix] == 0 and K_y[iy, ix] == 0


def test_otf_support_radius(setup):
    x, y, X, Y, otf, k_c, _ = setup
    K = np.hypot(*freq_grid(x, y))
    k_p = 0.9*k_c
    angles = np.arange(3)*np.pi/3
    sup = sr.otf_support(otf, x, y, k_p, angles)
    k_min = k_p*np.cos(np.pi/6) + np.sqrt(k_c**2 - (k_p/2)**2)
    dk = 2*np.pi/(len(x)*(x[1] - x[0]))
    assert abs(K[~sup].min() - k_min) < 2*dk
    assert abs(K[sup].max() - (k_c + k_p)) < 2*dk


def test_sim_beats_widefield(setup):
    x, y, X, Y, otf, k_c, _ = setup
    k_p = 0.9*k_c
    k_s = 1.5*k_c                            # beyond the wide field cutoff
    sample = (1 + np.cos(k_s*X))*np.exp(-(np.hypot(X, Y)/1.8)**8)
    angles, phases = np.arange(3)*np.pi/3, np.arange(3)*2*np.pi/3
    imgs = sr.sim_images(sample, x, y, otf, k_p, angles, phases)
    rec, _ = sr.sim_reconstruct(imgs, x, y, otf, k_p, angles, phases, wiener=0.01)
    wide = sr.wiener_deconvolution(imgs.sum(axis=(0, 1)), otf, x, y, wiener=0.01)
    K_x, K_y = freq_grid(x, y)
    i = np.argmin(np.abs(K_x[0] - k_s))
    c = len(y)//2
    assert np.abs(FT(rec))[c, i] > 50*np.abs(FT(wide))[c, i]


def test_vortex_focus_zero_depends_on_handedness():
    x, y, X, Y = grid2d(-3, 3, -3, 3, 256)
    c = 128
    I = {}
    for jones in ((1, 1j), (1, -1j)):
        E = sr.vortex_focus(x, y, 0.775, 1.4, n=1.518, charge=1, jones=jones)
        tot = np.sum(np.abs(E)**2, -1)
        I[jones] = tot[c, c]/tot.max()
    assert I[(1, 1j)] < 1e-3
    assert I[(1, -1j)] > 0.5


def test_sted_psf_and_fwhm():
    r = np.linspace(-1, 1, 200001)
    d0 = 0.25
    exc = np.exp(-4*np.log(2)*r**2/d0**2)
    beta = 4/d0**2                           # textbook case beta*d0**2/4 = 1
    for zeta in (1, 10, 100):
        # doughnut r**2 (peak 1 at r = 1): zeta' = zeta*beta gives exp(-ln2*zeta*beta*r**2)
        h = sr.sted_psf(exc, r**2, zeta*beta)
        above = r[h >= 0.5*h.max()]
        assert np.isclose(above[-1] - above[0], sr.sted_fwhm(1, 1, zeta, d0=d0), rtol=0.01)
    assert np.isclose(sr.sted_fwhm(0.6, 1.2, 3), 0.125)


def test_localization_precision_limits():
    assert np.isclose(sr.localization_precision(0.1, 100), 0.01)
    assert np.isclose(sr.localization_precision(0.1, 100, method="lsq"), 0.01*4/3)
    s0 = sr.localization_precision(0.1, 1000, pixel=0.1)
    s1 = sr.localization_precision(0.1, 1000, pixel=0.1, background=5.0)
    assert s1 > s0 > 0.1/np.sqrt(1000)


def test_gaussian_spot_total():
    xs = (np.arange(41) - 20)*0.05
    spot = sr.gaussian_spot(xs, xs, 0.01, -0.02, 0.1, 500.0, background=2.0)
    assert np.isclose(spot.sum() - 2.0*spot.size, 500.0, rtol=1e-6)


@pytest.mark.parametrize("method", ["mle", "lsq", "centroid"])
def test_localize_precision(method):
    rng = np.random.default_rng(1)
    xs = (np.arange(11) - 5)*0.1
    err = []
    for _ in range(150):
        x0, y0 = rng.uniform(-0.05, 0.05, 2)
        img = rng.poisson(sr.gaussian_spot(xs, xs, x0, y0, 0.1, 1000, 0.0))
        err.append(sr.localize(img, xs, xs, 0.1, method)[0] - x0)
    expected = sr.localization_precision(0.1, 1000, 0.1, method="lsq" if method == "lsq" else "mle")
    assert abs(np.mean(err)) < 0.5*expected
    assert 0.75*expected < np.std(err) < 1.3*expected
