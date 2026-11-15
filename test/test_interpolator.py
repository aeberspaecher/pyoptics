#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.interpolator (trigonometric interpolation, Gibbs
windows) and the Fourier threshold CD of litho.metrics."""

import numpy as np
import pytest

from pyoptics.interpolator import (WINDOWS, TrigInterpolator, fourier_upsample, lanczos_sigma,
                                   level_crossings, smoothing_window, trig_interpolate,
                                   trig_interpolate_2d, upsampled_grid)


L = 3.0


def band_limited(u):
    return 1 + 0.5*np.cos(2*np.pi*u/L) + 0.3*np.sin(2*np.pi*4*u/L + 0.2)


@pytest.mark.parametrize("N", [9, 10, 16])
def test_1d_exact_for_band_limited(N):
    x = -0.7 + np.arange(N)*L/N
    xo = np.linspace(-2, 5, 301)
    assert np.allclose(trig_interpolate(band_limited(x), x, xo), band_limited(xo), atol=1E-12)


def test_nyquist_cosine_is_reproduced_real():
    """For even N the Nyquist cosine is split symmetrically: exact and real."""

    N = 10
    x = np.arange(N)*L/N
    f = lambda u: np.cos(2*np.pi*(N/2)*u/L)
    xo = np.linspace(0, L, 77)
    p = trig_interpolate(f(x), x, xo)
    assert np.isrealobj(p)
    assert np.allclose(p, f(xo), atol=1E-12)


def test_interpolates_samples_and_axes():
    rng = np.random.default_rng(1)
    x = np.arange(12)*0.5
    F = rng.standard_normal((3, 12, 2)) + 1j*rng.standard_normal((3, 12, 2))
    p = trig_interpolate(F, x, x, axis=1)
    assert p.shape == F.shape and np.allclose(p, F)
    assert trig_interpolate(F, x, np.zeros((4, 5)), axis=1).shape == (3, 4, 5, 2)
    assert trig_interpolate(F, x, 0.25, axis=1).shape == (3, 2)
    # periodicity
    assert np.allclose(trig_interpolate(F, x, x + 6.0, axis=1), F)


def test_2d_grid_points_and_upsample_agree():
    Nx, Ny = 12, 10
    x, y = np.arange(Nx)*2/Nx - 1, np.arange(Ny)/Ny
    g = lambda X, Y: (np.cos(2*np.pi*3*X/2)*np.sin(2*np.pi*2*Y)
                      + np.exp(1j*2*np.pi*(X/2 - 4*Y)))
    X, Y = np.meshgrid(x, y)
    xo, yo = np.linspace(-1, 1, 37), np.linspace(0, 1, 23)
    Xo, Yo = np.meshgrid(xo, yo)
    assert np.allclose(trig_interpolate_2d(g(X, Y), x, y, xo, yo), g(Xo, Yo), atol=1E-12)
    assert np.allclose(trig_interpolate_2d(g(X, Y), x, y, Xo, Yo, grid=False), g(Xo, Yo), atol=1E-12)
    Xu, Yu = np.meshgrid(upsampled_grid(x, 3), upsampled_grid(y, 3))
    assert np.allclose(fourier_upsample(g(X, Y), 3), g(Xu, Yu), atol=1E-12)
    # trailing axes are interpolated alike
    stack = np.dstack([g(X, Y), 2*g(X, Y)])
    assert np.allclose(trig_interpolate_2d(stack, x, y, xo, yo)[..., 1], 2*g(Xo, Yo), atol=1E-12)


@pytest.mark.parametrize("kind", WINDOWS)
def test_upsample_equals_direct_sum_with_windows(kind):
    Nx, Ny = 16, 8
    x, y = np.arange(Nx)/Nx, np.arange(Ny)/Ny
    X, Y = np.meshgrid(x, y)
    h = ((np.abs(X - 0.5) < 0.2) & (np.abs(Y - 0.5) < 0.3)).astype(float)
    a = fourier_upsample(h, (2, 4), axes=(1, 0), smoothing=kind)
    b = trig_interpolate_2d(h, x, y, upsampled_grid(x, 2), upsampled_grid(y, 4), smoothing=kind)
    assert np.isrealobj(a) and np.allclose(a, b, atol=1E-12)


def test_window_values():
    nu = np.array([0.0, 0.5, 1.0, 1.5])
    assert np.allclose(smoothing_window(nu, "lanczos"), [1, 2/np.pi, 0, 0])
    assert np.allclose(smoothing_window(nu, "fejer"), [1, 0.5, 0, 0])
    assert np.allclose(smoothing_window(nu, "hann"), [1, 0.5, 0, 0])
    assert np.allclose(smoothing_window(nu, None), [1, 1, 1, 0])
    s = lanczos_sigma(8)
    assert s[0] == 1 and s[4] == 0 and np.allclose(s[1], np.sinc(0.25))
    assert np.allclose(lanczos_sigma(8, normal_order=True), np.fft.fftshift(s))
    with pytest.raises(ValueError):
        smoothing_window(nu, "boxcar")


def test_gibbs_overshoot_of_a_step():
    """Interpolating a sampled step overshoots by 14 %, Lanczos sigma
    factors bring it to about 2 %, Fejer removes it."""

    N = 256
    x = np.arange(N)/N
    step = (x < 0.5).astype(float)
    over = {k: fourier_upsample(step, 16, axes=0, smoothing=k).max() - 1
            for k in ("none", "lanczos", "fejer")}
    assert 0.13 < over["none"] < 0.15
    assert 0.0 < over["lanczos"] < 0.025
    assert over["fejer"] <= 0.0
    over2 = fourier_upsample(step, 16, axes=0, smoothing="lanczos", order=2).max() - 1
    assert over2 < over["lanczos"]


def test_level_crossings_exact():
    N = 9
    x = np.arange(N)*L/N
    c, rising = level_crossings(x, band_limited(x), 1.2)
    assert len(c) == 4
    assert np.allclose(band_limited(c), 1.2, atol=1E-10)
    assert np.all(rising[1:] != rising[:-1])  # alternating
    assert np.all((c >= x[0]) & (c < x[0] + L))


def test_trig_interpolator_cutoff_filters_and_interpolates():
    N = 32
    x = np.arange(N)/N*2.0
    X, Y = np.meshgrid(x, x)
    smooth = np.cos(2*np.pi*X) + np.sin(2*np.pi*Y)  # frequency 1
    noise = 0.1*np.cos(2*np.pi*6*X)  # frequency 6
    ti = TrigInterpolator(smooth + noise, x, x, f_cutoff=3.0)
    assert np.all(np.abs(ti.F_X) <= 3.0)
    xo = np.linspace(0, 2, 41)
    Xo, Yo = np.meshgrid(xo, xo)
    assert np.allclose(ti(xo, xo), np.cos(2*np.pi*Xo) + np.sin(2*np.pi*Yo), atol=1E-12)
    assert np.allclose(ti(Xo, Yo, grid=False), ti(xo, xo), atol=1E-12)
    plain = TrigInterpolator(smooth + noise, x, x)
    assert np.allclose(plain(x, x), smooth + noise, atol=1E-12)


def test_rejects_non_equidistant():
    with pytest.raises(ValueError):
        trig_interpolate(np.ones(4), [0, 1, 2, 4], [0.5])


def test_threshold_cd_fourier_on_coarse_grid():
    """A two-harmonic image: the Fourier CD from 5 samples per period equals
    the exact CD, linear interpolation of the same samples misses."""

    from pyoptics.litho import metrics

    p = 100.0
    I = lambda u: 0.5 - 0.4*np.cos(2*np.pi*u/p) - 0.1*np.cos(4*np.pi*u/p)
    # dark line at 0 (I < threshold): exact edges from a fine cut
    uf = np.linspace(-p/2, p/2, 200001)
    cd_ref = metrics.threshold_cd(uf, I(uf), 0.3, 0.0)
    u = -p/2 + p*np.arange(5)/5
    assert abs(metrics.threshold_cd(u, I(u), 0.3, 0.0, interpolation="fourier") - cd_ref) < 1E-6
    u = -p/2 + p*np.arange(8)/8
    cd_lin = metrics.threshold_cd(u, I(u), 0.3, 0.0)
    assert abs(cd_lin - cd_ref) > 0.1
    assert abs(metrics.threshold_cd(u, I(u), 0.3, 0.0, interpolation="fourier") - cd_ref) < 1E-6
    # the grid need not start at the feature; edges around a center near the cell boundary
    u = 13.0 + p*np.arange(7)/7
    left, right = metrics.feature_edges(u, I(u), 0.3, center=p, interpolation="fourier")
    assert abs((right - left) - cd_ref) < 1E-6 and left < p < right
    # bright feature (space) and a feature that does not print
    assert abs(metrics.threshold_cd(u, I(u), 0.3, center=p/2, bright=True, interpolation="fourier")
               - (p - cd_ref)) < 1E-6
    assert np.isnan(metrics.threshold_cd(u, I(u), -0.05, 0.0, interpolation="fourier"))
