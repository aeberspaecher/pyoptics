#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test ISO 11146 beam widths and M² against Gaussian and Hermite-Gaussian
beams.
"""

import numpy as np
import pytest

from pyoptics.beam_quality import (beam_widths, fit_caustic, second_moments,
                                   widths_from_moments)
from pyoptics.beams import gauss_hermite
from pyoptics.utils import grid2d


WL = 1.0
W0 = 10.0
Z_R = np.pi*W0**2/WL


@pytest.fixture(scope="module")
def grid():
    return grid2d(-200, 200, -200, 200, 512)


@pytest.mark.parametrize("iterate", [False, True])
def test_gaussian_beam_width_is_2w(grid, iterate):
    x, y, X, Y = grid
    w = 15.0
    I = np.exp(-2*((X - 20)**2 + (Y + 10)**2)/w**2)
    widths = beam_widths(I, x, y, iterate=iterate)

    assert np.allclose([widths.d_x, widths.d_y, widths.d_r], 2*w, rtol=1E-6)
    assert np.allclose([widths.x_c, widths.y_c], [20, -10])
    assert widths.ellipticity == pytest.approx(1.0)


@pytest.mark.parametrize("phi", [0.0, 0.3, -0.6, np.pi/4, 1.2])
def test_rotated_elliptical_beam(grid, phi):
    x, y, X, Y = grid
    w_u, w_v = 30.0, 12.0
    U = np.cos(phi)*X + np.sin(phi)*Y
    V = -np.sin(phi)*X + np.cos(phi)*Y
    I = np.exp(-2*(U**2/w_u**2 + V**2/w_v**2))
    widths = beam_widths(I, x, y)

    # the principal "x" axis is the one closer to the lab x axis:
    if abs(phi) <= np.pi/4:
        assert np.allclose([widths.d_x, widths.d_y, widths.phi], [2*w_u, 2*w_v, phi], rtol=1E-6)
    else:
        assert np.allclose([widths.d_x, widths.d_y, widths.phi], [2*w_v, 2*w_u, phi - np.pi/2], rtol=1E-6)


@pytest.mark.parametrize("l, m", [(1, 0), (0, 2), (3, 1)])
def test_hermite_gauss_widths(grid, l, m):
    """sigma_x**2 = (2l + 1)*w**2/4 for HG_lm, so d_x = 2*w*sqrt(2l + 1)."""

    x, y, X, Y = grid
    I = np.abs(gauss_hermite(l, m, x, y, 0.0, W0, Z_R, WL))**2
    widths = beam_widths(I, x, y, iterate=False)

    assert np.allclose([widths.d_x, widths.d_y], [2*W0*np.sqrt(2*l + 1), 2*W0*np.sqrt(2*m + 1)], rtol=1E-6)


def test_background_subtraction(grid):
    x, y, X, Y = grid
    I = np.exp(-2*(X**2 + Y**2)/20.0**2)

    assert beam_widths(I + 0.01, x, y, background=0.01).d_x == pytest.approx(40.0)
    # without subtraction, the offset over the window inflates the width:
    assert beam_widths(I + 0.01, x, y, iterate=False).d_x > 2*40.0


def test_iteration_suppresses_noise(grid):
    x, y, X, Y = grid
    rng = np.random.default_rng(0)
    I = np.exp(-2*(X**2 + Y**2)/20.0**2) + 0.002*rng.standard_normal(X.shape)

    assert beam_widths(I, x, y).d_x == pytest.approx(40.0, rel=0.02)
    assert abs(beam_widths(I, x, y, iterate=False).d_x - 40.0) > abs(beam_widths(I, x, y).d_x - 40.0)


@pytest.mark.parametrize("l, m", [(0, 0), (1, 0), (2, 1)])
def test_caustic_fit_gives_M2(grid, l, m):
    """HG_lm has M²_x = 2l + 1 and M²_y = 2m + 1."""

    x, y, X, Y = grid
    z_0 = 50.0
    z = z_0 + Z_R*np.array([-3, -2, -1, -0.5, -0.2, 0.0, 0.3, 0.6, 1.0, 2.0, 2.5, 3.0])
    widths = [beam_widths(np.abs(gauss_hermite(l, m, x, y, zi - z_0, W0, Z_R, WL))**2, x, y) for zi in z]

    for d, M2_expected in (([w.d_x for w in widths], 2*l + 1), ([w.d_y for w in widths], 2*m + 1)):
        fit = fit_caustic(z, d, WL)
        assert fit.M2 == pytest.approx(M2_expected, rel=1E-3)
        assert fit.z_0 == pytest.approx(z_0, abs=1E-3*Z_R)
        assert fit.d_0 == pytest.approx(2*W0*np.sqrt(M2_expected), rel=1E-3)
        assert fit.z_R == pytest.approx(Z_R, rel=1E-3)  # same Rayleigh length for all HG modes
        assert np.allclose(fit.d(z), d, rtol=1E-3)


def test_caustic_fit_rejects_non_waist():
    with pytest.raises(ValueError):
        fit_caustic([0, 1, 2, 3], [1, 2, 2, 1], WL)


def test_moments_with_mask():
    x = y = np.arange(5.0)
    I = np.zeros((5, 5))
    I[2, 1] = I[2, 3] = 1.0
    I[0, 0] = 100.0
    mask = np.ones((5, 5))
    mask[0, 0] = 0
    moments = second_moments(I, x, y, mask=mask)

    assert np.allclose(moments, [2, 2, 1, 0, 0])
    assert widths_from_moments(moments).d_x == pytest.approx(4.0)


def test_iteration_with_camera_noise_and_offset():
    """ISO window iteration on camera images (shot noise, read noise, offset
    subtracted with a dark frame) gives unbiased widths."""

    from pyoptics.processing import CameraNoiser, Scaler

    x, y, X, Y = grid2d(-128, 128, -128, 128, 256)
    beam = np.exp(-2*(X**2 + Y**2)/12.0**2)
    camera = CameraNoiser(model="camera", read_noise=5.0, offset=100.0, rng=2)
    dark = np.mean([camera(np.zeros_like(beam)) for _ in range(20)], axis=0)
    d = [beam_widths(camera(Scaler(1000.0)(beam)), x, y, background=dark).d_x for _ in range(20)]

    assert np.mean(d) == pytest.approx(24.0, rel=0.01)
    assert np.std(d) < 0.03*24.0
