#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for the Shack-Hartmann sensor (shack_hartmann).
"""

import numpy as np
import pytest

from pyoptics.bases import fringe_zernike
from pyoptics.shack_hartmann import ShackHartmann, Slopes


WL = 0.5e-3


def sensor(**kwargs):
    return ShackHartmann(16, 0.15, 15.0, WL, **kwargs)


def zernike_phase(coeffs):
    return lambda P_x, P_y: 2*np.pi*sum(c*fringe_zernike(j, P_x, P_y) for j, c in coeffs.items())


def test_geometry():
    sh = sensor()

    assert sh.samples % 2 == 0 and sh.samples >= max(16, 2*sh.fresnel_number)
    assert len(sh.x) == 16*sh.samples
    assert np.allclose(sh.centers, sh.centers[::-1]*-1)
    assert np.isclose(sh.pupil_radius, 1.2)
    assert np.isclose(sh.max_slope, 0.15/30)
    assert np.isclose(sh.slope_per_pixel, 0.15/16/15)
    assert np.isclose(np.mean(sh.x), 0)


def test_plane_wave_spots_are_centred_and_conserve_power():
    sh = sensor()
    img = sh.spots(sh.field(amplitude=1.0))
    cx, cy, flux = sh.centroids(img)

    assert np.allclose(cx, 0, atol=1e-12) and np.allclose(cy, 0, atol=1e-12)
    # a square lenslet sends about 87 % of its power into a window of
    # +-pitch/2 at Fresnel number 3
    assert 0.85 < img.sum() <= 1.0
    assert np.allclose(flux, flux[0, 0])


@pytest.mark.parametrize("sx, sy", [(1e-3, 0.0), (0.0, -2e-3), (1.5e-3, 1e-3)])
def test_tilt_moves_spots_by_f_times_slope(sx, sy):
    sh = sensor()
    k = 2*np.pi/WL
    R = sh.pupil_radius
    img = sh.spots(sh.field(lambda P_x, P_y: k*R*(sx*P_x + sy*P_y), amplitude=1.0))
    s = sh.slopes(img, threshold=0.1)

    assert np.all(s.valid)
    assert np.allclose(s.sx, sx, atol=0.02*sh.slope_per_pixel)
    assert np.allclose(s.sy, sy, atol=0.02*sh.slope_per_pixel)


def test_model_slopes_of_a_tilt():
    sh = sensor()
    sx, sy = sh.model_slopes(zernike_phase({2: 1.0}))

    # Z2 = P_x: W = wavelength*x/R
    assert np.allclose(sx[np.isfinite(sx)], WL/sh.pupil_radius)
    assert np.allclose(sy[np.isfinite(sy)], 0, atol=1e-15)
    assert np.isnan(sx[0, 0])                       # dark corner lenslet


def test_reference_image_removes_a_common_tilt():
    sh = sensor()
    k = 2*np.pi/WL
    tilt = lambda P_x, P_y: k*sh.pupil_radius*2e-3*P_x
    reference = sh.spots(sh.field(tilt))
    s = sh.slopes(reference, reference=reference)

    assert np.allclose(s.sx[s.valid], 0) and np.allclose(s.sy[s.valid], 0)


def test_modal_reconstruction_of_model_slopes_is_exact():
    sh = sensor()
    true = {2: 0.3, 3: -0.2, 4: 0.5, 5: 0.2, 6: -0.3, 7: 0.15, 8: 0.1, 9: 0.2, 11: -0.05, 15: 0.05}
    sx, sy = sh.model_slopes(zernike_phase(true))
    valid = np.isfinite(sx)
    c = sh.reconstruct_zernikes(Slopes(sx, sy, valid, None), 15)

    assert sorted(c) == list(range(2, 16))
    assert np.allclose([c[j] for j in c], [true.get(j, 0.0) for j in c], atol=1e-10)


def test_end_to_end_zernikes_and_zonal():
    sh = sensor()
    true = {2: 0.3, 3: -0.2, 4: 0.5, 5: 0.2, 6: -0.3, 7: 0.15, 8: 0.1, 9: 0.2, 11: -0.05, 15: 0.05}
    phase = zernike_phase(true)
    s = sh.slopes(sh.spots(sh.field(phase)))
    c = sh.reconstruct_zernikes(s, 15)
    err = np.array([c[j] - true.get(j, 0.0) for j in c])

    assert np.sqrt(np.sum(err**2)) < 0.01                 # waves

    W = sh.reconstruct_zonal(s)
    C_x, C_y = np.meshgrid(sh.centers/sh.pupil_radius, sh.centers/sh.pupil_radius)
    W_true = phase(C_x, C_y)/(2*np.pi)
    d = (W - W_true)[s.valid]
    d -= d.mean()

    assert np.all(np.isnan(W[~s.valid]))
    assert np.sqrt(np.mean(d**2)) < 0.02
    assert np.isclose(np.nanmean(W), 0, atol=1e-12)


def test_zonal_reconstruction_of_a_quadratic_is_exact():
    # Southwell's averaging is exact for quadratic wavefronts
    sh = sensor()
    a, b = 2e-3, -1e-3                                  # W = a*x**2/2 + b*x*y
    C_x, C_y = np.meshgrid(sh.centers, sh.centers)
    valid = C_x**2 + C_y**2 < 1.0
    sx = np.where(valid, a*C_x + b*C_y, np.nan)
    sy = np.where(valid, b*C_x, np.nan)
    W = sh.reconstruct_zonal(Slopes(sx, sy, valid, None))
    W_true = (a*C_x**2/2 + b*C_x*C_y)/WL
    W_true -= W_true[valid].mean()

    assert np.allclose(W[valid], W_true[valid], atol=1e-8)


def test_min_flux_excludes_rim_lenslets():
    sh = sensor()
    img = sh.spots(sh.field())
    loose, strict = sh.slopes(img, min_flux=0.1), sh.slopes(img, min_flux=0.99)

    assert strict.valid.sum() < loose.valid.sum()
    assert np.all(loose.valid[strict.valid])
    assert np.isclose(loose.flux.max(), 1.0)
