#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.echelle: grating equation, blaze, free spectral range,
resolving power and cross-dispersion against closed forms."""

import numpy as np
import pytest

from pyoptics import echelle as ec
from pyoptics import glasses


def harps_like():
    return ec.Echelle(31.6, np.arctan(4.0))


def test_orders_and_fsr():
    E = harps_like()
    m = E.orders(378.0, 691.0)
    assert (m[0], m[-1]) == (162, 89)
    lam = E.blaze_wavelength(m)
    # Littrow on the blaze: beta = theta_B, efficiency 1
    assert np.allclose(E.beta(m, lam), E.blaze)
    assert np.allclose(E.efficiency(m, lam), 1.0)
    # free spectral ranges tile the spectrum, lambda/m wide to first order
    lo, hi = E.fsr_limits(m)
    assert np.allclose(hi[:-1], lo[1:])
    assert np.allclose(hi - lo, E.free_spectral_range(m), rtol=1e-4)
    # Littrow dispersion 2 tan(theta_B)/lambda, the same for all orders
    assert np.allclose(E.dispersion(m, lam), 2*np.tan(E.blaze)/lam)


def test_order_shares():
    E = harps_like()
    m = E.central_order(500.0)
    ms = np.arange(m - 3, m + 4)
    for lam in (E.blaze_wavelength(m) + 1.0, E.blaze_wavelength(m + 0.5)):
        ref = np.sinc(ms - E.littrow_order(lam))**2
        assert np.allclose(ec.order_efficiencies(E, lam, ms), ref, atol=2e-3)
    # at the edge of the free spectral range two orders share 4/pi^2 each
    lam = E.blaze_wavelength(m + 0.5)
    assert ec.order_efficiencies(E, lam, [m, m + 1]) == pytest.approx([4/np.pi**2]*2, abs=2e-3)
    # the exact blaze function agrees near the blaze peak
    lam = E.blaze_wavelength(m) + 0.5
    assert E.efficiency(m, lam) == pytest.approx(np.sinc(m - E.littrow_order(lam))**2, abs=2e-3)


def test_resolving_power():
    E = harps_like()
    m = E.central_order(500.0)
    lam = E.blaze_wavelength(m)
    N = 60
    delta, I = ec.staircase_far_field(E, lam, n_facets=N, samples=16, pad=32)
    near = np.abs(delta) < 2*lam/(N*E.facet)
    w = ec.fwhm(delta[near], I[near])
    R = lam/(w/E.dispersion(m, lam))
    assert R == pytest.approx(m*N/0.886, rel=2e-3)
    # slit limited and diffraction limited instrument profiles
    W = 200e6                                         # 200 mm beam, in nm
    delta = np.linspace(-1e-4, 1e-4, 200001)
    for s in (2e-5, 5e-5):
        w = ec.fwhm(delta, ec.instrument_profile(delta, s, W, lam))
        R = lam/(w/E.dispersion(m, lam))
        assert R == pytest.approx(ec.slit_resolving_power(E.blaze, 1.0, s), rel=0.01)
    w = ec.fwhm(delta, ec.instrument_profile(delta, 0.0, W, lam))
    assert lam/(w/E.dispersion(m, lam)) == pytest.approx(2*np.tan(E.blaze)*W/lam/0.886, rel=1e-3)
    # R phi product
    assert ec.slit_resolving_power(E.blaze, D_coll=0.2, D_tel=4.0, phi=np.radians(1/3600)) \
        == pytest.approx(2*4*0.2/4/np.radians(1/3600))


def test_cross_dispersion():
    E = harps_like()
    f = 700.0
    cd = ec.GratingCrossDisperser(150.0)
    m = np.array([150, 100])
    lam = E.blaze_wavelength(m)
    _, y0 = ec.order_position(E, cd, m, lam, f)
    _, y1 = ec.order_position(E, cd, m - 1, E.blaze_wavelength(m - 1), f)
    dy = y1 - y0
    assert dy == pytest.approx(f*cd.dispersion(lam)*lam/m, rel=0.02)
    # grating: spacing ~ lambda^2
    assert dy[1]/dy[0] == pytest.approx((lam[1]/lam[0])**2, rel=0.03)
    # a prism spaces the orders more evenly
    pr = ec.PrismCrossDisperser(glasses.get("N-F2"), np.radians(60.0))
    _, p0 = ec.order_position(E, pr, m, lam, f)
    _, p1 = ec.order_position(E, pr, m - 1, E.blaze_wavelength(m - 1), f)
    ratio = (p1 - p0)[1]/(p1 - p0)[0]
    assert 0.3 < abs(ratio) < dy[1]/dy[0]
    # order length 2 f tan(theta_B)/m
    lo, hi = E.fsr_limits(m)
    x_lo, _ = ec.order_position(E, cd, m, lo, f)
    x_hi, _ = ec.order_position(E, cd, m, hi, f)
    assert x_hi - x_lo == pytest.approx(2*f*np.tan(E.blaze)/m, rel=0.02)


def test_echellogram_line():
    E = harps_like()
    cd = ec.GratingCrossDisperser(150.0)
    line = lambda lam: np.exp(-0.5*((lam - 589.0)/0.003)**2)
    m = E.central_order(589.0)
    img, ext = ec.echellogram(line, E, cd, 700.0, [m], 0.015, (2000, 4000), slit_height=0.015)
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    x, y = ec.order_position(E, cd, m, 589.0, 700.0)
    assert ext[0] + (ix + 0.5)*0.015 == pytest.approx(x, abs=0.015)
    assert ext[2] + (iy + 0.5)*0.015 == pytest.approx(y, abs=0.015)
    assert img.sum() == pytest.approx(np.sqrt(2*np.pi)*0.003*E.efficiency(m, 589.0), rel=1e-3)


def test_wavelength_inverse():
    E = ec.Echelle(31.6, np.arctan(4.0), gamma=np.radians(1.0))
    lam = np.linspace(500, 504, 5)
    assert np.allclose(E.wavelength(123, E.beta(123, lam)), lam)
