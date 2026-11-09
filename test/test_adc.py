#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.adc: air index, refraction against closed forms and a
GRIN trace, Amici prisms and the dispersion corrector."""

import numpy as np
import pytest

from pyoptics import adc, glasses, materials


def test_edlen_index():
    lam = np.array([400.0, 589.3, 900.0])
    # standard air: Edlén (Birch and Downs) and Ciddor agree to a few 1e-9
    assert np.allclose(adc.edlen_index(lam), materials.air_index(lam), atol=5e-9)
    # n - 1 scales with the density
    n1 = adc.edlen_index(550.0, 15.0, 101325.0) - 1
    n2 = adc.edlen_index(550.0, 15.0, 74300.0) - 1
    assert n2/n1 == pytest.approx(74300/101325, rel=1e-3)
    # water vapour lowers the index slightly
    f = adc.water_vapour_pressure(15.0, 0.5)
    assert f == pytest.approx(853, rel=0.01)
    assert adc.edlen_index(550.0, f=f) - adc.edlen_index(550.0) == pytest.approx(-3.1e-7, rel=0.02)


def test_refraction():
    n0 = adc.edlen_index(550.0)
    H = adc.scale_height(15.0)
    z = np.radians([10.0, 45.0, 60.0, 75.0])
    R = adc.refraction_integral(z, n0, H)
    assert R[1]/adc.ARCSEC == pytest.approx(57.2, abs=0.1)
    assert np.allclose(adc.refraction_laplace(z[:3], n0, H), R[:3], rtol=1e-4)
    assert np.allclose(adc.refraction_flat(z[:2], n0), R[:2], rtol=3e-3)
    # GRIN trace through the spherical atmosphere
    Rt = adc.trace_refraction(z, n0, H, step=100.0)
    assert np.allclose(Rt, R, atol=1e-4*adc.ARCSEC)


def test_dispersion():
    z = np.radians(60.0)
    lam = np.array([400.0, 900.0])
    dR = adc.atmospheric_dispersion(lam, z)
    assert dR[0] > 0 > dR[1]
    n = adc.edlen_index(lam)
    first = (n - adc.edlen_index(550.0))*np.tan(z)
    assert np.allclose(dR, first, rtol=0.01)
    ex = adc.atmospheric_dispersion(lam, z, exact=True)
    assert np.allclose(dR, ex, rtol=1e-3)


def test_amici_and_adc():
    g1, g2 = glasses.get("N-KF9"), glasses.get("N-PK51")
    lam = np.linspace(400, 900, 26)
    pr = adc.AmiciPrism(g1, g2, np.radians(4.0))
    assert np.allclose(pr.deviation_exact(lam), pr.deviation(lam), rtol=0.02, atol=1e-6)
    # counter-rotated by 90 degrees the two prisms cancel
    assert np.allclose(adc.adc_dispersion(pr, lam, np.pi/2), 0, atol=1e-15)
    # a pair matched to the air dispersion leaves a small residual
    best = adc.best_glass_pairs({"N-KF9": g1, "N-PK51": g2, "N-BK7": glasses.get("N-BK7"),
                                 "N-F2": glasses.get("N-F2")}, lam, max_delta_n=0.2, k=2)
    assert {best[0][1], best[0][2]} == {"N-KF9", "N-PK51"}
    z = np.radians(50.0)
    dR = adc.atmospheric_dispersion(lam, z)
    M = 82.0
    phi, res = adc.adc_residual(pr, lam, dR, M)
    assert 0 < phi < np.pi/2
    assert np.std(res) < 0.05*np.std(dR)


def test_smeared_star():
    x = np.linspace(-3, 3, 121)
    img = adc.smeared_star(x, x, [500.0, 600.0], [1.0, 1.0], [0.5, -0.5], 0.5)
    assert img.sum() == pytest.approx(1.0)
    yc = (img.sum(axis=1)*x).sum()
    assert yc == pytest.approx(0.0, abs=1e-12)
