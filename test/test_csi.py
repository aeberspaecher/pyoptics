#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.csi."""

import numpy as np
import pytest

from pyoptics import csi


K = np.linspace(2*np.pi/0.75, 2*np.pi/0.45, 60)
S = csi.led_spectrum(K, 0.57, 0.1)
Z = np.arange(-3.0, 3.0, 0.04)
H = np.array([0.0, 0.3137, -1.2345])


def test_pupil_quadrature_and_obliquity():
    for NA in (0.1, 0.55, 0.95):
        c, w = csi.pupil_quadrature(NA, 24)
        assert np.sum(w) == pytest.approx(1.0)
        assert np.sum(w*c) == pytest.approx(csi.obliquity_factor(NA), rel=1e-10)
    assert csi.obliquity_factor(0.0) == 1.0
    assert csi.obliquity_factor(1e-4) == pytest.approx(1 - 1e-8/4)


def test_coherence_length():
    I = csi.correlogram(Z, np.linspace(5, 20, 400), csi.led_spectrum(np.linspace(5, 20, 400),
                                                                      0.57, 0.05))
    env = csi.envelope(I)
    above = Z[env > env.max()/2]
    assert above[-1] - above[0] == pytest.approx(csi.coherence_length(0.57, 0.05), rel=0.06)


@pytest.mark.parametrize("NA", [0.0, 0.5, 0.9])
def test_heights_from_envelope_and_fda(NA):
    I = csi.correlogram(Z, K, S, H, NA=NA)
    assert csi.envelope_peak(Z, I) == pytest.approx(H, abs=3e-3)
    assert csi.envelope_peak(Z, I, "parabola") == pytest.approx(H, abs=3e-3)
    hc, hp, K0 = csi.fda(Z, I, (2*K.min()*np.sqrt(1 - NA**2)*0.9, 2*K.max()*1.05))
    assert hc == pytest.approx(H, abs=2e-3)
    assert hp == pytest.approx(H, abs=3e-5)
    # the fringe frequency is reduced by the obliquity factor
    assert np.mean(K0)/(2*np.sum(S*K)/np.sum(S)) == pytest.approx(csi.obliquity_factor(NA),
                                                                  rel=0.03)


def test_film_reflectance_bare_and_coated():
    r_s, r_p = csi.film_reflectance([], 1.5)
    assert r_s(10.0, np.array([1.0])) == pytest.approx(-0.2)
    assert r_p(10.0, np.array([1.0])) == pytest.approx(-0.2)
    # a quarter wave MgF2 layer on glass lowers the reflectance
    r_s, _ = csi.film_reflectance([(1.38, 0.55/(4*1.38))], 1.52)
    assert abs(r_s(2*np.pi/0.55, np.array([1.0]))[0])**2 < 0.015
    # reflectance callable lists average in correlogram()
    I1 = csi.correlogram(Z, K, S, 0.0, reflectance=-0.2)
    I2 = csi.correlogram(Z, K, S, 0.0, reflectance=csi.film_reflectance([], 1.5))
    assert I1 == pytest.approx(I2)


def test_metal_phase_offset():
    # Al and glass differ in the phase of r: the phase heights differ by phi/(2 k)
    hp = []
    for n in (1.5, 1.0 + 6.5j):
        I = csi.correlogram(Z, K, S, 0.0, reflectance=csi.film_reflectance([], n))
        hp.append(csi.fda(Z, I, (2*K.min()*0.9, 2*K.max()*1.05))[1])
    r_al = (1 - (1.0 + 6.5j))/(1 + (1.0 + 6.5j))
    k0 = np.sum(S*K)/np.sum(S)
    assert hp[1] - hp[0] == pytest.approx((np.angle(r_al) - np.pi)/(2*k0), abs=2e-3)


def test_profile_scan_flat_and_step():
    x = np.arange(256)*0.1
    z = np.arange(-2.0, 2.5, 0.04)
    P = csi.profile_scan(x, 0.2 + 0*x, z, K[::3], S[::3], 0.4, n_illumination=5)
    assert np.std(P, axis=1).max() < 1e-12
    assert csi.envelope_peak(z, P.T) == pytest.approx(0.2, abs=1e-4)
    h = np.where(np.abs(x - 12.8) < 6.4, 0.3, 0.0)
    P = csi.profile_scan(x, h, z, K[::3], S[::3], 0.4, n_illumination=5)
    _, hp, _ = csi.fda(z, P.T, (2*K.min()*0.85, 2*K.max()*1.05))
    assert hp[128] == pytest.approx(0.3, abs=2e-3) and hp[5] == pytest.approx(0.0, abs=2e-3)
