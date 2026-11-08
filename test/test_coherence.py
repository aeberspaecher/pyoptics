#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.coherence against analytic coherence functions."""

import numpy as np
import pytest

from pyoptics import coherence
from pyoptics.propagators import fresnel_scaled
from pyoptics.utils import grid2d


WL = 550e-9


def test_uniform_disk_first_zero():
    theta = 47e-3/206265
    B0 = coherence.uniform_disk_first_zero(theta, WL)
    assert B0 == pytest.approx(1.2197*WL/theta, rel=1e-4)
    assert abs(coherence.uniform_disk_visibility(B0, theta, WL)) < 1e-12
    assert coherence.uniform_disk_visibility(0.0, theta, WL) == pytest.approx(1.0)
    # minimum of 2 J1(v)/v
    v = np.linspace(4, 6, 20001)
    mu = coherence.uniform_disk_visibility(v*WL/(np.pi*theta), theta, WL)
    assert mu.min() == pytest.approx(-0.1323, abs=1e-4)


@pytest.mark.parametrize("u", [0.0, 0.4, 1.0])
def test_limb_darkened_disk_against_numerical_vcz(u):
    theta = 1e-7
    x, y, X, Y = grid2d(-0.6*theta, 0.6*theta, -0.6*theta, 0.6*theta, 600)
    rr = np.sqrt(X**2 + Y**2)/(theta/2)
    I = np.where(rr < 1, 1 - u*(1 - np.sqrt(np.clip(1 - rr**2, 0, None))), 0.0)
    B = np.linspace(0, 3*WL/theta, 31)
    mu_num = coherence.mutual_intensity_vcz(I, x, y, WL, B, np.array([0.0]))[0]
    mu = coherence.limb_darkened_disk_visibility(B, theta, WL, u)
    assert np.max(np.abs(mu_num - mu)) < 3e-3
    if u == 0:
        np.testing.assert_allclose(mu, coherence.uniform_disk_visibility(B, theta, WL), atol=1e-12)


def test_binary():
    mu = coherence.binary_visibility(np.linspace(0, 11, 2201)*WL/1e-7, 1e-7, 0.5, WL)
    assert np.abs(mu).max() == pytest.approx(1.0)
    assert np.abs(mu).min() == pytest.approx(1/3, abs=1e-3)


def test_vcz_by_random_realizations():
    """Average of propagated random-phase sources against van Cittert-Zernike."""
    rng = np.random.default_rng(1)
    wl, z, a = 0.5e-3, 1000.0, 0.1                     # mm
    xs, ys, Xs, Ys = grid2d(-0.15, 0.15, -0.15, 0.15, 96)
    I_src = (Xs**2 + Ys**2 < a**2).astype(float)
    x_out = np.linspace(0, 6, 25)
    y_out = np.array([0.0])
    J = 0
    I_mean = 0
    for _ in range(400):
        u, *_ = fresnel_scaled(coherence.incoherent_source_realization(I_src, rng), xs, ys, z, wl,
                               x_out, y_out)
        J = J + u[0]*np.conj(u[0, 0])
        I_mean = I_mean + np.abs(u[0])**2
    mu = J/np.sqrt(I_mean*I_mean[0])
    mu_red = mu*np.exp(-1j*np.pi*x_out**2/(wl*z))    # remove the quadratic phase psi
    mu_th = coherence.uniform_disk_visibility(x_out, 2*a/z, wl)
    assert np.max(np.abs(mu_red - mu_th)) < 0.08


def test_degree_of_coherence_gaussian_line():
    dnu = 1.0
    nu = np.linspace(-5, 5, 4001)
    S = np.exp(-4*np.log(2)*nu**2/dnu**2)
    tau = np.linspace(-3, 3, 601)
    gamma = coherence.degree_of_coherence(nu, S, tau)
    np.testing.assert_allclose(np.abs(gamma), coherence.line_coherence(tau, dnu), atol=1e-6)
    assert coherence.coherence_time(tau, gamma) == pytest.approx(
        coherence.coherence_time_analytic(dnu), rel=1e-4)


@pytest.mark.parametrize("shape", coherence.SHAPES)
def test_chaotic_light_siegert(shape):
    rng = np.random.default_rng(2)
    dt, dnu = 1.0, 0.02
    u = coherence.chaotic_light(2**16, dt, dnu, shape, rng, size=8)
    lags = np.arange(101)
    g1 = coherence.g1(u, 100)
    g2 = coherence.g2(np.abs(u)**2, 100)
    np.testing.assert_allclose(np.abs(g1), coherence.line_coherence(lags*dt, dnu, shape), atol=0.03)
    np.testing.assert_allclose(g2, 1 + np.abs(g1)**2, atol=0.05)
    assert g2[0] == pytest.approx(2.0, abs=0.05)


def test_correlated_pair():
    rng = np.random.default_rng(3)
    mu = 0.6*np.exp(0.7j)
    u1, u2 = coherence.correlated_chaotic_pair(2**18, 1.0, 0.05, mu, rng=rng)
    assert np.mean(u1*np.conj(u2)) == pytest.approx(mu, abs=0.02)
    c = coherence.intensity_correlation(np.abs(u1)**2, np.abs(u2)**2)
    assert c == pytest.approx(abs(mu)**2, abs=0.03)


def test_fit_fringes():
    x = np.linspace(0, 10, 500)
    V, phi = coherence.fit_fringes(x, 3*(1 + 0.4*np.cos(2*np.pi*x/1.3 - 0.5)), 1.3)
    assert V == pytest.approx(0.4)
    assert phi == pytest.approx(0.5)
    V, phi = coherence.fit_fringes(x, 1 - 0.1*np.cos(2*np.pi*x/1.3), 1.3)
    assert V == pytest.approx(0.1)
    assert abs(phi) == pytest.approx(np.pi)
