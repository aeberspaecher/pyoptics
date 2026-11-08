#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pytest

from pyoptics import speckle as sp


LAM, DX, N, NA = 0.5, 0.5, 256, 0.05


def _speckle(rng, sigma_h=10.0, wavelength=LAM, h=None):
    if h is None:
        h = sp.rough_surface(N, DX, sigma_h, 0.5, rng)
    return np.abs(sp.coherent_image(np.exp(4j*np.pi*h/wavelength), DX, NA, LAM))**2, h


def test_rough_surface_statistics():
    h = sp.rough_surface(512, 0.1, 2.0, 1.0, rng=0)
    assert np.isclose(h.std(), 2.0)
    assert abs(h.mean()) < 1e-12
    # correlation at one correlation length is about 1/e
    c = np.mean(h*np.roll(h, 10, axis=1))/h.var()
    assert c == pytest.approx(np.exp(-1), abs=0.08)


def test_fully_developed_speckle_contrast_and_autocorrelation():
    rng = np.random.default_rng(0)
    I, _ = _speckle(rng)
    assert sp.speckle_contrast(I) == pytest.approx(1.0, abs=0.06)
    r, g = sp.intensity_autocorrelation(I, DX)
    g_th = sp.speckle_autocorrelation(r, NA, LAM)
    assert g[0] == pytest.approx(2.0, abs=0.15)
    assert np.allclose(g[:15], g_th[:15], atol=0.12)


def test_speckle_pdf_is_normalized_with_the_right_moments():
    I = np.linspace(0, 40, 40001)
    for M in (1, 2, 5.5):
        p = sp.speckle_pdf(I, 2.0, M)
        assert np.trapezoid(p, I) == pytest.approx(1.0, rel=1e-3)
        mean = np.trapezoid(I*p, I)
        var = np.trapezoid((I - mean)**2*p, I)
        assert mean == pytest.approx(2.0, rel=1e-3)
        assert np.sqrt(var)/mean == pytest.approx(1/np.sqrt(M), rel=1e-3)


def test_summed_contrast_eigenvalues_and_limits():
    assert sp.summed_contrast(np.eye(9)) == pytest.approx(1/3)
    assert sp.summed_contrast(np.ones((4, 4))) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5)) + 1j*rng.standard_normal((5, 5))
    J = A @ A.conj().T
    d = np.sqrt(np.real(np.diag(J)))
    mu = J/np.outer(d, d)
    lam = np.linalg.eigvalsh(mu)
    assert sp.summed_contrast(mu) == pytest.approx(np.sqrt(np.sum(lam**2))/np.sum(lam))


def test_polarization_contrast():
    assert sp.polarization_contrast(1.0) == pytest.approx(1.0)
    assert sp.polarization_contrast(0.0) == pytest.approx(1/np.sqrt(2))


def test_wavelength_correlation_matches_simulation():
    rng = np.random.default_rng(2)
    I1, h = _speckle(rng)
    dl = 0.001
    I2, _ = _speckle(rng, wavelength=LAM + dl, h=h)
    rho = np.corrcoef(I1.ravel(), I2.ravel())[0, 1]
    assert rho == pytest.approx(sp.wavelength_correlation(dl, LAM, 20.0), abs=0.06)


def test_spectral_contrast_gaussian_and_sampled_agree():
    s = 0.002
    lam = LAM + np.linspace(-4*s, 4*s, 161)
    S = np.exp(-(lam - LAM)**2/(2*s**2))
    C_gauss = sp.spectral_contrast(s, LAM, 20.0)
    C_sampled = sp.spectral_contrast(None, LAM, 20.0, lam, S)
    assert C_sampled == pytest.approx(C_gauss, rel=0.01)
    assert sp.spectral_contrast(0.0, LAM, 20.0) == pytest.approx(1.0)


def test_angular_diversity_limits():
    assert sp.angular_diversity_contrast(0.0, NA, LAM) == 1.0
    assert sp.angular_diversity_contrast(0.005*NA, NA, LAM, n=256) == pytest.approx(1.0, abs=0.01)
    C = sp.angular_diversity_contrast(8*NA, NA, LAM)
    assert C*8 == pytest.approx(sp.ANGULAR_LIMIT, rel=0.03)


def test_moving_diffuser_matches_simulation():
    rng = np.random.default_rng(3)
    NA_ill = 2*NA
    C_inf = sp.angular_diversity_contrast(NA_ill, NA, LAM)
    for M in (1, 8):
        cs = []
        for _ in range(3):
            h = sp.rough_surface(N, DX, 10.0, 0.5, rng)
            r = np.exp(4j*np.pi*h/LAM)
            I = sum(np.abs(sp.coherent_image(r*sp.diffuser_field(N, DX, NA_ill, LAM, rng),
                                             DX, NA, LAM))**2 for _ in range(M))
            cs.append(sp.speckle_contrast(I))
        assert np.mean(cs) == pytest.approx(float(sp.moving_diffuser_contrast(M, C_inf)), rel=0.08)


def test_diffuser_field_fills_the_aperture():
    u = sp.diffuser_field(N, DX, 0.1, LAM, rng=4)
    assert np.mean(np.abs(u)**2) == pytest.approx(1.0)
    F = np.abs(np.fft.fft2(u))**2
    fx, fy = sp.frequency_grid(N, DX)
    assert np.all(F[np.hypot(fx, fy) > 0.1/LAM + 1e-9] < 1e-12*F.max())
    assert np.all(sp.diffuser_field(8, DX, 0.0, LAM) == 1)
