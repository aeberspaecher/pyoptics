#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.coronagraph.

References
----------
R. Soummer, C. Aime and P. E. Falloon, Astron. Astrophys. 397, 1161 (2003)
(prolate apodization); D. Mawet et al., Astrophys. J. 633, 1191 (2005)
(vortex Lyot plane fields).
"""

import numpy as np
import pytest
from scipy.special import j0, j1

from pyoptics import coronagraph as cg
from pyoptics.bases import fringe_zernike


@pytest.fixture(scope="module")
def grid():
    x = cg.pupil_grid(160, 1.25)       # 128 samples across D
    P = cg.telescope_pupil(x, x, oversample=16)
    return x, P


def test_pupil_grid_and_area(grid):
    x, P = grid
    assert np.allclose(x, -x[::-1])
    assert np.isclose(x[1] - x[0], 1/128)
    assert np.sum(P)*(x[1] - x[0])**2 == pytest.approx(np.pi/4, rel=1e-4)
    Po = cg.telescope_pupil(x, x, obscuration=0.3, oversample=8)
    assert np.sum(Po)/np.sum(P) == pytest.approx(1 - 0.09, rel=1e-3)
    Ps = cg.telescope_pupil(x, x, n_spiders=4, spider_width=0.02, oversample=8)
    assert np.sum(P - Ps)*(x[1] - x[0])**2 == pytest.approx(4*0.5*0.02, rel=0.05)   # vanes overlap at the centre


def test_mft_airy(grid):
    x, P = grid
    u = np.linspace(0.0, 6.0, 61)
    F = cg.mft(P, x, x, u, [0.0])[0]
    assert F[0].real == pytest.approx(np.pi/4, rel=1e-4)
    I = np.abs(F/F[0])**2
    airy = np.ones_like(u)
    airy[1:] = (2*j1(np.pi*u[1:])/(np.pi*u[1:]))**2
    assert np.max(np.abs(I - airy)) < 2e-4
    # a tilted wave is imaged at u = alpha
    us = np.linspace(2, 4, 81)
    I = np.abs(cg.mft(P*cg.tilt(x, x, 3.0), x, x, us, [0.0])[0])**2
    assert us[np.argmax(I)] == pytest.approx(3.0, abs=0.03)
    # inverse transform returns the pupil
    uf = (np.arange(-512, 512) + 0.5)/16
    back = cg.mft(cg.mft(P, x, x, uf, uf), uf, uf, x, x, inverse=True)
    assert np.abs(back[80, 80] - 1) < 0.02


def test_airy_envelope():
    u = np.linspace(20, 40, 20001)
    mean = np.mean((2*j1(np.pi*u)/(np.pi*u))**2*u**3)
    assert mean == pytest.approx(4/np.pi**4, rel=0.01)


def test_sonine(grid):
    x, P = grid
    u = np.linspace(0.1, 8, 40)
    for nu in (0, 1, 2):
        A = cg.sonine_apodizer(x, x, nu)
        F = cg.mft(P*A, x, x, np.r_[0.0, u], [0.0])[0]
        I = np.abs(F[1:]/F[0])**2
        assert np.allclose(I, cg.sonine_psf(u, nu), atol=3e-4)
        dA = (x[1] - x[0])**2
        assert np.sum(P*A**2)/np.sum(P) == pytest.approx(1/(2*nu + 1), rel=3e-3)
        assert np.abs(np.sum(P*A)*dA)**2/(np.pi/4)**2 == pytest.approx(1/(nu + 1)**2, rel=3e-3)
    assert cg.sonine_psf(0.0, 2) == pytest.approx(1.0)


def test_prolate_energy(grid):
    x, P = grid
    pro, A = cg.prolate_apodizer(1.0, x, x)
    assert 0 < pro.eigenvalue < 1
    assert np.max(pro.values) == pytest.approx(1.0)
    ug = cg.pupil_grid(128, 2.0)
    F = cg.mft(P*A, x, x, ug, ug)
    disk = cg.telescope_pupil(ug, ug, D=2.0, oversample=8)
    ee = np.sum(np.abs(F)**2*disk)*(ug[1] - ug[0])**2/(np.sum((P*A)**2)*(x[1] - x[0])**2)
    assert ee == pytest.approx(pro.eigenvalue, abs=1e-3)
    # larger disks concentrate more
    assert cg.prolate_apodizer(2.0).eigenvalue > pro.eigenvalue


def test_aplc_field_is_scaled_apodizer(grid):
    x, P = grid
    X, Y = np.meshgrid(x, x)
    pro, A = cg.prolate_apodizer(1.0, x, x)
    EL = cg.occulter_lyot_field(P*A, x, x, 1.0)
    inside = np.hypot(X, Y) < 0.45
    assert np.median(EL[inside].real/A[inside]) == pytest.approx(1 - pro.eigenvalue, rel=0.02)


def test_lyot_field_analytic(grid):
    x, P = grid
    c = len(x)//2
    for a in (1.0, 3.0):
        EL = cg.occulter_lyot_field(P, x, x, a)
        assert cg.occulter_lyot_analytic(0.0, a) == pytest.approx(j0(np.pi*a), abs=1e-10)
        assert EL[c, c].real == pytest.approx(j0(np.pi*a), abs=2e-3)
        r = np.abs(x[c:])
        ana = cg.occulter_lyot_analytic(r, a)
        assert np.max(np.abs(EL[c, c:].real - ana)[r < 0.45]) < 2e-3
    # a transparent disk changes nothing
    assert np.allclose(cg.occulter_lyot_field(P, x, x, 2.0, transmission=1.0), P)


def test_vortex_lyot_plane(grid):
    x, P = grid
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    for l in (2, 4):
        EL = cg.vortex_lyot_field(P, x, x, l)
        assert np.max(np.abs(EL[R < 0.45])**2) < 5e-6
        ana = cg.vortex_lyot_analytic(R, l)*np.exp(1j*l*np.arctan2(Y, X))
        ring = (R > 0.53) & (R < 0.6)
        assert np.max(np.abs(EL - ana)[ring]) < 0.01
    assert np.allclose(cg.vortex_lyot_analytic([0.2, 0.6], 0), [1, 0])
    with pytest.raises(ValueError):
        cg.vortex_lyot_analytic(0.6, 6)


def test_vortex_tilt_leak(grid):
    x, P = grid
    stop = cg.telescope_pupil(x, x, D=0.95, oversample=8)
    for l, alpha, law in ((2, 0.02, np.pi**2*0.02**2/8*0.95**4), (4, 0.1, np.pi**4*0.1**4/192*0.95**6)):
        co = cg.Coronagraph(x, x, P, "vortex", charge=l, lyot_stop=stop)
        EL = co.lyot_field(cg.tilt(x, x, alpha))
        leak = np.sum(np.abs(EL*stop)**2)/np.sum(P**2)
        assert leak == pytest.approx(law, rel=0.05)


def test_vortex_selection_rule(grid):
    x, P = grid
    X, Y = np.meshgrid(x, x)
    co = cg.Coronagraph(x, x, P, "vortex", charge=4, lyot_stop=cg.telescope_pupil(x, x, D=0.95))
    eps = 0.01
    for j, blocked in ((4, True), (5, False)):           # defocus (n=2, m=0), astigmatism (n=2, m=2)
        Z = fringe_zernike(j, X/0.5, Y/0.5)
        Z /= np.sqrt(np.sum(Z**2*P)/np.sum(P))
        leak = np.sum(np.abs(co.lyot_field(np.exp(1j*eps*Z))*co.lyot_stop)**2)/np.sum(P**2)/eps**2
        assert (leak < 0.01) == blocked
    assert cg.vortex_blocks(4, 1, 1) and cg.vortex_blocks(6, 2, 2) and not cg.vortex_blocks(6, 3, 3)
    assert not cg.vortex_blocks(2, 1, 1) and cg.vortex_blocks(2, 0, 0)


def test_coronagraph_normalization(grid):
    x, P = grid
    tel = cg.Coronagraph(x, x, P)
    assert tel.image([0.0], [0.0])[0, 0] == pytest.approx(1.0)
    assert tel.throughput == pytest.approx(1.0)
    stop = cg.telescope_pupil(x, x, D=0.8)
    co = cg.Coronagraph(x, x, P, "lyot", mask_radius=3.0, lyot_stop=stop)
    assert co.throughput == pytest.approx(0.64, rel=0.01)
    # a planet far from the disk keeps its peak
    us = np.linspace(9, 11, 41)
    assert co.image(us, [0.0], cg.tilt(x, x, 10.0)).max() == pytest.approx(1.0, abs=0.03)
    with pytest.raises(ValueError):
        cg.Coronagraph(x, x, P, "disk")


def test_radial_profile():
    u = cg.pupil_grid(64, 16)
    U, V = np.meshgrid(u, u)
    I = np.hypot(U, V)
    edges = np.arange(0, 8.5, 1.0)
    rc, p = cg.radial_profile(I, u, u, edges)
    a, b = edges[:-1], edges[1:]
    assert np.allclose(p, 2/3*(b**3 - a**3)/(b**2 - a**2), atol=0.05)   # area weighted mean radius
    _, pm = cg.radial_profile(I, u, u, np.arange(0, 8.5, 1.0), stat="max")
    assert np.all(pm >= p)
    _, ph = cg.radial_profile(I, u, u, [100, 101])
    assert np.isnan(ph[0])


def test_aberration_screen_variance():
    x = cg.pupil_grid(128, 2.0)
    K = 1e-4
    rng = np.random.default_rng(0)
    f = np.fft.fftfreq(128, x[1] - x[0])
    FX, FY = np.meshgrid(f, f)
    expected = np.sum(K*(1 + FX**2 + FY**2)**-1.5)/(2.0**2)
    var = np.mean([np.var(cg.aberration_screen(x, x, K, 3.0, 1.0, rng)) for _ in range(40)])
    assert var == pytest.approx(expected, rel=0.25)


def test_dm_and_efc(grid):
    x, P = grid
    H, centres = cg.dm_influence(x, x, 8, coupling=0.2)
    pitch = 1/8
    k = np.argmin(np.hypot(*centres.T))
    c0 = centres[k]
    ix = np.argmin(np.abs(x - (c0[0] + pitch)))
    iy = np.argmin(np.abs(x - c0[1]))
    assert H[k, iy, ix] == pytest.approx(0.2, abs=0.03)   # nearest grid point
    assert np.all(np.hypot(*centres.T) <= 0.5 + 0.5*pitch + 1e-12)
    # EFC on a linear toy problem: one step cancels a field in the range of G
    rng = np.random.default_rng(1)
    G = rng.standard_normal((40, 10)) + 1j*rng.standard_normal((40, 10))
    c_true = rng.standard_normal(10)
    E = -G @ c_true
    c = cg.efc_step(G, E, regularization=1e-9)
    assert np.allclose(c, c_true, atol=1e-5)


def test_efc_jacobian_matches_finite_difference(grid):
    x, P = grid
    co = cg.Coronagraph(x, x, P, "lyot", mask_radius=2.0, lyot_stop=cg.telescope_pupil(x, x, D=0.9))
    H, _ = cg.dm_influence(x, x, 6)
    u = cg.pupil_grid(24, 12)
    U, V = np.meshgrid(u, u)
    sel = (np.hypot(U, V) > 3) & (U > 0)
    G = cg.efc_jacobian(co, 1.0, H[:5], u, u, sel, batch=2)
    h = 1e-4
    fd = (co.image_field(u, u, np.exp(1j*h*H[3]))[sel] - co.image_field(u, u)[sel])/h
    assert np.allclose(G[:, 3], fd, atol=1e-3*np.max(np.abs(fd)))
