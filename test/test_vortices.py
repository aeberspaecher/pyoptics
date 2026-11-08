#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the optical vortex tools against analytical results: charges and OAM
of LG modes, the spiral spectrum of multilevel and fractional plates, vortex
splitting, and the orders of fork gratings.
"""

import numpy as np
import pytest

from pyoptics import modes, vortices
from pyoptics.utils import grid2d


WL = 0.5
W = 10.0


@pytest.fixture(scope="module")
def grid():
    return grid2d(-40, 40, -40, 40, 256)


@pytest.mark.parametrize("l", [-3, -1, 1, 2, 4])
def test_lg_charge_and_oam(grid, l):
    x, y, _, _ = grid
    u = modes.lg_mode(0, l, x, y, W, WL, z=3.0)
    assert vortices.topological_charge(u, x, y, 8.0) == pytest.approx(l, abs=1e-6)
    _, _, q = vortices.phase_singularities(u, x, y)
    assert q.sum() == l
    assert vortices.mean_oam(u, x, y) == pytest.approx(l, abs=1e-6)
    assert vortices.mean_oam(u, x, y, method="difference") == pytest.approx(l, rel=1e-2)
    ls, P = vortices.oam_spectrum(u, x, y, 6)
    assert P[ls == l][0] == pytest.approx(1.0, abs=1e-6)


def test_oam_about_shifted_axis(grid):
    # a displaced LG_01: the spiral spectrum broadens, the mean OAM stays 1
    x, y, _, _ = grid
    u = modes.lg_mode(0, 1, x, y, W, WL, x0=0.5*W)
    ls, P = vortices.oam_spectrum(u, x, y, 12)
    assert P[ls == 1][0] < 0.8
    assert np.sum(P) == pytest.approx(1.0, abs=1e-3)
    assert vortices.mean_oam(u, x, y) == pytest.approx(1.0, abs=1e-6)
    assert vortices.mean_oam(u, x, y, x0=0.5*W) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("N", [4, 8])
def test_multilevel_plate_spectrum(grid, N):
    # N steps: only l = 1 + q*N occur, with weights sinc^2((1 + q*N)/N)
    x, y, _, _ = grid
    g = modes.hg_mode(0, 0, x, y, W, WL)
    ls, P = vortices.oam_spectrum(g*vortices.spiral_phase_plate(x, y, 1, levels=N), x, y, 2*N,
                                  n_phi=1024)
    for m in (1, 1 - N, 1 + N):
        assert P[ls == m][0] == pytest.approx(np.sinc(m/N)**2, abs=5e-3)
    assert P[ls == 2][0] < 1e-3


def test_fractional_plate_spectrum(grid):
    x, y, _, _ = grid
    a = 1.5
    g = modes.hg_mode(0, 0, x, y, W, WL)
    ls, P = vortices.oam_spectrum(g*vortices.spiral_phase_plate(x, y, a), x, y, 5, n_phi=1024)
    assert np.allclose(P, np.sin(np.pi*a)**2/(np.pi**2*(a - ls)**2), atol=2e-3)


def test_splitting_of_charge_3(grid):
    # LG_03 + eps*LG_00 at the waist: three unit vortices at r = w/sqrt(2)*(eps*sqrt(3!))**(1/3)
    x, y, _, _ = grid
    eps = 0.05
    u = modes.lg_mode(0, 3, x, y, W, WL) + eps*modes.lg_mode(0, 0, x, y, W, WL)
    xs, ys, q = vortices.phase_singularities(u, x, y, mask=np.hypot(*np.meshgrid(x, y)) < 2*W)
    assert list(q) == [1, 1, 1]
    r = W/np.sqrt(2)*(eps*np.sqrt(6))**(1/3)
    assert np.allclose(np.hypot(xs, ys), r, atol=0.5*(x[1] - x[0]))


def test_plate_height():
    x = np.array([-1.0, 1.0])
    h = vortices.spiral_plate_height(x, np.array([-1e-9, 1e-9]), 2, 0.6328, 1.457)
    assert h.max() == pytest.approx(2*0.6328/0.457, rel=1e-6)
    assert h.min() >= 0


@pytest.mark.parametrize("kind, expected", [
    ("binary", {0: 0.25, 1: 1/np.pi**2, 2: 0.0, 3: 1/(9*np.pi**2)}),
    ("sinusoidal", {0: 0.25, 1: 1/16, 2: 0.0}),
    ("blazed", {0: 0.0, 1: 1.0, -1: 0.0}),
    ("binary_phase", {0: 0.0, 1: 4/np.pi**2, 2: 0.0}),
])
def test_fork_grating_orders(kind, expected):
    # far field of a broad Gaussian through the fork grating: order m at
    # m/period carries the charge m*l; efficiencies relative to the incident
    # power (a sampled blaze of 16 samples per period reaches sinc(1/16)**2)
    N, period, l = 512, 16, 2
    x, y, X, Y = grid2d(-N/2, N/2, -N/2, N/2, N)
    g = np.exp(-(X**2 + Y**2)/(N/5)**2)
    t = vortices.fork_grating(x, y, l, period, kind=kind)
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g*t)))
    P_in = np.sum(np.abs(g)**2)*N**2
    f = np.fft.fftshift(np.fft.fftfreq(N, 1.0))
    for m, eta in expected.items():
        near = np.hypot(*np.meshgrid(f - m/period, f)) < 0.5/period
        assert np.sum(np.abs(F[near])**2)/P_in == pytest.approx(eta, abs=0.025)
        if eta > 0.05:
            q = vortices.topological_charge(F, f, f, 3/N, x0=m/period, n=360)
            assert q == pytest.approx(m*l, abs=1e-6)


def test_orbital_current_of_vortex(grid):
    # azimuthal current l/r*|u|^2
    x, y, X, Y = grid
    l = 2
    u = modes.lg_mode(0, l, x, y, W, WL)
    jx, jy = vortices.orbital_current(u, x, y)
    R = np.hypot(X, Y)
    ring = (R > 0.5*W) & (R < 1.5*W)
    j_phi = (-Y*jx + X*jy)[ring]/R[ring]
    assert np.allclose(j_phi, l/R[ring]*np.abs(u[ring])**2, rtol=1e-6,
                       atol=1e-8*np.abs(u).max()**2)
