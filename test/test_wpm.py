#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the wave propagation method against analytical results and the
homogeneous propagators.

All lengths are given in units of the wavelength (wl = 1). Every check runs
with both implementations of the WPM sum ("direct" and "levels"), which must
agree to rounding errors.
"""

import numpy as np
import pytest

from pyoptics.propagators import rayleigh_sommerfeld_I_TF
from pyoptics.utils import grid1d, grid2d

wpm = pytest.importorskip("pyoptics.wpm")


WL = 1.0
K0 = 2*np.pi/WL
METHODS = ("direct", "levels")


def _homogeneous(n):
    return lambda X, Y, z: n*np.ones_like(X)


@pytest.mark.parametrize("method", METHODS)
def test_homogeneous_matches_rayleigh_sommerfeld(method):
    """In a homogeneous medium, the WPM is the transfer function propagator."""

    x, y, X, Y = grid2d(-12, 12, -12, 12, 48, assume_periodicity=True)
    field = np.exp(-(X**2 + 2*Y**2)/2.0**2)*np.exp(0.7j*X)
    n, dz = 1.5, 3.0

    U = wpm.wpm_propagate(_homogeneous(n), field, x, y, 0.0, dz, WL, method=method)
    U_rs, _, _ = rayleigh_sommerfeld_I_TF(field, x, y, dz, WL, n=n)

    assert np.allclose(U, U_rs, atol=1E-12)


@pytest.mark.parametrize("method", METHODS)
def test_one_dimensional_field(method):
    """A 1D field equals the y-invariant 2D problem."""

    x = grid1d(-16, 16, 128)
    u0 = np.exp(-(x - 1)**2/1.5**2)
    n_wg = lambda X, Y, z: np.where(np.abs(X) <= 3, 1.5, 1.46)

    u = wpm.wpm_propagate(n_wg, u0, x, None, 0.0, 2.0, WL, method=method)

    y2 = grid1d(-1, 1, 2)
    U = wpm.wpm_propagate(n_wg, np.tile(u0, (2, 1)), x, y2, 0.0, 2.0, WL, method=method)

    assert u.shape == x.shape
    assert np.allclose(u, U[0], atol=1E-12)
    assert np.allclose(U[0], U[1], atol=1E-12)


@pytest.mark.parametrize("method", METHODS)
def test_tilted_plane_wave_off_center_grid(method):
    """The result must not depend on where the grid lies: a tilted plane wave
    on a window that does not contain x = 0 propagates exactly."""

    L, N, dz, n = 16.0, 32, 2.0, 1.3
    x = 5.0 + np.arange(N)*L/N
    y = -3.0 + np.arange(N)*L/N
    X, Y = np.meshgrid(x, y)
    kx, ky = 2*np.pi*3/L, -2*np.pi*2/L
    field = np.exp(1j*(kx*X + ky*Y))

    U = wpm.wpm_propagate(_homogeneous(n), field, x, y, 0.0, dz, WL, method=method)
    kz = np.sqrt((n*K0)**2 - kx**2 - ky**2)

    assert np.allclose(U, field*np.exp(1j*kz*dz), atol=1E-12)


@pytest.mark.parametrize("method", METHODS)
def test_absorbing_medium(method):
    """n + i*kappa with kappa > 0 damps a plane wave by exp(-k0 kappa dz)."""

    x = grid1d(-8, 8, 32)
    n, dz = 1.5 + 0.02j, 4.0

    u = wpm.wpm_propagate(_homogeneous(n), np.ones_like(x), x, None, 0.0, dz, WL, method=method)

    assert np.allclose(u, np.exp(1j*K0*n*dz))
    assert np.all(np.abs(u) < 1)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n", (1.0, 1.0 - 0.0j, 1.0 + 1E-3j))
def test_evanescent_wave_decays(method, n):
    """A plane wave with k_x > n k0 decays, it must never grow."""

    L, N, dz = 8.0, 64, 0.5
    x = grid1d(-L/2, L/2, N)
    kx = 2*np.pi*12/L   # 1.5 k0
    u0 = np.exp(1j*kx*x)

    u = wpm.wpm_propagate(_homogeneous(n), u0, x, None, 0.0, dz, WL, method=method)
    kz = np.sqrt(complex((n*K0)**2 - kx**2))
    kz = kz if kz.imag >= 0 else -kz

    assert np.allclose(u, u0*np.exp(1j*kz*dz), atol=1E-12)
    assert np.abs(u).max() < np.exp(-0.5*np.sqrt(kx**2 - K0**2)*dz)


@pytest.mark.parametrize("method", METHODS)
def test_direct_and_levels_agree(method):
    """Both implementations compute the same sum, also for a complex index
    with several levels in 2D."""

    x, y, X, Y = grid2d(-6, 6, -5, 5, 24, 20, assume_periodicity=True)
    field = np.exp(-(X**2 + Y**2)/4)*np.exp(1j*(X - 0.5*Y))
    n_func = lambda X, Y, z: np.select([X**2 + Y**2 < 4, X > 3], [1.6 + 0.01j, 1.2], 1.0)

    U = wpm.wpm_propagate(n_func, field, x, y, 0.0, 0.7, WL, method=method)
    U_auto = wpm.wpm_propagate(n_func, field, x, y, 0.0, 0.7, WL)

    assert np.allclose(U, U_auto, atol=1E-13)


def test_slab_phase():
    """A plane wave through a slab picks up the optical path length. With the
    index sampled in the middle of the slice, slab faces on slice boundaries
    are exact; sampling at the start shifts the slab by one slice."""

    x = grid1d(-4, 4, 16)
    slab = lambda X, Y, z: np.where((z > 1.0) & (z < 3.0), 1.5, 1.0)*np.ones_like(X)
    z_vals = np.linspace(0, 4, 9)

    U = wpm.wpm_propagate_z(slab, np.ones_like(x), x, None, z_vals, WL)
    assert np.allclose(U[-1], np.exp(1j*K0*(2*1.0 + 2*1.5)))
    assert np.allclose(U[4], np.exp(1j*K0*(1.0 + 1.5)))

    U_start = wpm.wpm_propagate_z(slab, np.ones_like(x), x, None, z_vals, WL, n_sampling="start")
    assert np.allclose(U_start[4], np.exp(1j*K0*(1.5 + 0.5*1.5)))


def test_averaged_index_sampling():
    """n_sampling=int averages the index over the slice: a normally incident
    plane wave sees the mean index of a linear gradient in z."""

    x = grid1d(-4, 4, 16)
    n_func = lambda X, Y, z: (1.0 + 0.1*z)*np.ones_like(X)
    u = wpm.wpm_propagate(n_func, np.ones_like(x), x, None, 2.0, 1.0, WL, n_sampling=3)

    assert np.allclose(u, np.exp(1j*K0*1.25))


def test_averaging_z_sampling_polynomial():
    """Gauss-Legendre with N nodes is exact for polynomials of degree 2N-1."""

    X, Y = np.meshgrid(np.arange(3.0), np.arange(2.0))
    func = lambda X, Y, z: X*z**3 + Y*z
    z1, z2 = 1.0, 3.0

    avg = wpm.averaging_z_sampling(func, X, Y, z1, z2, 2)
    expected = X*(z2**4 - z1**4)/4/(z2 - z1) + Y*(z2**2 - z1**2)/2/(z2 - z1)

    assert np.allclose(avg, expected)


def test_grin_fundamental_mode():
    """The Gaussian w_m = sqrt(2/(k0 n0 g)) is the (paraxial) fundamental mode
    of the parabolic profile n0 (1 - g^2 x^2/2) and keeps its shape."""

    n0, P = 1.5, 150.0
    g = 2*np.pi/P
    w_m = np.sqrt(2/(K0*n0*g))
    x = grid1d(-24, 24, 192)
    n_grin = lambda X, Y, z: n0*(1 - g**2*X**2/2)
    u0 = np.exp(-x**2/w_m**2)
    z_vals = np.linspace(0, P/4, 76)

    U = wpm.wpm_propagate_z(n_grin, u0, x, None, z_vals, WL)
    I = np.abs(U)**2

    assert np.allclose(I/I.max(axis=1, keepdims=True), np.abs(u0)**2, atol=2E-2)
    # the WPM is not strictly unitary in inhomogeneous media, but for a smooth
    # lossless profile the power is nearly conserved:
    assert np.allclose(I.sum(axis=1), I[0].sum(), rtol=1E-4)


def test_wpm_propagate_z_matches_single_steps():
    x, y, X, Y = grid2d(-4, 4, -4, 4, 16, assume_periodicity=True)
    field = np.exp(-(X**2 + Y**2)/2)
    n_func = lambda X, Y, z: np.where(X**2 + Y**2 < (1 + 0.2*z)**2, 1.4, 1.0)
    z_vals = np.array([0.0, 0.3, 1.0, 1.2])

    U = wpm.wpm_propagate_z(n_func, field, x, y, z_vals, WL)

    assert U.shape == (4, 16, 16)
    assert np.allclose(U[0], field)
    u = field
    for z, z_next, U_k in zip(z_vals[:-1], z_vals[1:], U[1:]):
        u = wpm.wpm_propagate(n_func, u, x, y, z, z_next - z, WL)
        assert np.allclose(U_k, u)


def test_input_validation():
    x = grid1d(-4, 4, 16)
    y = grid1d(-4, 4, 8)
    n_func = _homogeneous(1.0)

    with pytest.raises(ValueError):
        wpm.wpm_propagate(n_func, np.ones((16, 8)), x, y, 0.0, 1.0, WL)
    with pytest.raises(ValueError):
        wpm.wpm_propagate(n_func, np.ones(8), x, None, 0.0, 1.0, WL)
    with pytest.raises(ValueError):
        wpm.wpm_propagate(n_func, np.ones(16), x, None, 0.0, 1.0, WL, method="fast")
    with pytest.raises(ValueError):
        wpm.wpm_propagate(n_func, np.ones(16), x, None, 0.0, 1.0, WL, n_sampling="end")


def test_max_sampling_distance():
    """At dx = wl/(2 n), the Nyquist frequency is the largest propagating
    frequency n/wl."""

    dx = wpm.max_sampling_distance(0.5, 1.5 + 0.1j)
    assert np.isclose(1/(2*dx), 1.5/0.5)
