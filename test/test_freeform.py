#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pytest
from scipy.special import erf

from pyoptics import freeform as ff
from pyoptics import raytrace as rt


def test_radial_map_gaussian_to_flat_top():
    w, R0 = 2.0, 3.0
    r = np.linspace(0, 2*w, 2001)
    R = np.linspace(0, R0, 2001)
    Rr = ff.radial_map(r, np.exp(-2*r**2/w**2), R, np.ones_like(R))
    expected = R0*np.sqrt((1 - np.exp(-2*r**2/w**2))/(1 - np.exp(-8.0)))
    assert np.allclose(Rr, expected, atol=2e-3)
    # identity for equal profiles
    assert np.allclose(ff.radial_map(r, np.ones_like(r), r, np.ones_like(r)), r, atol=1e-9)


def test_fermi_dirac_edges():
    assert ff.fermi_dirac(0.0, 1.0, 20) == pytest.approx(1.0, abs=1e-8)
    assert ff.fermi_dirac(1.0, 1.0, 20) == pytest.approx(0.5)
    assert ff.fermi_dirac(5.0, 1.0, 1e4) == 0.0


def _shaper_system(w=2.0, R0=3.0, n=1.5, D=100.0):
    r = np.linspace(0, 2*w, 1001)
    Rg = np.linspace(0, 1.6*R0, 2001)
    Rr = ff.radial_map(r, np.exp(-2*r**2/w**2), Rg, ff.fermi_dirac(Rg, R0, 30))
    z1, z2 = ff.flattop_shaper(r, Rr, n, D)
    return rt.System([rt.Surface(np.inf, 5.0, n_after=n, semi_diameter=5.0),
                      ff.TabulatedSurface(r=r, z=z1, thickness=D, n_after=1.0),
                      ff.TabulatedSurface(r=Rr, z=z2, thickness=5.0, n_after=n),
                      rt.Surface(np.inf, 10.0, n_after=1.0)], 0, 550e-6, object_distance=10.0)


def test_flattop_shaper_collimates_flattens_and_keeps_the_wavefront_plane():
    w, R0 = 2.0, 3.0
    S = _shaper_system(w, R0)
    rng = np.random.default_rng(0)
    xy = rng.normal(0, w/2, (100_000, 2))
    xy = xy[np.hypot(xy[:, 0], xy[:, 1]) < 2*w*0.999]
    N = len(xy)
    B = S.trace(rt.Rays(np.column_stack([xy, np.full(N, -10.0)]), np.tile([0, 0, 1.0], (N, 1)),
                        np.zeros(N)))
    D_out, P_out = B.directions[-1], B.points[-1]
    assert np.all(np.isfinite(D_out))
    assert np.max(np.hypot(D_out[:, 0], D_out[:, 1])) < 1e-4
    opl = B.opl[-1]
    assert np.ptp(opl) < 1e-4                      # mm: well below a wave
    r_out = np.hypot(P_out[:, 0], P_out[:, 1])
    h, e = np.histogram(r_out, bins=12, range=(0, 0.8*R0))
    E = h/(np.pi*(e[1:]**2 - e[:-1]**2))
    assert np.std(E)/np.mean(E) < 0.05


def test_integrate_gradient_recovers_a_surface():
    x = np.linspace(-1, 1, 41)
    y = np.linspace(-1.2, 1.2, 51)
    X, Y = np.meshgrid(x, y, indexing="ij")
    z = 0.3*X**2 - 0.2*X*Y + 0.1*Y**3
    gx = 0.6*X - 0.2*Y
    gy = -0.2*X + 0.3*Y**2
    zr = ff.integrate_gradient(gx, gy, x[1] - x[0], y[1] - y[0])
    assert np.allclose(zr - zr.mean(), z - z.mean(), atol=2e-3)


def test_ot_map_is_separable_for_product_densities():
    # Gaussian to a uniform square: the optimal map is a product of the 1D maps
    n, A, w, H = 40, 3.0, 2.0, 5.0
    x = np.linspace(-A, A, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    a = np.exp(-2*(X**2 + Y**2)/w**2)
    y = np.linspace(-H, H, n)
    b = np.ones((n, n))
    Tx, Ty, err = ff.ot_map(a, (x, x), b, (y, y))
    assert err < 1e-6
    # 1D map of the truncated Gaussian onto [-H - dy/2, H + dy/2]
    edge = H + (y[1] - y[0])/2
    c = erf(np.sqrt(2)*x/w)/erf(np.sqrt(2)*A/w)
    T1 = edge*c
    inner = np.abs(x) < 0.8*A
    assert np.allclose(Tx[inner][:, inner.nonzero()[0]], T1[inner][:, None], atol=0.35)
    assert np.allclose(Tx, Ty.T, atol=1e-6)


def test_freeform_lens_lands_the_rays_on_the_map():
    n, A, L = 32, 3.0, 100.0
    x = np.linspace(-A, A, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Tx, Ty = 4*X + 0.3*X**2 - 0.2*Y**2, 4*Y - 0.4*X*Y      # a gradient map
    z = ff.freeform_lens(x, x, Tx, Ty, 1.5, L)
    S = rt.System([rt.Surface(np.inf, 5.0, n_after=1.5, semi_diameter=10.0),
                   ff.TabulatedSurface(x=x, y=x, z=z, thickness=L, n_after=1.0)],
                  0, 550e-6, object_distance=10.0)
    sel = (slice(4, -4), slice(4, -4))
    P = np.column_stack([X[sel].ravel(), Y[sel].ravel(), np.full(X[sel].size, -10.0)])
    B = S.trace(rt.Rays(P, np.tile([0, 0, 1.0], (len(P), 1)), np.zeros(len(P))))
    Q = B.points[-1]
    err = np.hypot(Q[:, 0] - Tx[sel].ravel(), Q[:, 1] - Ty[sel].ravel())
    assert np.max(err) < 0.05


def test_tabulated_radial_surface_matches_a_sphere():
    R = 50.0
    r = np.linspace(0, 10, 401)
    z = R - np.sqrt(R**2 - r**2)
    tab = rt.System([rt.Surface(np.inf, 5.0, n_after=1.5, semi_diameter=10.0),
                     ff.TabulatedSurface(r=r, z=z, thickness=80.0, n_after=1.0)],
                    0, 550e-6, object_distance=10.0)
    sph = rt.System([rt.Surface(np.inf, 5.0, n_after=1.5, semi_diameter=10.0),
                     rt.Surface(R, 80.0, n_after=1.0, semi_diameter=10.0)],
                    0, 550e-6, object_distance=10.0)
    y0 = np.linspace(-8, 8, 9)
    P = np.column_stack([y0, 0.3*y0, np.full(9, -10.0)])
    d = np.tile([0, 0, 1.0], (9, 1))
    Qa = tab.trace(rt.Rays(P, d, np.zeros(9))).points[-1]
    Qb = sph.trace(rt.Rays(P, d, np.zeros(9))).points[-1]
    assert np.allclose(Qa, Qb, atol=1e-5)
