#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for the quasi-conformal carpet cloak pyoptics.cloak: the map
against first-order theory, its boundary conditions, the beam source
against its angular spectrum, and rays through the cloak.

References
----------
J. Li and J. B. Pendry, "Hiding under the carpet: a new strategy for
cloaking", Phys. Rev. Lett. 101, 203901 (2008).

B. Zhang, T. Chan and B.-I. Wu, "Lateral shift makes a ground-plane cloak
detectable", Phys. Rev. Lett. 104, 233903 (2010).
"""

import numpy as np
import pytest

from pyoptics import cloak


def _map(h0=0.2, boundary="mirror"):
    return cloak.CarpetCloak(cloak.cos2_bump(h0, 1.5), 6.0, 3.0, n_xi=241, n_eta=61, boundary=boundary)


def test_flat_mirror_is_identity():
    c = cloak.CarpetCloak(lambda x: 0*np.asarray(x, dtype=float), 4.0, 2.0, n_xi=81, n_eta=41)
    assert np.allclose(c.u, c.x, atol=1e-10) and np.allclose(c.v, c.y, atol=1e-10)
    assert np.allclose(c.n2_grid, 1, atol=1e-8) and np.allclose(c.alpha, 1, atol=1e-6)


def test_boundary_conditions():
    c = _map(boundary="mirror")
    assert np.allclose(c.v[:, 0], 0) and np.allclose(c.v[:, -1], c.H)
    assert c.boundary_mismatch() == pytest.approx((0, 0), abs=1e-10)
    s = _map(boundary="sliding")
    assert s.boundary_mismatch()[0] > 1e-3          # the top slides
    f = _map(boundary="fixed")
    assert np.allclose(f.u[:, 0], f.x[:, 0])
    # sliding along the mirror keeps the map nearly conformal, pinning it does not
    assert c.alpha.max() < 1.05 < f.alpha.max()


def test_linear_theory_second_order():
    devs = []
    for h0 in (0.05, 0.1):
        c = _map(h0)
        sl = (slice(60, 181, 3), slice(None, None, 3))
        devs.append(np.abs(c.n2_grid[sl] - c.linear_n2(c.x[sl], c.y[sl])).max())
    assert 3.2 < devs[1]/devs[0] < 4.8


def test_index_outside_and_clipping():
    c = _map(0.4)
    assert c.permittivity(np.array([7.0, 0.0]), np.array([1.0, 4.0])) == pytest.approx([1, 1])
    assert c.n2_grid.min() < 1
    cc = cloak.CarpetCloak(cloak.cos2_bump(0.4, 1.5), 6.0, 3.0, n_xi=241, n_eta=61, n_min=1.0)
    X, Y = c.x[:, 1:-1], c.y[:, 1:-1]
    assert cc.permittivity(X, Y).min() >= 1 - 1e-3


def test_force_against_numerical_gradient():
    c = _map(0.3)
    r = np.array([[0.3, 0.8], [-1.0, 1.5], [1.7, 0.4]])
    e = 1e-5
    num = np.stack([(c.n2(r + [e, 0]) - c.n2(r - [e, 0]))/(4*e),
                    (c.n2(r + [0, e]) - c.n2(r - [0, e]))/(4*e)], axis=-1)
    assert np.allclose(c.force(r), num, atol=1e-4)


def test_rays_normal_incidence_symmetric():
    c = _map(0.3)
    q = np.array([-1.0, 1.0])
    x_end, ex, _ = c.trace(q, 4.0, 0.0, 3.5, step=0.02)
    assert x_end[0] == pytest.approx(-x_end[1], abs=1e-6)
    assert ex[0] == pytest.approx(-ex[1], abs=1e-8)


def test_beam_sources_spectrum():
    from pyoptics import fdfd, fdtd
    wl, dx = 1.0, 0.05
    grid = fdtd.Grid((16.0, 8.0), dx, center=(0.0, 0.0))
    sim = fdfd.Simulation(grid)
    xs, ys = sim.coords("Ez")
    y_src = ys[np.argmin(np.abs(ys - 2.5))]
    srcs, u_inc = cloak.beam_sources(xs[np.abs(xs) < 7], y_src, wl, np.radians(30), 2.0, spacing=dx)
    u = sim.solve(wl, srcs).main
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    m = (np.abs(X) < 5) & (Y > -2.5) & (Y < 1.5)
    ue = u_inc(X[m], Y[m])
    assert np.linalg.norm(u[m] - ue)/np.linalg.norm(ue) < 0.05
