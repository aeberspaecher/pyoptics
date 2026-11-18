#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the Forbes Q aspheres (notebook 128, Forbes Q aspheres)."""

import numpy as np
import pytest
from scipy.special import eval_jacobi

from pyoptics import qaspheres as qa
from pyoptics.raytrace import System, Surface


def test_qbfs_low_orders_match_forbes():
    x = np.linspace(0, 1, 11)
    assert np.allclose(qa.qbfs(0, x), 1.0)
    assert np.allclose(qa.qbfs(1, x), (13 - 16*x)/np.sqrt(19))
    q2 = np.sqrt(2/95)*(29 - 4*x*(25 - 19*x))
    assert np.allclose(qa.qbfs(2, x), q2)


def test_qbfs_slope_orthonormality():
    # (2/pi) int_0^1 S_m' S_n' du/sqrt(1-u^2) = delta_mn with S_m = u^2 (1-u^2) Q_m(u^2)
    N = 400
    u = np.cos(np.pi*(np.arange(N) + 0.5)/(2*N))
    h = 1e-6
    S = lambda m, u: u**2*(1 - u**2)*qa.qbfs(m, u**2)
    dS = [(S(m, u + h) - S(m, u - h))/(2*h) for m in range(6)]
    G = np.array([[np.mean(a*b) for b in dS] for a in dS])
    assert np.allclose(G, np.eye(6), atol=1e-6)


def test_qcon_is_jacobi():
    x = np.linspace(0, 1, 7)
    for m in range(5):
        assert np.allclose(qa.qcon(m, x), eval_jacobi(m, 0, 4, 2*x - 1))


def test_qbfs_surface_sag_vertex_and_rim():
    R, rn = 30.0, 10.0
    s = qa.QbfsSurface(R, 1.0, (0.05, -0.01, 0.002), rn)
    sphere = lambda r: r**2/R/(1 + np.sqrt(1 - r**2/R**2))
    # the departure vanishes at the vertex and at the rim: both on the best fit sphere
    assert abs(s.sag(np.array([0.0]))[0]) < 1e-14
    assert np.isclose(s.sag(np.array([rn**2]))[0], sphere(rn), atol=1e-12)
    # the vertex curvature includes the departure: c = c_bfs + 2 sum a_m Q_m(0)/rn^2
    S0 = sum(a*qa.qbfs(m, 0.0) for m, a in enumerate(s.q))
    assert np.isclose(s.c, 1/R + 2*S0/rn**2)
    # paraxial: sag ~ c r^2/2
    r = 1e-3
    assert np.isclose(s.sag(np.array([r**2]))[0], s.c*r**2/2, rtol=1e-6)


def test_qcon_surface_matches_conic_plus_departure():
    c, K, rn = 1/25.0, -0.8, 8.0
    a = (2e-3, -5e-4)
    r = np.linspace(0, rn, 9)
    x = (r/rn)**2
    expected = c*r**2/(1 + np.sqrt(1 - (1 + K)*c**2*r**2)) + x**2*(a[0]*qa.qcon(0, x) + a[1]*qa.qcon(1, x))
    assert np.allclose(qa.sag_qcon(r, c, K, rn, a), expected, atol=1e-12)


def test_fit_qbfs_recovers_coefficients_and_is_stable():
    rn = 12.0
    a = np.array([0.02, -0.004, 1e-3, -2e-4])
    ref = qa.QbfsSurface(40.0, 0.0, a, rn)
    sag = lambda r: ref.sag(np.asarray(r)**2)
    fit = qa.fit_qbfs(sag, rn, n_terms=6)
    assert np.isclose(fit.R, 40.0, rtol=1e-9)
    assert np.allclose(fit.q[:4], a, atol=1e-7)
    assert np.allclose(fit.q[4:], 0, atol=1e-7)
    fit3 = qa.fit_qbfs(sag, rn, n_terms=3)
    assert np.allclose(fit3.q, fit.q[:3], atol=1e-9)
    # rms slope = sqrt(sum a^2)/rn
    assert np.isclose(qa.rms_slope(sag, rn), np.sqrt(np.sum(a**2))/rn, rtol=1e-3)


def test_fit_even_asphere_and_fourth_order():
    c, K, A = 1/20.0, -1.2, (2e-5, -3e-8)
    rn = 7.0
    sag = lambda r: qa.even_asphere_sag(r, c, K, A)
    fit = qa.fit_qbfs(sag, rn, n_terms=10)
    r = np.linspace(0, rn, 31)
    assert np.allclose(fit.sag(r**2), sag(r), atol=1e-7)
    con = qa.fit_qcon(sag, rn, c, K, n_terms=8)
    assert np.allclose(con.sag(r**2), sag(r), atol=1e-9)
    # the fourth order term (Seidel) agrees with the even asphere
    a4 = K*c**3/8 + A[0]
    assert np.isclose(con.fourth_order_departure(), a4, rtol=1e-6)
    assert np.isclose(fit.fourth_order_departure(), a4, rtol=1e-3)


def test_qsurface_traces_like_even_asphere():
    c, K, A = 1/30.0, -0.6, (1e-5, -2e-8)
    rn = 8.0
    sag = lambda r: qa.even_asphere_sag(r, c, K, A)
    q = qa.fit_qbfs(sag, rn, n_terms=10)
    q = qa.QbfsSurface(q.R, 6.0, q.q, rn, n_after=1.5168)
    e = Surface(1/c, 6.0, conic=K, aspheric=A, n_after=1.5168)
    back = [Surface(-60.0, 50.0)]
    s1 = System([Surface(np.inf, 10.0, semi_diameter=rn), q] + back, 0, 587.56e-6)
    s2 = System([Surface(np.inf, 10.0, semi_diameter=rn), e] + back, 0, 587.56e-6)
    p1, p2 = s1.paraxial(), s2.paraxial()
    assert np.isclose(p1.efl, p2.efl, rtol=1e-9)
    for s in (s1, s2):
        s.surfaces[-1].thickness = p2.image_distance
    w1 = s1.opd(0.0, N=32).W
    w2 = s2.opd(0.0, N=32).W
    assert np.nanmax(np.abs(w1 - w2)) < 1e-3   # waves


def test_q_coefficient_variable():
    s = System([Surface(np.inf, 1.0, semi_diameter=4.0), qa.QbfsSurface(30.0, 2.0, (0.01,), 5.0, n_after=1.5),
                Surface(-30.0, 40.0)], 0, 587.56e-6)
    v = qa.QCoefficient(1, 2)
    assert v.get(s) == 0.0
    v.set(s, 0.003)
    assert v.get(s) == 0.003
    assert s.surfaces[1].q == (0.01, 0.0, 0.003)


def _q2d_terms():
    return [(0, 0, "cos"), (0, 2, "cos"), (1, 0, "cos"), (1, 1, "sin"), (2, 0, "cos"),
            (2, 1, "sin"), (3, 0, "cos"), (3, 2, "cos"), (4, 1, "sin")]


def test_q2d_gradients_are_orthonormal():
    # (1/pi^2) int int grad f . grad g du dphi / sqrt(1 - u^2): Gauss-Chebyshev in u
    n = 300
    u = np.cos(np.pi*(np.arange(n) + 0.5)/(2*n))
    phi = np.linspace(0, 2*np.pi, 97)[:-1]
    U, PHI = np.meshgrid(u, phi)
    grads = []
    for m, k, kind in _q2d_terms():
        terms = {(m, k): 1.0}
        S = qa.Q2DSurface(np.inf, 0.0, cos=terms if kind == "cos" else None,
                          sin=terms if kind == "sin" else None, norm_radius=1.0)
        _, gx, gy = S._departure_and_gradient(U*np.cos(PHI), U*np.sin(PHI))
        grads.append((gx, gy))
    G = np.array([[np.mean(a[0]*b[0] + a[1]*b[1]) for b in grads] for a in grads])
    G *= (np.pi/2)*(2*np.pi)/np.pi**2
    assert np.allclose(G, np.eye(len(grads)), atol=1e-9)


def test_q2d_m0_is_qbfs_and_gradient_is_analytic():
    a = (0.02, -0.01, 0.005)
    q2 = qa.Q2DSurface(25.0, 0.0, cos={(0, n): v for n, v in enumerate(a)}, norm_radius=6.0)
    qb = qa.QbfsSurface(25.0, 0.0, a, 6.0)
    r = np.linspace(0, 6.0, 31)
    assert np.allclose(q2.sag_xy(r, 0*r), qb.sag(r**2), atol=1e-14)
    assert np.isclose(q2.c, qb.c)
    f = qa.Q2DSurface(20.0, 0.0, cos={(0, 1): 0.01, (2, 0): 0.02, (3, 1): -0.01, (1, 1): 0.005},
                      sin={(2, 0): 0.01, (1, 0): 0.003}, norm_radius=5.0)
    x, y = np.array([1.3, -0.4, 3.1]), np.array([-2.1, 0.7, 2.0])
    gx, gy = f._gradient(x, y)
    h = 1e-6
    assert np.allclose(gx, (f.sag_xy(x + h, y) - f.sag_xy(x - h, y))/(2*h), atol=1e-8)
    assert np.allclose(gy, (f.sag_xy(x, y + h) - f.sag_xy(x, y - h))/(2*h), atol=1e-8)


def test_q2d_surface_traces_on_its_sag():
    f = qa.Q2DSurface(-40.0, 0.0, cos={(2, 0): 0.02, (3, 0): 0.01}, sin={(1, 1): 0.004},
                      norm_radius=8.0, mirror=True)
    rng = np.random.default_rng(1)
    P = np.column_stack([rng.uniform(-6, 6, 50), rng.uniform(-6, 6, 50), np.full(50, -20.0)])
    d = np.column_stack([rng.normal(0, 0.05, 50), rng.normal(0, 0.05, 50), np.ones(50)])
    d /= np.linalg.norm(d, axis=1)[:, None]
    t = f.intersect(P, d)
    Q = P + t[:, None]*d
    assert np.allclose(Q[:, 2], f.sag_xy(Q[:, 0], Q[:, 1]), atol=1e-12)
    v = qa.Q2DCoefficient(0, 2, 0, "cos")
    assert v.name == "a2_0_0"
    with pytest.raises(ValueError):
        qa.Q2DCoefficient(0, 0, 1, "sin")


@pytest.mark.skipif(not qa._qaspheres_numba.available, reason="needs numba")
def test_compiled_intersections_match_numpy(monkeypatch):
    rng = np.random.default_rng(3)
    surfaces = [
        qa.QbfsSurface(-80.0, q=(0.02, -0.01, 0.004), norm_radius=12.0),
        qa.QconSurface(-80.0, q=(0.02, -0.01, 0.004), norm_radius=12.0, conic=-0.6),
        qa.Q2DSurface(-80.0, cos={(0, 0): 0.01, (2, 1): 0.004, (3, 0): -0.003},
                      sin={(1, 1): 0.002}, norm_radius=12.0),
        qa.QconQ2DSurface(-80.0, q=(0.01, 0.003), norm_radius=12.0, conic=-0.4,
                          cos={(2, 0): 0.003, (1, 2): 0.001}, sin={(3, 1): -0.002},
                          freeform_radius=5.0, centre=(1.0, 4.0)),
    ]
    N = 500
    P = np.stack([rng.uniform(-8, 8, N), rng.uniform(-8, 8, N), np.full(N, -20.0)], axis=-1)
    d = np.stack([rng.normal(0, 0.05, N), rng.normal(0, 0.05, N), np.ones(N)], axis=-1)
    d /= np.linalg.norm(d, axis=1)[:, None]
    for s in surfaces:
        t_compiled = s.intersect(P, d)
        monkeypatch.setattr(qa, "use_numba", False)
        t_numpy = s.intersect(P, d)
        monkeypatch.setattr(qa, "use_numba", True)
        assert np.all(np.isfinite(t_compiled))
        assert np.allclose(t_compiled, t_numpy, rtol=0, atol=1e-11)
        Q = P + t_compiled[:, None]*d
        z = s.sag_xy(Q[:, 0], Q[:, 1]) if hasattr(s, "sag_xy") else s.sag(Q[:, 0]**2 + Q[:, 1]**2)
        assert np.max(np.abs(Q[:, 2] - z)) < 1e-11


def test_q2d_hessians_are_analytic():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-6, 6, 100), rng.uniform(-6, 6, 100)
    h = 1e-5
    q2d = qa.Q2DSurface(-80.0, cos={(0, 0): 0.01, (0, 2): 0.002, (2, 1): 0.004, (3, 0): -0.003},
                        sin={(1, 1): 0.002, (2, 2): 0.001, (5, 0): 0.002}, norm_radius=8.0)
    hybrid = qa.QconQ2DSurface(-80.0, q=(0.01, 0.003), norm_radius=12.0, conic=-0.4,
                               cos={(2, 0): 0.003, (1, 2): 0.001, (0, 1): 0.002}, sin={(3, 1): -0.002},
                               freeform_radius=5.0, centre=(1.0, 4.0))
    for H, gradient in ((q2d.sag_hessian, q2d._gradient),
                        (hybrid.departure_hessian, hybrid.departure_gradient)):
        gx1, gy1 = gradient(x + h, y)
        gx0, gy0 = gradient(x - h, y)
        gx3, gy3 = gradient(x, y + h)
        gx2, gy2 = gradient(x, y - h)
        reference = ((gx1 - gx0)/(2*h), (gy1 - gy0)/(2*h), (gy3 - gy2)/(2*h))
        for a, b in zip(H(x, y), reference):
            assert np.allclose(a, b, rtol=0, atol=1e-9)
