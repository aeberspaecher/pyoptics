#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the illumination tools against analytical results: Lambertian
sources, the étendue estimators for uniform and Gaussian beams, the
symplectic ray map of a real lens (and the non-symplectic paraxial ideal
lens), the skew invariant, light pipes, diffusers and the concentration
limit of sunlight.
"""

import numpy as np
import pytest

from pyoptics import illumination as il
from pyoptics import raytrace as rt


def test_lambertian_directions_are_cosine_weighted():
    d = il.lambertian_directions(400000, rng=1)
    assert np.allclose(np.linalg.norm(d, axis=1), 1)
    # Lambertian: P(theta < t) = sin(t)**2
    theta = np.arccos(d[:, 2])
    for t in (0.3, 0.7, 1.2):
        assert np.mean(theta < t) == pytest.approx(np.sin(t)**2, abs=3e-3)
    d = il.lambertian_directions(1000, na=0.6, n=1.5, rng=2)
    assert np.max(np.hypot(d[:, 0], d[:, 1])) <= 0.4 + 1e-12
    with pytest.raises(ValueError):
        il.lambertian_directions(10, na=1.2)


def test_source_etendue_and_rays():
    src = il.LambertianSource(2.0, na=0.25, power=3.0)
    assert src.etendue == pytest.approx(np.pi**2*4*0.0625)
    assert src.radiance == pytest.approx(3.0/src.etendue)
    R = src.rays(1000, rng=0)
    assert R.points.shape == (1000, 3) and np.all(np.hypot(R.points[:, 0], R.points[:, 1]) <= 2)
    sq = il.LambertianSource(2.0, na=1.0, shape="square", height=1.0)
    assert sq.etendue == pytest.approx(2*np.pi)
    P = sq.points(1000, rng=0)
    assert np.all(np.abs(P[:, 0]) <= 1) and np.all(np.abs(P[:, 1]) <= 0.5)


def test_effective_etendue_uniform_2d_and_4d():
    rng = np.random.default_rng(3)
    N = 400000
    X = np.column_stack([(rng.random(N) - 0.5)*2.0, (2*rng.random(N) - 1)*0.3])
    G = il.effective_etendue(X, 200, [(-1.2, 1.2), (-0.36, 0.36)])
    assert G == pytest.approx(il.etendue_2d(2.0, 0.3), rel=2e-2)
    assert il.occupied_etendue(X, 50, [(-1, 1), (-0.3, 0.3)]) == pytest.approx(1.2)
    src = il.LambertianSource(1.0, na=0.3)
    R = src.rays(10**6, rng=4)
    X4 = il.phase_space(R.points, R.directions)
    G4 = il.effective_etendue(X4, 30, [(-1.05, 1.05)]*2 + [(-0.315, 0.315)]*2)
    # biased high by the bins on the rim of the disks
    assert src.etendue < G4 < 1.08*src.etendue


def test_effective_etendue_gaussian_and_diffuser():
    rng = np.random.default_rng(5)
    N = 500000
    sx, sp, z, sd = 1.0, 0.05, 30.0, 0.05
    x = sx*rng.standard_normal(N)
    p = sp*rng.standard_normal(N)
    rng_range = [(-6, 6), (-0.3, 0.3)]
    assert il.effective_etendue(np.column_stack([x, p]), 300, rng_range) == \
        pytest.approx(4*np.pi*sx*sp, rel=2e-2)
    # a diffuser behind free space: G = 4 pi sqrt(det Sigma)
    P = np.column_stack([x, np.zeros(N), np.zeros(N)])
    d = il.directions_from_sines(p, 0*p)
    Pz = il.propagate(P, d, z)
    d2 = il.diffuse(d, sd, rng=6)
    d2[:, 1] = 0
    d2 = il.directions_from_sines(d2[:, 0], 0*p)
    G = il.effective_etendue(np.column_stack([Pz[:, 0], d2[:, 0]]), 400, [(-12, 12), (-0.6, 0.6)])
    sxz2 = sx**2 + (z*sp)**2  # paraxial
    assert G == pytest.approx(4*np.pi*np.sqrt(sx**2*sp**2 + sd**2*sxz2), rel=3e-2)


def test_real_lens_is_symplectic_and_conserves_skew():
    S = rt.caldwell_krf_lens()
    rng = np.random.default_rng(0)
    X = np.column_stack([il.disk_points(50, 40.0, rng), il.disk_points(50, 0.12, rng)])
    f = lambda X: il.system_phase_map(S, X, z_in=-S.object_distance)
    J = il.phase_space_jacobian(f, X, h=np.array([1e-4, 1e-4, 1e-6, 1e-6]))
    assert np.allclose(np.linalg.det(J), 1, atol=1e-6)
    assert np.max(il.symplectic_defect(J)) < 1e-4
    P = np.column_stack([X[:, :2], np.full(len(X), -S.object_distance)])
    d = il.directions_from_sines(X[:, 2], X[:, 3])
    B = S.trace(rt.Rays(P, d, np.zeros(len(X))))
    h0 = il.skew_invariant(P, d)
    h1 = il.skew_invariant(B.points[-1], B.directions[-1], B.n[-1])
    assert np.allclose(h0, h1, atol=1e-10)


def test_paraxial_ideal_lens_is_not_symplectic():
    S = rt.System([rt.IdealLens(50.0, 50.0, semi_diameter=60)], 0, 550e-6)
    f = lambda X: il.system_phase_map(S, X, z_in=-1.0)
    J = il.phase_space_jacobian(f, np.array([[1.0, 0, 0, 0], [40.0, 0, 0, 0]]), h=1e-5)
    det = np.linalg.det(J)
    assert det[0] == pytest.approx(1, abs=1e-3)
    assert det[1] < 0.6


def test_stigmatic_lens_peak_concentration():
    # hyperbolic plano-convex lens: on axis, C_peak = NA'**2/sin(alpha)**2
    n, f, a = 1.5, 50.0, 25.0
    S = rt.System([rt.Surface(np.inf, 10.0, n_after=n, semi_diameter=a),
                   rt.Surface(-(n - 1)*f, f, n_after=1.0, conic=-n**2)], 0, 550e-6)
    alpha = il.SUN_ANGULAR_RADIUS
    R = il.LambertianSource(a, na=np.sin(alpha), z=-1.0).rays(400000, rng=7)
    B = S.trace(R)
    Q, e = S.to_image_frame(B.points[-1], B.directions[-1])
    na = np.nanmax(np.hypot(e[:, 0], e[:, 1]))
    r0 = 0.1*f*alpha
    C = np.mean(np.hypot(Q[:, 0], Q[:, 1]) < r0)*a**2/r0**2
    assert C == pytest.approx(na**2/np.sin(alpha)**2, rel=3e-2)
    assert C < il.concentration_limit(alpha)


def test_concentration_limit_and_temperature():
    alpha = il.SUN_ANGULAR_RADIUS
    assert il.concentration_limit(alpha) == pytest.approx(46200, rel=2e-3)
    assert il.concentration_limit(alpha, n=1.5) == pytest.approx(2.25*il.concentration_limit(alpha))
    assert il.concentration_limit(0.1, dim=2) == pytest.approx(1/np.sin(0.1))
    assert il.concentration_temperature(il.concentration_limit(alpha)) == \
        pytest.approx(il.SUN_TEMPERATURE)
    assert il.concentration_temperature(1.0) == pytest.approx(394, abs=1)
    assert il.coupling_limit(2.0, 0.5) == 0.25 and il.coupling_limit(1.0, 3.0) == 1.0


def test_light_pipe_folds_and_conserves_directions():
    rng = np.random.default_rng(8)
    src = il.LambertianSource(4.0, na=0.5, shape="square")
    R = src.rays(200000, rng)
    P, d, k = il.light_pipe(R.points, R.directions, 4.0, 200.0, n_pipe=1.5)
    assert np.all(np.isfinite(P))  # NA 0.5 into n = 1.5: all TIR
    assert np.all(np.abs(P[:, :2]) <= 2 + 1e-12) and np.allclose(P[:, 2], 200.0)
    assert np.allclose(np.abs(d[:, :2]), np.abs(R.directions[:, :2]))
    assert np.max(k) >= 10
    # the exit face is uniformly lit
    H, _ = np.histogram(P[:, 0], bins=8, range=(-2, 2))
    assert np.std(H)/np.mean(H) < 0.02
    # a hollow pipe in air with TIR condition loses everything that hits a wall
    P2, _, k2 = il.light_pipe(R.points, R.directions, 4.0, 200.0, n_pipe=1.0, n_outside=1.0)
    assert np.all(np.isnan(P2[np.any(k2 > 0, axis=1), 0]))
    P3, _, _ = il.light_pipe(R.points, R.directions, 4.0, 200.0, n_pipe=1.0, mirror=True)
    assert np.all(np.isfinite(P3))


def test_refraction_keeps_direction_sines():
    d = il.lambertian_directions(1000, rng=9)
    d2 = il.refract_plane(d, 1.0, 1.5)
    assert np.allclose(1.5*d2[:, :2], d[:, :2])
    assert np.allclose(il.refract_plane(d2, 1.5, 1.0), d)
    d3 = il.refract_plane(np.array([[0.8, 0, 0.6]]), 1.5, 1.0)
    assert np.all(np.isnan(d3))


@pytest.mark.parametrize("kind", ["2d", "3d"])
def test_cpc_geometry(kind):
    th = np.deg2rad(15)
    c = il.CPC(th, 1.0, kind=kind)
    assert c.entrance_radius == pytest.approx(1/np.sin(th))
    assert c.length == pytest.approx((1 + 1/np.sin(th))/np.tan(th))
    assert c.geometric_concentration == pytest.approx(c.ideal_concentration)
    z, r = c.profile()
    assert np.all(np.diff(z) > 0) and np.all(np.diff(r) <= 1e-12)
    # truncation keeps the exit and shortens the CPC
    t = il.CPC(th, 1.0, kind=kind, truncation=2.0)
    assert t.entrance_radius == pytest.approx(2.0) and t.length < c.length
    assert t.radius(t.length) == pytest.approx(1.0)


def test_cpc_2d_is_ideal_and_fills_exit_etendue():
    th = np.deg2rad(10)
    c = il.CPC(th, 1.0, kind="2d")
    for ang, T in ((0.0, 1), (0.9*th, 1), (1.1*th, 0)):
        P, d = il.tilted_beam(5000, c.entrance_radius, ang, kind="2d", rng=1)
        assert c.trace(P, d).transmitted.mean() == T
    rng = np.random.default_rng(2)
    N = 200000
    x = (2*rng.random(N) - 1)*c.entrance_radius
    P = np.column_stack([x, 0*x, 0*x])
    d = il.directions_from_sines(2*rng.random(N) - 1, 0*x)
    res = c.trace(P, d)
    assert res.transmitted.mean() == pytest.approx(np.sin(th), abs=3e-3)
    # the exit is filled uniformly in p
    H, _ = np.histogram(res.directions[res.transmitted, 0], bins=10, range=(-1, 1))
    assert np.std(H)/np.mean(H) < 0.05


def test_cpc_3d_transmission_and_etendue():
    th = np.deg2rad(10)
    c = il.CPC(th, 1.0, kind="3d")
    P, d = il.tilted_beam(20000, c.entrance_radius, 0.85*th, rng=3)
    assert c.trace(P, d).transmitted.mean() > 0.97
    P, d = il.tilted_beam(20000, c.entrance_radius, 1.15*th, rng=4)
    assert c.trace(P, d).transmitted.mean() < 0.03
    src = il.LambertianSource(c.entrance_radius, na=1.0)
    R = src.rays(200000, rng=5)
    eta = c.trace(R.points, R.directions).transmitted.mean()
    assert eta*src.etendue == pytest.approx(np.pi**2*1.0, rel=0.02)


def test_dielectric_cpc_and_exit_into_air():
    th = np.deg2rad(30)
    c = il.CPC(th, 1.0, n=1.5, kind="3d")
    assert c.geometric_concentration == pytest.approx(1.5**2/np.sin(th)**2)
    P, d = il.tilted_beam(10000, c.entrance_radius, np.deg2rad(15), rng=6)
    assert c.trace(P, d).transmitted.mean() == 1.0
    R = il.LambertianSource(c.entrance_radius, na=np.sin(th)).rays(200000, rng=7)
    eta_air = c.trace(R.points, R.directions, n_out=1.0).transmitted.mean()
    assert eta_air == pytest.approx(1/1.5**2, abs=0.01)
    # mirror walls: no TIR losses near the acceptance angle
    m = il.CPC(th, 1.0, n=1.5, kind="3d", mirror=True)
    P, d = il.tilted_beam(10000, m.entrance_radius, 0.97*th, rng=8)
    assert np.sum(m.trace(P, d).fate == 2) == 0


def test_cpc_numba_matches_numpy():
    th = np.deg2rad(12)
    c = il.CPC(th, 1.0, n=1.4, kind="3d")
    R = il.LambertianSource(c.entrance_radius, na=0.5).rays(3000, rng=9)
    old = il.use_numba
    try:
        il.use_numba = False
        a = c.trace(R.points, R.directions)
        il.use_numba = True
        b = c.trace(R.points, R.directions)
    finally:
        il.use_numba = old
    assert np.array_equal(a.fate, b.fate) and np.array_equal(a.n_reflections, b.n_reflections)
    ok = a.transmitted
    assert np.allclose(a.points[ok], b.points[ok], atol=1e-9)
    res, V = c.trace(R.points[:5], R.directions[:5], paths=True)
    assert V.shape[1] == 5


def _singlet():
    return rt.System([rt.Surface(25.0, 5.0, n_after=1.5, semi_diameter=10.0, conic=-0.5),
                      rt.Surface(-60.0, 40.0, semi_diameter=10.0)], 0, 550e-6,
                     object_distance=80.0, object_height=2.0)


def test_reverse_system_retraces_rays():
    # a ray traced forward and then backwards through the reversed system returns
    # to its start with the opposite direction; reversing twice gives the original
    S = _singlet()
    R = il.reverse_system(S)
    RR = il.reverse_system(R)
    for a, b in zip(S.surfaces, RR.surfaces):
        assert np.isclose(a.thickness, b.thickness) and np.isclose(a.conic, b.conic)
        assert a.R == b.R or np.isclose(a.R, b.R)
    assert np.isclose(RR.object_distance, S.object_distance)
    P = np.array([[0.3, -0.5, -S.object_distance]])
    d = il.directions_from_sines(np.array([0.02]), np.array([0.05]))
    B = S.trace(rt.Rays(P, d, np.zeros(1)))
    Q, e = S.to_image_frame(B.points[-1], B.directions[-1])
    P2 = np.array([[Q[0, 0], Q[0, 1], -R.object_distance]])
    B2 = R.trace(rt.Rays(P2, -e*np.array([1, 1, -1]), np.zeros(1)))
    Q2, e2 = R.to_image_frame(B2.points[-1], B2.directions[-1])
    assert np.allclose(Q2[0, :2], P[0, :2], atol=1e-9)
    assert np.allclose(e2[0, :2], -d[0, :2], atol=1e-9)


def test_backward_irradiance_of_a_uniform_source_behind_a_stop():
    # a uniform source seen through a circular stop of radius r at the distance D:
    # E = pi L sin^2(theta) with tan(theta) = r/D
    r, D = 3.0, 10.0
    S = rt.System([rt.Surface(np.inf, D, semi_diameter=r)], 0, 550e-6, object_distance=50.0,
                  object_height=1.0)
    E = il.backward_irradiance(S, np.zeros((1, 2)), 0.6, lambda x, y: np.ones_like(x),
                               n_directions=200000, rng=3)
    assert np.isclose(E[0], np.pi*r**2/(r**2 + D**2), rtol=0.01)
    # a source that fills only half the plane gives half
    E2 = il.backward_irradiance(S, np.zeros((1, 2)), 0.6, lambda x, y: x > 0,
                                n_directions=200000, rng=4)
    assert np.isclose(E2[0], E[0]/2, rtol=0.02)
