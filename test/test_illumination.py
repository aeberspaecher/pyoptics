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
