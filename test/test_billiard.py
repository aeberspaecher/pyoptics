#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.billiard (bounce map, open dielectric billiard and
boundary Husimi functions) and of the Husimi function of an FDFD mode."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import jv, jvp

from pyoptics import billiard, fdfd, fdtd
from pyoptics import analytical as an


def _jacobian(B, s0, p0, e=1e-6):
    def F(s, p):
        return np.array(B.bounce(s, p), float).ravel()
    J = np.column_stack([(F(s0 + e, p0) - F(s0 - e, p0))/(2*e), (F(s0, p0 + e) - F(s0, p0 - e))/(2*e)])
    return np.linalg.det(J)


def test_circle_conserves_p_and_steps_by_the_chord():
    B = billiard.Billiard(fdtd.Limacon((0.3, -0.2), 1.5, 0.0))
    assert abs(B.length - 3*np.pi) < 1e-12
    p0 = np.array([0.5, -0.9, 0.99])
    S, P = B.orbit(np.array([0.3, 1.0, 2.0]), p0, 40)
    assert np.allclose(P, p0, atol=1e-12)
    # successive bounces are 2 arccos(p) apart in polar angle (mod 2 pi)
    step = np.mod(S[1] - S[0], B.length)/1.5
    assert np.allclose(step, np.mod(2*np.arccos(p0), 2*np.pi), atol=1e-10)


@pytest.mark.parametrize("shape", [fdtd.Limacon((0, 0), 1.0, 0.43), fdtd.Stadium((0, 0), 1.0, 1.0)])
def test_bounce_map_is_area_preserving_and_reversible(shape):
    B = billiard.Billiard(shape)
    for s0, p0 in ((1.234, 0.3), (3.0, -0.7), (0.5, 0.95)):
        assert abs(_jacobian(B, s0, p0) - 1) < 1e-6
        s1, p1 = B.bounce(s0, p0)
        s2, p2 = B.bounce(s1, -p1)          # time reversal: back to (s0, -p0)
        assert abs(s2 - s0) < 1e-10 and abs(p2 + p0) < 1e-10


def test_perimeters_and_arc_length():
    st = billiard.Billiard(fdtd.Stadium((0, 0), 1.0, 0.7))
    assert abs(st.length - (2*np.pi + 4*0.7)) < 1e-6
    lim = fdtd.Limacon((0, 0), 1.0, 0.43)
    B = billiard.Billiard(lim)
    speed = lambda f: np.hypot(lim.radius(f), lim.dradius(f))
    assert abs(B.length - quad(speed, 0, 2*np.pi, epsabs=1e-13)[0]) < 1e-11
    for phi in (0.3, 2.0, 5.5):
        assert abs(B.arc_length(phi) - quad(speed, 0, phi, epsabs=1e-13)[0]) < 1e-9
        assert abs(B.polar_angle(B.arc_length(phi)) - phi) < 1e-12
        t, n = B.tangent(phi), B.normal(phi)
        assert abs(np.dot(t, n)) < 1e-14 and np.dot(n, B.point(phi)) > 0


def test_stadium_shape():
    st = fdtd.Stadium((0, 0), 1.0, 0.8)
    phi = np.linspace(0, 2*np.pi, 721)
    b = st.boundary(721)
    on_line = np.abs(np.abs(b[:, 1]) - 1.0) < 1e-12
    on_cap = np.abs(np.hypot(np.abs(b[:, 0]) - 0.8, b[:, 1]) - 1.0) < 1e-12
    assert np.all(on_line | on_cap)
    h = 1e-6
    assert np.allclose(st.dradius(phi[1:-1]), (st.radius(phi[1:-1] + h) - st.radius(phi[1:-1] - h))/(2*h), atol=1e-6)


def test_open_billiard_keeps_whispering_gallery_rays():
    n = 3.3
    B = billiard.Billiard(fdtd.Limacon((0, 0), 1.0, 0.0))
    # circle: rays above the critical line never leave, below they lose (1 - R) at every bounce
    S, P = B.orbit(np.zeros(2), np.array([0.5, 0.2]), 20)
    s_e, p_e, D, decay = billiard.steady_distribution(S[:, :1], P[:, :1], n, B.length, skip=0)
    assert np.allclose(decay, 1)
    R = billiard.fresnel_reflectance(0.2, n)
    _, _, _, decay = billiard.steady_distribution(S[:, 1:], P[:, 1:], n, B.length, skip=0)
    assert np.allclose(decay, R**np.arange(21))
    assert abs(billiard.fresnel_reflectance(0.0, n) - ((n - 1)/(n + 1))**2) < 1e-14


def test_limacon_rays_escape_along_the_axis():
    n = 3.3
    B = billiard.Billiard(fdtd.Limacon((0, 0), 1.0, 0.43))
    rng = np.random.default_rng(1)
    S, P = B.orbit(rng.uniform(0, B.length, 4000), rng.uniform(-1, 1, 4000), 40)
    phi, I = billiard.escape_directions(B, S, P, n, bins=72, skip=15)
    assert abs(np.degrees(phi[np.argmax(I)])) < 15        # the main beam at phi = 0
    assert abs(I.mean() - 1) < 1e-12
    # the escape angle obeys Snell: a ray at normal incidence leaves along the normal
    assert abs(B.escape_angle(0.0, 0.0, n)) < 1e-12


def test_boundary_husimi_of_an_analytic_whispering_gallery_wave():
    m, n, k, R = 25, 3.0, 10.0, 1.0
    B = billiard.Billiard(fdtd.Limacon((0, 0), R, 0.0))
    s, P, _, _ = B.boundary(1024)
    phi = s/R
    psi = jv(m, n*k*R)*np.exp(1j*m*phi)              # travels counter-clockwise
    dpsi = n*k*jvp(m, n*k*R)*np.exp(1j*m*phi)
    q, p, H_out, H_in = billiard.boundary_husimi(s, psi, dpsi, B.length, n, k)
    prof = H_out.sum(1)
    assert abs(p[np.argmax(prof)] - billiard.circle_wgm_p(m, n, k, R)) < 0.02
    assert prof[p < 0].sum() < 1e-6*prof.sum()           # nothing clockwise
    assert abs(H_out.sum()/H_in.sum() - 1) < 1e-2        # a standing wave across the rim
    assert np.ptp(H_out.max(0))/H_out.max() < 1e-6       # uniform along the circle


def test_husimi_birkhoff_of_a_disk_mode():
    m, R, n = 6, 1.0, 3.3
    k = an.disk_resonance(m, R, n, 2.4)
    sim = fdfd.Simulation(fdtd.Grid((4.0, 4.0), 0.02), [(fdtd.Circle((0, 0), R), fdtd.Medium.from_index(n))])
    r = sim.resonances(2*np.pi/k.real, n=1)[0]
    q, p, H_inc, H_ref = fdfd.husimi_birkhoff(r.field, fdtd.Limacon((0, 0), R, 0.0), n)
    prof = H_inc.sum(1)
    assert abs(abs(p[np.argmax(prof)]) - m/(n*k.real*R)) < 0.03
    assert abs(H_inc.sum()/H_ref.sum() - 1) < 0.01
    with pytest.raises(ValueError):
        sim_h = fdfd.Simulation(fdtd.Grid((2.0, 2.0), 0.1), mode="Hz")
        fdfd.husimi_birkhoff(sim_h.solve(1.0, [fdfd.PointSource((0, 0))]), fdtd.Circle((0, 0), 0.5), n)
