#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for the optical forces of pyoptics.tweezers.

References
----------
A. Ashkin, "Forces of a single-beam gradient laser trap on a dielectric
sphere in the ray optics regime", Biophys. J. 61, 569-582 (1992).

C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light by
Small Particles", Wiley (1983), chapter 5 (Rayleigh limit).
"""

import numpy as np
import pytest

from pyoptics import tweezers as tw

WL, N_W, N_PS = 1.064e-6, 1.33, 1.57


@pytest.mark.parametrize("n_rel", [1.1, 1.2, 1.5, 0.8])
@pytest.mark.parametrize("pol", ["s", "p"])
def test_ashkin_closed_form_equals_traced_series(n_rel, pol):
    for th in np.linspace(0.02, 1.5, 9):
        q = np.array(tw.ashkin_efficiencies(th, n_rel, pol), dtype=float)
        q_tr = np.array(tw.traced_efficiencies(th, n_rel, pol, 300))
        assert np.allclose(q, q_tr, atol=1e-9)


def test_ashkin_normal_incidence():
    # at normal incidence the force is along the ray: 2R for reflection plus the
    # retro-reflected part of the internal series, 4R/(1 + R) in total
    R = ((1.2 - 1)/(1.2 + 1))**2
    q_s, q_g = tw.ashkin_efficiencies(0.0, 1.2, "s")
    assert np.isclose(q_s, 4*R/(1 + R)) and abs(q_g) < 1e-14


def test_radiation_pressure_rayleigh_limit():
    a = 5e-9
    x = 2*np.pi*N_W/WL*a
    m = N_PS/N_W
    q_sca = 8/3*x**4*((m**2 - 1)/(m**2 + 2))**2
    assert np.isclose(tw.radiation_pressure_efficiency(a, N_PS, WL, N_W), q_sca, rtol=1e-3)


def test_polarizability_optical_theorem():
    # with the radiative correction Im(1/alpha) = -k³/(6 pi eps0 n²): extinction = scattering
    a = 50e-9
    alpha = tw.polarizability(a, N_PS, WL, N_W)
    k = 2*np.pi*N_W/WL
    assert np.isclose(np.imag(1/alpha), -k**3/(6*np.pi*tw._EPS0*N_W**2), rtol=1e-10)


def test_beam_power_equals_poynting_flux():
    pw = tw.focus_plane_waves(0.6, N_W, 24, 48, 1.0, "x")
    x = np.linspace(-6e-6, 6e-6, 241)
    X, Y = np.meshgrid(x, x, indexing="ij")
    pts = np.stack([X, Y, np.full_like(X, 0.7e-6)], axis=-1)
    E, H = tw.focus_field(pts, WL, pw, N_W)
    Z0 = 1/(tw._EPS0*tw.C_LIGHT)
    S_z = 0.5*np.real(np.cross(E, np.conj(H)))[..., 2]/Z0
    P = np.sum(S_z)*(x[1] - x[0])**2
    assert np.isclose(P, tw.beam_power(pw, WL, N_W), rtol=2e-3)


def test_focus_field_is_transverse_plane_wave_sum():
    pw = tw.focus_plane_waves(1.2, N_W, 8, 16, None, "circular")
    d, e, a, w = pw
    assert np.allclose(np.sum(d*e, axis=1), 0, atol=1e-14)
    assert np.allclose(np.linalg.norm(e, axis=1), 1)
    # gradient against finite differences
    r = np.array([0.1e-6, -0.2e-6, 0.15e-6])
    E, _, dE = tw.focus_field(r, WL, pw, N_W, gradient=True)
    h = 1e-10
    for l in range(3):
        dr = np.zeros(3)
        dr[l] = h
        fd = (tw.focus_field(r + dr, WL, pw, N_W)[0] - tw.focus_field(r - dr, WL, pw, N_W)[0])/(2*h)
        assert np.allclose(dE[:, l], fd, rtol=1e-5, atol=1e-6*np.abs(E).max()/1e-6)


def test_mie_force_small_bead_equals_dipole_force():
    a = 15e-9
    trap = tw.MieTrap(a, N_PS, WL, 1.2, N_W, max_offset=0.6e-6)
    r = np.array([[0.15e-6, 0, 0.1e-6], [0, 0.2e-6, 0.05e-6], [0, 0, -0.3e-6]])
    q = trap.force(r)
    q_d = tw.dipole_force(r, a, N_PS, WL, trap.plane_waves, N_W)
    assert np.allclose(q, q_d, rtol=5e-3, atol=1e-3*np.abs(q_d).max())


def test_mie_force_wide_beam_equals_radiation_pressure():
    a = 0.5e-6
    trap = tw.MieTrap(a, N_PS, WL, 0.01, N_W, filling=None, max_offset=0.0)
    E0, _ = tw.focus_field(np.zeros(3), WL, trap.plane_waves, N_W)
    I0 = N_W/(2*376.730313668)*np.sum(np.abs(E0)**2)
    P = tw.beam_power(trap.plane_waves, WL, N_W)
    q_pr = trap.force(np.zeros(3))[2]*P/(np.pi*a**2*I0)
    assert np.isclose(q_pr, tw.radiation_pressure_efficiency(a, N_PS, WL, N_W), rtol=2e-3)


def test_mie_force_numba_equals_numpy():
    trap = tw.MieTrap(0.4e-6, N_PS, WL, 1.1, N_W)
    r = np.array([0.2e-6, -0.1e-6, 0.3e-6])
    old = tw.use_numba
    try:
        tw.use_numba = False
        q_np = trap.force(r)
        tw.use_numba = True
        q_nb = trap.force(r)
    finally:
        tw.use_numba = old
    assert np.allclose(q_np, q_nb, atol=1e-9)


def test_trap_symmetry_and_restoring_forces():
    trap = tw.MieTrap(0.5e-6, N_PS, WL, 1.2, N_W)
    q0 = trap.force(np.zeros(3))
    assert np.allclose(q0[:2], 0, atol=1e-8) and q0[2] > 0
    z_eq = tw.equilibrium(lambda z: trap.force(np.array([0, 0, z]))[2], -0.2e-6, 1.0e-6, num=13)
    assert 0 < z_eq < 0.5e-6
    kappa = tw.stiffness(trap.force, [0, 0, z_eq], 10e-9)
    assert np.all(kappa > 0) and kappa[2] < kappa[0]


def test_ray_optics_force():
    # on the axis no lateral force; equilibrium beyond the focus; restoring laterally
    z = np.linspace(-1, 1.5, 26)
    q = tw.ray_optics_force(np.stack([0*z, 0*z, z], -1), 1.0, N_PS, 1.2, N_W)
    assert np.allclose(q[:, :2], 0, atol=1e-10)
    i = np.nonzero((q[:-1, 2] > 0) & (q[1:, 2] <= 0))[0]
    assert len(i) == 1 and 0 < z[i[0]] < 0.5
    qx = tw.ray_optics_force(np.array([[0.3, 0, 0.25], [-0.3, 0, 0.25]]), 1.0, N_PS, 1.2, N_W)[:, 0]
    assert qx[0] < 0 < qx[1] and np.isclose(qx[0], -qx[1])
    # efficiencies do not depend on the sphere size in ray optics
    assert np.allclose(tw.ray_optics_force(np.array([0.3, 0.1, 0.2]), 1.0, N_PS, 1.2, N_W),
                       tw.ray_optics_force(np.array([3.0, 1.0, 2.0]), 10.0, N_PS, 1.2, N_W))


def test_ray_optics_large_bead_close_to_wave_model():
    a = 3e-6
    trap = tw.MieTrap(a, N_PS, WL, 1.2, N_W, max_offset=0.6*a)
    for r in ([0.5, 0, 0.2], [0, 0, 0.5]):
        q_w = trap.force(np.array(r)*a)
        q_r = tw.ray_optics_force(np.array(r), 1.0, N_PS, 1.2, N_W)
        assert np.allclose(q_w, q_r, atol=0.01)


def test_brownian_equipartition_and_psd():
    rng = np.random.default_rng(1)
    gamma, T, kappa = 1e-8, 300.0, 1e-5
    f_c = kappa/(2*np.pi*gamma)
    dt = 1/(50*f_c)
    x = tw.brownian_trajectory(lambda r: -kappa*r, np.zeros((200, 1)), gamma, T, dt, 2000, rng)
    var = np.var(x[500:])
    assert np.isclose(var, tw.K_B*T/kappa, rtol=0.05)
    # integral of the one-sided Lorentzian over all frequencies is the variance
    f = np.linspace(0, 1e4*f_c, 2_000_001)
    assert np.isclose(np.trapezoid(tw.lorentzian_psd(f, f_c, gamma, T), f), tw.K_B*T/kappa, rtol=1e-3)
