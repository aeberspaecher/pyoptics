#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.fibres.

References
----------
A. W. Snyder and J. D. Love, "Optical Waveguide Theory", Chapman and Hall
(1983).

T. Erdogan, "Fiber grating spectra", J. Lightwave Technol. 15, 1277-1294
(1997).
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from pyoptics import fibres, modes, thin_films
from pyoptics.materials import germania_silica_index
from pyoptics.raytrace import fused_silica


def test_germania_silica():
    wl = np.linspace(500, 1700, 13)
    assert np.allclose(germania_silica_index(wl, 0.0), fused_silica(wl), atol=2E-6)
    dn = germania_silica_index(1550, 0.01) - germania_silica_index(1550, 0.0)
    assert 1.3E-3 < dn < 1.7E-3


def test_scalar_modes_step_against_lp_modes():
    f = fibres.step_index(25.0, 1.46, 1.45)
    ms = f.modes(0.85)
    lp = {(m.l, m.m): m.n_eff for m in modes.lp_modes(core_radius=25, n_core=1.46, n_clad=1.45,
                                                      wavelength=0.85)}
    assert len(ms) == len(lp)
    assert max(abs(m.n_eff - lp[(m.l, m.m)]) for m in ms) < 3E-6
    # normalization and sign
    m = ms[0]
    h = m.r[1] - m.r[0]
    assert np.isclose(np.sum(m.u**2*m.r)*h, 1.0)
    assert m.u[0] > 0


def test_parabolic_mode_groups():
    a, n1, wl = 25.0, 1.46, 0.85
    g = fibres.graded_index(a, 2, n1, 1.45)
    D = g.delta(wl)
    for m in g.modes(wl, r_max=60, n_points=4000)[:10]:
        exact = fibres.parabolic_betas(a, n1, D, wl, 2*m.m + m.l - 1)
        assert abs(m.beta - exact) < 1E-8*exact


def test_vector_modes_reduce_to_lp():
    a, n1, n2, wl = 4.1, 1.4495, 1.444, 1.0
    vm = fibres.vector_modes(a, n1, n2, wl)
    lp = modes.lp_modes(core_radius=a, n_core=n1, n_clad=n2, wavelength=wl)
    names = [(m.kind, m.nu) for m in vm]
    assert names[0] == ("HE", 1) and set(names[1:4]) == {("TE", 0), ("TM", 0), ("HE", 2)}
    # TE_0m is exactly LP_1m (the same eigenvalue equation)
    te = [m for m in vm if m.kind == "TE"][0]
    assert abs(te.n_eff - lp[1].n_eff) < 1E-12
    # splitting of order Delta, HE11 close to LP01
    assert abs(vm[0].n_eff - lp[0].n_eff) < 2E-5
    assert max(abs(m.n_eff - lp[1].n_eff) for m in vm[1:4]) < 2E-5
    # stronger guidance: larger splitting
    vm2 = fibres.vector_modes(a, 1.50, 1.444, wl)
    group = [m.n_eff for m in vm2 if (m.kind, m.nu) in (("TE", 0), ("TM", 0), ("HE", 2))
             and m.m == 1]
    assert np.ptp(group) > 10*np.ptp([m.n_eff for m in vm[1:4]])


def test_smf28_like():
    s = fibres.smf28()
    l0 = brentq(lambda w: fibres.dispersion(s, w), 1.25, 1.40)
    assert 1.29 < l0 < 1.33
    D = fibres.dispersion(s, 1.55)
    assert 16 < D < 18.5
    assert 0.07 < fibres.dispersion_slope(s, l0) < 0.1
    # Gloge's split with the cladding material dispersion
    Dm, _ = fibres.material_dispersion(s.n_clad, 1.55)
    assert abs(Dm + fibres.waveguide_dispersion(s, 1.55) - D) < 0.5
    mfd = fibres.petermann_mfd(fibres.scalar_mode(s, 1.55))
    assert 9 < mfd < 11
    assert abs(fibres.beta2_from_D(17, 1550) + 21.7) < 0.05
    assert np.isclose(fibres.D_from_beta2(fibres.beta2_from_D(5, 1300), 1300), 5)


def test_group_index_material_only():
    # a "fibre" with equal indices has no waveguide: the group index of the glass
    n = fibres.ge_doped(0.0)[0]
    _, Ng = fibres.material_dispersion(n, 1.3)
    assert abs(Ng - 1.4626) < 2E-3


def test_ray_delays():
    n1, n2 = 1.47, 1.455
    theta_c = np.arccos(n2/n1)
    assert np.isclose(fibres.step_ray_delay(theta_c, n1) - fibres.step_ray_delay(0, n1),
                      fibres.step_delay_spread(n1, n2))
    assert np.isclose(fibres.acceptance_angle(np.sqrt(n1**2 - n2**2)),
                      np.arcsin(np.sqrt(n1**2 - n2**2)))
    # a skew ray stays inside the core, reflections on the cylinder
    pts = fibres.step_ray(10.0, [5.0, 0, 0], [0, 0.15, 1], 500.0)
    assert np.allclose(np.hypot(pts[1:-1, 0], pts[1:-1, 1]), 10.0)
    assert np.isclose(pts[-1, 2], 500.0)
    # the closest approach of a skew ray is conserved
    d = np.diff(pts, axis=0)
    rmin = np.abs(pts[:-1, 0]*d[:, 1] - pts[:-1, 1]*d[:, 0])/np.hypot(d[:, 0], d[:, 1])
    assert np.allclose(rmin, rmin[0])
    assert list(fibres.ray_class([1.46, 1.44, 1.40], [0, 0.3, 0.1], n1, n2)) == [0, 1, 2]


def test_graded_ray_transit_time():
    # meridional rays in the parabolic profile: delay (n0**2 + b**2)/(2 b) per length
    a, n1, D = 25.0, 1.47, 0.01
    g = fibres.graded_index(a, 2, n1, n1*np.sqrt(1 - 2*D))
    medium = g.index_field(0.85)
    from pyoptics import grin
    theta = np.radians([2.0, 6.0])
    rays = grin.trace(medium, np.zeros((2, 2)), np.c_[np.sin(theta), np.cos(theta)], 2.0,
                      stop=lambda r: r[:, -1] - 5000.0, record=False)
    b = n1*np.cos(theta)
    assert np.allclose(rays.opl/5000.0, (n1**2 + b**2)/(2*b), rtol=1E-4)


def test_wkb_and_optimal_alpha():
    D, N1 = 0.01, 1.47
    a_opt = fibres.optimal_alpha(D)
    assert np.isclose(a_opt, 2 - 12*D/5)
    alphas = np.linspace(1.8, 2.2, 81)
    rms = [fibres.wkb_delay_spread(al, D, N1) for al in alphas]
    assert abs(alphas[np.argmin(rms)] - a_opt) < 0.01
    # parabolic: full spread N1 Delta**2/2 L/c, step about N1 Delta L/c
    assert np.isclose(fibres.wkb_delay_spread(2.0, D, N1, kind="max"),
                      N1*D**2/2*1E9/fibres.C_UM_PER_PS, rtol=1E-3)
    assert np.isclose(fibres.wkb_delay_spread(1E6, D, N1, kind="max"), N1*D*(1 + 1.5*D)*1E9/fibres.C_UM_PER_PS,
                      rtol=1E-3)


def test_graded_mode_delays_against_wkb():
    a, n1, D, wl = 25.0, 1.47, 0.01, 0.85
    g = fibres.graded_index(a, 2, n1, n1*np.sqrt(1 - 2*D))
    k = 2*np.pi/wl
    for l, m in ((0, 1), (3, 2), (5, 3)):
        Ng = fibres.group_index(g, wl, l, m, step=0.002, r_max=2.5*a, n_points=2500)
        neff = fibres.effective_index(g, wl, l, m, r_max=2.5*a, n_points=2500)
        depth = (n1**2 - neff**2)/(2*D*n1**2)
        tau = Ng*1E9/fibres.C_UM_PER_PS
        assert abs(tau - fibres.wkb_group_delay(depth, 2.0, D, n1)) < 2.0   # ps/km


def test_bend_loss_monotonic():
    s = fibres.smf28()
    L = fibres.bend_loss(s, 1.55, [10, 15, 20, 30])
    assert np.all(np.diff(L) < 0)
    assert fibres.bend_loss(s, 1.31, 15) < fibres.bend_loss(s, 1.55, 15)


def test_pmd_mean():
    dgd = fibres.pmd_dgd(300, 0.1, 300, rng=3)
    assert abs(dgd.mean()/(np.sqrt(8*300/(3*np.pi))*0.1) - 1) < 0.08


def test_attenuation_windows():
    a = fibres.attenuation(np.array([0.85, 1.31, 1.55]))
    assert 1.5 < a[0] < 2.5 and 0.28 < a[1] < 0.4 and 0.15 < a[2] < 0.22
    wl = np.linspace(1.2, 1.8, 601)
    assert 1.5 < wl[np.argmin(fibres.attenuation(wl))] < 1.6
    assert fibres.attenuation(1.383, oh_peak=1.0) - fibres.attenuation(1.383) == pytest.approx(1.0)


def test_fbg_uniform_and_transfer_matrix():
    neff, lb, L, dn = 1.447, 1.55, 5000.0, 1E-4
    wl = np.linspace(1.548, 1.552, 401)
    r, t = fibres.fbg_uniform(wl, L, neff, dn, bragg=lb)
    assert np.allclose(np.abs(r)**2 + np.abs(t)**2, 1)
    assert np.isclose(np.max(np.abs(r)**2), np.tanh(np.pi*dn/lb*L)**2, rtol=1E-4)
    r2, _ = fibres.fbg_transfer_matrix(wl, np.linspace(0, L, 51), neff, lb/(2*neff), dn)
    assert np.allclose(r, r2, atol=1E-12)
    # dn_dc shifts the Bragg wavelength
    r3, _ = fibres.fbg_uniform(wl, L, neff, dn, bragg=lb, dn_dc=2E-4)
    assert np.isclose(wl[np.argmax(np.abs(r3))], fibres.bragg_wavelength(lb/(2*neff), neff, 2E-4),
                      atol=2E-5)


def test_fbg_against_thin_films():
    neff, lb, dn = 1.447, 1.55, 5E-4
    per = lb/(2*neff)
    N, sub = 600, 8
    z = (np.arange(N*sub) + 0.5)/sub*per
    n = neff + dn*np.cos(2*np.pi*z/per)
    t = np.full(len(n), per/sub)
    wl = np.linspace(1.546, 1.554, 9)
    rt = np.array([thin_films.r_t_coeffs(0.0, n, 0*n, t, neff, 0, neff, 0, "s", w)[0] for w in wl])
    ru, _ = fibres.fbg_uniform(wl, N*per, neff, dn, period=per)
    assert np.max(np.abs(np.abs(rt)**2 - np.abs(ru)**2)) < 0.02


def test_chirped_fbg_delay():
    neff, L = 1.447, 2E4
    z = np.linspace(0, L, 401)
    zc = (z[1:] + z[:-1])/2
    per = (1.548 + 0.004*zc/L)/(2*neff)
    wl = np.linspace(1.546, 1.554, 2001)
    r, _ = fibres.fbg_transfer_matrix(wl, z, neff, per, 2E-4)
    tau, d = fibres.fbg_delay(wl, r)
    sel = (wl > 1.549) & (wl < 1.551)
    slope = np.polyfit(wl[sel]*1E3, tau[sel], 1)[0]
    expected = 2*neff*L/fibres.C_UM_PER_PS/4.0   # ps over the 4 nm band
    assert abs(slope/expected - 1) < 0.05


def test_phase_shifted_fbg():
    neff, lb, L, dn = 1.447, 1.55, 1E4, 1.5E-4
    z = np.linspace(0, L, 201)
    wl = np.linspace(1.5495, 1.5505, 2001)
    r, t = fibres.fbg_transfer_matrix(wl, z, neff, lb/(2*neff), dn, phase_shifts={100: np.pi})
    i = np.argmin(np.abs(wl - lb))
    assert np.abs(t[i])**2 > 0.99


def test_bragg_shift():
    assert np.isclose(fibres.bragg_shift(1550.0, strain=1E-6)*1E3, 1.209, rtol=1E-3)   # pm/µε
    assert 9 < fibres.bragg_shift(1550.0, dT=1.0)*1E3 < 13   # pm/K
