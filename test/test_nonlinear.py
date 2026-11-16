#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.nonlinear: phase matching against published angles and
periods, the coupled amplitude equations against their analytic solutions,
Boyd–Kleinman, parametric gain and the OPO, the Kerr lens and
self-focusing."""

import numpy as np
import pytest

from pyoptics import abcd
from pyoptics import lasers
from pyoptics import nonlinear as nl
from pyoptics import splitstep as ss


LAM = 1064E-9
D = 14E-12
WLS = [1550E-9, 1/(1/1064E-9 - 1/1550E-9), 1064E-9]
NS = [2.14, 2.08, 2.15]


@pytest.mark.parametrize("crystal, pol, deg", [("BBO", "ooe", 22.8), ("BBO", "eoe", 32.8),
                                               ("KDP", "ooe", 41.2), ("KDP", "eoe", 59.1)])
def test_phase_matching_angles(crystal, pol, deg):
    # 1064 nm SHG (Nikogosyan: BBO 22.8° and 32.9°, KDP 41.2° and 59.2°)
    theta = nl.phase_matching_angle(crystal, 1064.0, pol=pol)
    assert np.degrees(theta) == pytest.approx(deg, abs=0.15)
    assert nl.delta_k(crystal, 1064.0, theta=theta, pol=pol) == pytest.approx(0.0, abs=1E-3)


def test_no_phase_matching_returns_nan():
    # KDP cannot double 400 nm (the harmonic would be at 200 nm, too dispersive)
    assert np.isnan(nl.phase_matching_angle("KDP", 420.0, pol="ooe"))


def test_uniaxial_waves_against_index_ellipsoid():
    n_o, n_e = nl.bbo_indices(532.0)
    th = np.linspace(0, np.pi/2, 7)
    n_ord, n_ext, rho = nl.uniaxial_waves(n_o, n_e, th)
    assert np.allclose(n_ord, n_o)
    assert np.allclose(n_ext, nl.extraordinary_index(n_o, n_e, th))
    # walk-off tan(rho) = n_e(th)**2/2 (1/n_e**2 - 1/n_o**2) sin(2 th)
    ne_th = nl.extraordinary_index(n_o, n_e, th)
    rho_ref = np.arctan(ne_th**2/2*np.abs(1/n_e**2 - 1/n_o**2)*np.sin(2*th))
    assert np.allclose(rho, rho_ref, atol=1E-9)
    # BBO type I at 1064 nm: 3.2° walk-off of the harmonic
    theta = nl.phase_matching_angle("BBO", 1064.0)
    assert np.degrees(nl.uniaxial_waves(*nl.bbo_indices(532.0), theta)[2]) == pytest.approx(3.19, abs=0.05)


def test_d_eff():
    assert nl.d_eff("BBO", np.radians(22.8)) == pytest.approx(2.2*np.cos(np.radians(22.8)) + 0.04*np.sin(np.radians(22.8)))
    assert nl.d_eff("KDP", np.radians(41.2), np.pi/4) == pytest.approx(0.39*np.sin(np.radians(41.2)))
    assert nl.qpm_d_eff(25.0) == pytest.approx(2/np.pi*25.0)
    assert nl.qpm_d_eff(25.0, order=2) == pytest.approx(0.0, abs=1E-12)
    assert nl.qpm_d_eff(25.0, order=3) == pytest.approx(2/(3*np.pi)*25.0)


def test_mgo_ln_and_qpm_periods():
    assert nl.mgo_ln_ne(1064.0) == pytest.approx(2.1482, abs=1E-4)
    # first order periods of MgO:PPLN (6.9 µm for 1064 nm SHG, 19.5 µm for 1550 nm SHG,
    # about 30.5 µm for a 1064 nm pumped OPO with a 1550 nm signal)
    assert nl.qpm_period("MgO:LN", 1064.0, T=25)*1E6 == pytest.approx(6.97, abs=0.1)
    assert nl.qpm_period("MgO:LN", 1550.0, T=25)*1E6 == pytest.approx(19.4, abs=0.2)
    wl_i = 1/(1/1064.0 - 1/1550.0)
    P = nl.qpm_period("MgO:LN", 1550.0, wl_i, T=100)
    assert P*1E6 == pytest.approx(30.3, abs=0.4)
    assert nl.delta_k("MgO:LN", 1550.0, wl_i, T=100, period=P) == pytest.approx(0.0, abs=1E-6)
    assert nl.qpm_signal("MgO:LN", 1064.0, P, T=100) == pytest.approx(1550.0, abs=1E-3)


def test_acceptance_bandwidth_linear():
    # linear dk(x) = s (x - x0): FWHM 5.566/(L s)
    s, L = 300.0, 0.02
    fwhm, lo, hi = nl.acceptance_bandwidth(lambda x: s*(x - 1.0), 1.0, L)
    assert fwhm == pytest.approx(5.5658/(L*s), rel=1E-4)
    assert (lo + hi)/2 == pytest.approx(1.0)


def test_shg_depleted_and_undepleted():
    n1 = n2 = 2.15
    L = 0.01
    z = np.linspace(0, L, 50)
    a1, a2 = nl.shg(z, 1E12, LAM, n1, n2, D)
    assert np.abs(a2[-1])**2/1E12 == pytest.approx(nl.shg_depleted(1E12, L, D, LAM, n1, n2), rel=1E-7)
    assert np.allclose(np.abs(a1)**2 + np.abs(a2)**2, 1E12, rtol=1E-8)
    for dk in (0.0, 300.0, 1500.0):
        a1, a2 = nl.shg(z, 1E7, LAM, n1, n2, D, dk=dk)
        eta = nl.shg_efficiency(1E7, L, D, LAM, n1, n2, dk)
        assert np.abs(a2[-1])**2/1E7 == pytest.approx(eta, rel=1E-3)


def test_qpm_square_wave_poling():
    # a domain pattern quasi-phase matches with d_eff = 2/pi d (field) away from the ends
    n1, n2 = 2.15, 2.23
    dk = 2*np.pi/7E-6
    L = 60*7E-6
    poled = lambda z: D*np.sign(np.cos(2*np.pi*z/7E-6))
    z = np.linspace(0, L, 301)
    a1, a2 = nl.shg(z, 1E6, LAM, n1, n2, poled, dk=dk, max_step=7E-6/40)
    eta_qpm = nl.shg_efficiency(1E6, L, 2/np.pi*D, LAM, n1, n2)
    assert np.abs(a2[-1])**2/1E6 == pytest.approx(eta_qpm, rel=0.02)


def test_three_wave_manley_rowe_and_sfg():
    z = np.linspace(0, 0.01, 40)
    a = nl.three_wave(z, [3E5, 1E5, 0], WLS, NS, D)
    om = 2*np.pi*nl.C0/np.array(WLS)
    flux = np.abs(a)**2/om
    assert np.allclose(flux[:, 0] + flux[:, 2], flux[0, 0], rtol=1E-7)
    assert np.allclose(flux[:, 1] + flux[:, 2], flux[0, 1], rtol=1E-7)
    assert np.allclose(np.sum(np.abs(a)**2, axis=1), np.sum(np.abs(a[0])**2), rtol=1E-8)
    # weak conversion: a3 = i omega_3 K a1 a2 L
    a = nl.three_wave(z, [1E3, 1E3, 0], WLS, NS, D)
    K = nl.coupling_K(D, *NS)
    assert np.abs(a[-1, 2]) == pytest.approx(om[2]*K*1E6*0.01, rel=2E-3)


def test_opa_gain_and_threshold():
    L = 0.01
    z = np.linspace(0, L, 20)
    for dk in (0.0, 200.0, 2000.0):
        Ip = 1E13
        Gam = nl.parametric_gain_coefficient(Ip, D, WLS, NS)
        a = nl.three_wave(z, [1.0, 0, np.sqrt(Ip)], WLS, NS, D, dk=dk)
        assert np.abs(a[-1, 0])**2 == pytest.approx(nl.opa_gain(Gam, L, dk), rel=1E-4)
    # threshold: the round trip gain cosh² equals 1/R
    I_th = nl.opo_threshold(L, D, WLS, NS, np.sqrt(0.95))
    Gam = nl.parametric_gain_coefficient(I_th, D, WLS, NS)
    assert np.cosh(Gam*L)**2*0.95 == pytest.approx(1.0)
    # small losses: Gamma² L² = 2 (1 - r_s), DRO (1 - r_s)(1 - r_i)
    r = 0.999
    G2 = lambda I: (nl.parametric_gain_coefficient(I, D, WLS, NS)*L)**2
    assert G2(nl.opo_threshold(L, D, WLS, NS, r)) == pytest.approx(2*(1 - r), rel=2E-3)
    assert G2(nl.opo_threshold(L, D, WLS, NS, r, r)) == pytest.approx((1 - r)**2, rel=2E-3)


def test_sro_pump_depletion_against_round_trips():
    assert nl.sro_pump_depletion(np.pi**2/4) == pytest.approx(1.0)
    assert nl.sro_pump_depletion(0.5) == 0.0
    L, r = 0.05, np.sqrt(0.99)
    I_th = nl.opo_threshold(L, D, WLS, NS, r)
    z = np.array([0.0, L])
    for N in (2.0, 3.5):
        Ip = N*I_th
        a_s = 0.3*np.sqrt(Ip) + 0j
        for _ in range(1500):
            a = nl.three_wave(z, [a_s, 0, np.sqrt(Ip)], WLS, NS, D, rtol=1E-7, atol=1E-9)
            a_s = a[-1, 0]*r
        assert 1 - np.abs(a[-1, 2])**2/Ip == pytest.approx(nl.sro_pump_depletion(N), abs=0.01)


def test_boyd_kleinman():
    h, s = nl.boyd_kleinman_optimum(2.84)
    assert h == pytest.approx(1.068, abs=2E-3)
    assert abs(s) == pytest.approx(0.57, abs=0.02)
    # near field: h = xi; walk-off lowers h
    assert nl.boyd_kleinman_h(0.01, 0.0) == pytest.approx(0.01, rel=1E-3)
    assert nl.boyd_kleinman_optimum(2.84, B=2.0)[0] < 0.5*h
    assert nl.boyd_kleinman_h(1.0, 0.3, B=0.0) == pytest.approx(nl.boyd_kleinman_h(1.0, 0.3, B=1E-9), rel=1E-6)
    # near field power equals the plane wave efficiency integrated over a Gaussian
    n1, n2, w0, P, L = 2.15, 2.23, 300E-6, 1.0, 1E-3
    b = 2*np.pi*n1*w0**2/LAM
    P2 = nl.shg_power_focused(P, L, D, LAM, n1, n2, nl.boyd_kleinman_h(L/b, 0.0))
    kap = nl.shg_kappa(D, LAM, n1, n2)
    assert P2 == pytest.approx(kap**2*L**2*P**2/(np.pi*w0**2), rel=2E-3)


def test_shg_pulse():
    t = np.linspace(-10, 10, 1024, endpoint=False)    # ps
    I0 = 1E12
    n1, n2, L = 2.15, 2.23, 0.01
    kap = nl.shg_kappa(D, LAM, n1, n2)
    A1 = np.sqrt(I0)*np.exp(-t**2/(2*2.0**2))
    a1, a2 = nl.shg_pulse(t, A1, L, kap, steps=200)
    # without walk-off every time slice converts like a plane wave
    assert np.allclose(np.abs(a2)**2, np.abs(A1)**2*np.tanh(kap*np.abs(A1)*L)**2, atol=1E-6*I0)
    # strong walk-off: the harmonic stretches to the walk-off time and its energy drops
    b1, b2 = nl.shg_pulse(t, A1*1E-3, L, kap, gvm=500.0, steps=200)
    c1, c2 = nl.shg_pulse(t, A1*1E-3, L, kap, steps=200)
    E = lambda x: np.sum(np.abs(x)**2)
    assert E(b2) < 0.75*E(c2)
    from pyoptics.ultrafast import fwhm
    assert fwhm(t, np.abs(b2)**2) > 3.5


def test_kerr_formulas():
    assert nl.critical_power(800E-9, 1.45, 2.7E-20)/1E6 == pytest.approx(2.45, abs=0.02)
    assert nl.kerr_phase(1E16, 2.7E-20, 1E-3, 800E-9) == pytest.approx(2*np.pi/800E-9*2.7E-20*1E16*1E-3)
    # the Kerr lens is the curvature of the nonlinear phase at the axis
    w, P, n2, L = 30E-6, 1E5, 3E-20, 2E-3
    r = np.linspace(0, 0.2E-6, 5)
    phi = 2*np.pi/800E-9*n2*L*2*P/(np.pi*w**2)*np.exp(-2*r**2/w**2)
    curv = np.polyfit(r**2, phi, 1)[0]                # phi = phi0 - k0 r²/(2 f)
    f = -2*np.pi/800E-9/(2*curv)
    assert f == pytest.approx(nl.kerr_lens_focal(P, w, n2, L), rel=1E-3)


def test_self_focusing_against_beam_propagation():
    lam, n0, n2, w0 = 800E-9, 1.45, 2.7E-20, 100E-6
    P_cr = nl.critical_power(lam, n0, n2)
    x = np.linspace(-0.5E-3, 0.5E-3, 256, endpoint=False)
    X, Y = np.meshgrid(x, x)
    P = 4*P_cr
    I0 = 2*P/(np.pi*w0**2)
    E, z, peak = ss.beam_propagate((x, x), np.sqrt(I0)*np.exp(-(X**2 + Y**2)/w0**2), 0.2, lam, n0, n2,
                                   max_phase=0.1, stop_intensity=50*I0)
    assert z[-1] == pytest.approx(nl.self_focusing_distance(P, P_cr, w0, lam, n0), rel=0.1)
    assert np.isnan(nl.self_focusing_distance(0.5*P_cr, P_cr, w0, lam, n0))


def _klm_cavity(delta=3E-3, frac=0.55, f=0.05, d1=0.6, d2=0.4):
    F, Lf = abcd.free_space, abcd.thin_lens
    zs = 2*f + delta
    z1, z2 = zs*frac, zs*(1 - frac)

    def cavity(fk):
        K = Lf(fk) if np.isfinite(fk) else np.eye(2)
        arm2 = abcd.compose(F(z2), Lf(f), F(d2), F(d2), Lf(f), F(z2))
        arm1 = abcd.compose(F(z1), Lf(f), F(d1), F(d1), Lf(f), F(z1))
        return abcd.compose(arm2, K, arm1, K), abcd.compose(F(z2), Lf(f), F(d2))
    return cavity


def test_kerr_lens_mode_and_loss():
    cav = _klm_cavity()
    lam = 800E-9
    w_k, w_a = nl.kerr_lens_mode(cav, 0.0, 3E-20, 3E-3, lam)
    q = abcd.eigenmode(cav(np.inf)[0])
    assert w_k == pytest.approx(abcd.beam_radius(q, lam))
    w_k1, w_a1 = nl.kerr_lens_mode(cav, 1E6, 3E-20, 3E-3, lam)
    assert w_a1 < 0.9*w_a
    # self-consistency: the mode with the Kerr lens of its own spot size
    q1 = abcd.eigenmode(cav(nl.kerr_lens_focal(1E6, w_k1, 3E-20, 3E-3))[0])
    assert abcd.beam_radius(q1, lam) == pytest.approx(w_k1, rel=1E-8)
    loss, table = nl.kerr_lens_loss(cav, 0.6E-3, 3E-20, 3E-3, lam, 1E6, n_powers=40)
    assert loss(0.0) > loss(5E5) > loss(1E6) > 0
    assert loss(0.0) == pytest.approx(-0.5*np.log(1 - np.exp(-2*0.6E-3**2/w_a**2)))


def test_kerr_lens_mode_locking_with_master_equation():
    # the Kerr lens loss as the fast saturable loss of the lasers master equation
    cav = _klm_cavity()
    loss, _ = nl.kerr_lens_loss(cav, 0.6E-3, 3E-20, 3E-3, 800E-9, 1E6, n_powers=60)
    t = np.linspace(-2000, 2000, 2**12, endpoint=False)        # fs, power in MW
    A0 = np.sqrt(0.05)/np.cosh(t/150)
    q_of = lambda P: loss(P*1E6) - loss(0.0)                    # only the saturable part
    A, hist = lasers.master_equation(t, A0.astype(complex), 3000, 0.1, 6.0, 0.05, 0.03 + loss(0.0),
                                     saturable_loss=q_of, record_every=100)
    A_off, _ = lasers.master_equation(t, A0.astype(complex), 3000, 0.1, 6.0, 0.05, 0.03 + loss(0.0),
                                      record_every=100)
    from pyoptics.ultrafast import fwhm
    assert fwhm(t, np.abs(A)**2) < 0.6*fwhm(t, np.abs(A_off)**2)
