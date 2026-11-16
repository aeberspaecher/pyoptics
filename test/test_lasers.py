#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.lasers: line shapes, gain saturation, rate equations,
Q switching, resonators, mode locking and the semiconductor laser."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp, quad

from pyoptics import lasers as L
from pyoptics import abcd


def test_line_shapes_normalized():
    for f in (L.lorentzian, L.gaussian_line):
        val, _ = quad(lambda nu: f(nu, 0.0, 1.0), -200, 200, limit=400)
        assert val == pytest.approx(1.0, abs=4E-3)
    # FWHM
    assert L.lorentzian(0.5, 0, 1)/L.lorentzian(0, 0, 1) == pytest.approx(0.5)
    assert L.gaussian_line(0.5, 0, 1)/L.gaussian_line(0, 0, 1) == pytest.approx(0.5)


def test_einstein_and_cross_section():
    wl, tau, dnu = 1064E-9, 230E-6, 120E9
    e = L.einstein_coefficients(1/tau, wl, g1=2, g2=4)
    assert e["B12"] == pytest.approx(2*e["B21"])
    assert e["B21"] == pytest.approx(wl**3/(8*np.pi*L.H*tau))
    nu0 = L.C0/wl
    s = L.cross_section(nu0, nu0, wl, tau, dnu)
    assert s == pytest.approx(L.peak_cross_section(wl, tau, dnu))
    # Nd:YAG: about 9 % of the decays of the upper level go into the
    # R2 -> Y3 Stark line at 1064 nm, which gives the effective 2.8E-23 m²
    s_yag = L.peak_cross_section(wl, 230E-6/0.09, dnu, n=1.82)
    assert s_yag == pytest.approx(L.GAIN_MEDIA["Nd:YAG"]["sigma"], rel=0.1)


def test_saturation():
    assert L.saturated_gain(1.0, 1.0, 1.0) == pytest.approx(0.5)
    assert L.saturated_gain(1.0, 3.0, 1.0, "inhomogeneous") == pytest.approx(0.5)
    Isat = L.saturation_intensity(1064E-9, 2.8E-23, 230E-6)
    assert Isat == pytest.approx(L.photon_energy(1064E-9)/(2.8E-23*230E-6))
    assert L.saturation_intensity(694E-9, 2.5E-24, 3E-3, levels=3) == pytest.approx(
        L.photon_energy(694E-9)/(2*2.5E-24*3E-3))


@pytest.mark.parametrize("x", [1E-4, 0.1, 1.0, 10.0])
def test_amplifier_gain_against_ode(x):
    G0 = 100.0
    g0 = np.log(G0)
    sol = solve_ivp(lambda z, I: g0*I/(1 + I), (0, 1), [x], rtol=1E-10, atol=1E-14)
    assert L.amplifier_gain(G0, x, 1.0) == pytest.approx(sol.y[0, -1]/x, rel=1E-6)


def test_frantz_nodvik_limits():
    G0 = 50.0
    assert L.frantz_nodvik(1E-6, 1.0, G0) == pytest.approx(G0*1E-6, rel=1E-4)
    E = 20.0
    assert L.frantz_nodvik(E, 1.0, G0) == pytest.approx(E + np.log(G0), rel=1E-6)


def test_level_schemes():
    assert L.three_level_inversion(1.0, 1.0) == pytest.approx(0.0)
    assert L.three_level_inversion(0.5, 1.0) < 0
    assert L.four_level_inversion(0.01, 1.0) > 0
    assert L.four_level_inversion(1E6, 1.0) == pytest.approx(1.0, rel=1E-5)


def test_rate_equations_steady_state_and_relaxation():
    tau, tau_c, r = 230E-6, 10E-9, 3.0
    t = np.linspace(0, 1.5E-3, 30001)
    n, p = L.rate_equations(t, r, tau, tau_c, beta=1E-9)
    assert n[-1] == pytest.approx(1.0, rel=1E-3)
    assert p[-1] == pytest.approx(r - 1, rel=1E-3)
    # ringing frequency from the zero crossings of p - (r - 1) late in the transient
    f_R, gamma = L.relaxation_oscillation(r, tau, tau_c)
    sel = (t > 0.5E-3) & (t < 1.0E-3)
    d = p[sel] - (r - 1)
    zc = np.nonzero(np.diff(np.sign(d)) != 0)[0]
    f_meas = (len(zc) - 1)/2/(t[sel][zc[-1]] - t[sel][zc[0]])
    assert f_meas == pytest.approx(f_R, rel=0.02)
    # decay of the envelope
    pk = np.abs(d)
    assert gamma == pytest.approx(r/(2*tau))


@pytest.mark.parametrize("n_i", [1.5, 3.0, 6.0])
def test_q_switch_against_rate_equations(n_i):
    # a long upper level lifetime: the theory neglects the decay during the pulse
    tau, tau_c = 0.1, 5E-9
    t = np.linspace(0, 3000*tau_c, 300001)
    n, p = L.rate_equations(t, 0.0, tau, tau_c, beta=1E-12, n0=n_i, p0=1E-10, max_step=tau_c/5)
    qs = L.q_switch(n_i, tau_c)
    # inversion at the end of the pulse (later it decays spontaneously)
    k = np.argmax(p)
    end = k + np.nonzero(p[k:] < 1E-6*p[k])[0][0]
    assert n[end] == pytest.approx(qs["n_f"], rel=2E-3)
    # p is in units of N_th tau_c/tau: peak photon number in N_th
    assert p.max()*tau_c/tau == pytest.approx(qs["p_peak"], rel=2E-3)
    energy = np.trapezoid(p, t)*tau_c/tau/tau_c   # photons out / N_th
    assert energy == pytest.approx(n_i - qs["n_f"], rel=2E-3)


def test_passive_q_switch_runs():
    tau, tau_c = 230E-6, 5E-9
    t = np.linspace(0, 2E-3, 200001)
    n, p, a = L.rate_equations(t, 5.0, tau, tau_c, beta=1E-9, absorber=(2.0, 10E-6, 50.0),
                               max_step=2E-9)
    # repetitive pulses: several peaks far above the cw level
    peaks = np.nonzero((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] > 100))[0]
    assert len(peaks) >= 2
    # the inversion builds up to the threshold of the unbleached cavity (1 + q_a)
    assert n.max() == pytest.approx(3.0, rel=0.05)
    assert a.min() < 0.1


def test_output_coupling():
    g0l2, di, Ps = 0.4, 0.02, 1.0
    T = np.linspace(0.001, 0.3, 30001)
    P = L.output_power(T, g0l2, di, Ps)
    T_opt, P_max = L.optimum_output_coupling(g0l2, di, Ps)
    assert T[np.argmax(P)] == pytest.approx(T_opt, abs=1E-4)
    assert P.max() == pytest.approx(P_max, rel=1E-6)
    assert L.output_power(0.5, g0l2, di, Ps) == 0.0


@pytest.mark.parametrize("R2", [0.95, 0.6, 0.1])
def test_rigrod_against_numerical(R2):
    g0l, R1 = 1.5, 1.0
    # shoot: I+(0) = R1 I-(0); integrate; demand I-(l) = R2 I+(l)
    def mismatch(Im0, full=False):
        def f(z, y):
            g = g0l/(1 + y[0] + y[1])
            return [g*y[0], -g*y[1]]
        sol = solve_ivp(f, (0, 1), [R1*Im0, Im0], rtol=1E-11, atol=1E-14)
        Ip, Im = sol.y[:, -1]
        return (Ip, Im) if full else Im - R2*Ip
    from scipy.optimize import brentq
    Im0 = brentq(mismatch, 1E-6, 50)
    Ip, _ = mismatch(Im0, True)
    assert L.rigrod_output(g0l, 1.0, R1, R2, 1.0) == pytest.approx((1 - R2)*Ip, rel=1E-6)


def test_slope_and_threshold():
    assert L.slope_efficiency(0.05, 0.05, 808E-9, 1064E-9) == pytest.approx(0.5*808/1064)
    P = L.threshold_pump_power(0.05, 2.8E-23, 230E-6, 808E-9, np.pi*(100E-6)**2)
    assert 0.01 < P < 1.0


def test_modes():
    assert L.free_spectral_range(0.15) == pytest.approx(L.C0/0.3)
    nu, q = L.mode_frequencies(0.15, 1E9, 5E9)
    assert np.allclose(np.diff(nu), L.C0/0.3)
    # confocal cavity: Gouy phase pi per round trip, transverse modes at FSR/2 steps
    Lc, R = 0.1, 0.1
    M = abcd.compose(abcd.free_space(Lc), abcd.mirror(R), abcd.free_space(Lc), abcd.mirror(R))
    nu = L.transverse_mode_frequencies(M, 2*Lc, [1000], max_order=2)[0]
    assert np.allclose(np.diff(nu), L.free_spectral_range(Lc)/2)
    # plane-parallel limit: no splitting
    M = abcd.compose(abcd.free_space(Lc), abcd.mirror(1E6), abcd.free_space(Lc), abcd.mirror(1E6))
    nu = L.transverse_mode_frequencies(M, 2*Lc, [1000], max_order=2)[0]
    assert np.ptp(np.diff(nu)) < 1E-6*L.free_spectral_range(Lc)
    assert L.cavity_linewidth(1/(2*np.pi)) == pytest.approx(1.0)
    st = L.schawlow_townes(4.7E14, 1E6, 1E-3)
    assert st == pytest.approx(np.pi*L.H*4.7E14*1E12/1E-3)


def test_mode_locked_field():
    N, fsr = 11, 1E9
    t = np.linspace(0, 2/fsr, 4001)
    A = L.mode_locked_field(t, N, fsr)
    P = np.abs(A)**2
    assert P.max() == pytest.approx(N**2)
    assert np.mean(P) == pytest.approx(N, rel=1E-2)
    Ar = L.mode_locked_field(t, N, fsr, phases="random", rng=1)
    assert np.mean(np.abs(Ar)**2) == pytest.approx(N, rel=1E-2)
    assert np.abs(Ar).max()**2 < N**2


def test_active_mode_locking_against_master_equation():
    # 100 MHz modulation, 100 GHz gain bandwidth (units: ns and GHz)
    f_m, dnu = 0.1, 100.0
    t = np.linspace(-0.6, 0.6, 2**12)
    A0 = np.exp(-t**2/(2*0.05**2)).astype(complex)
    g0, l, M = 0.5, 0.1, 0.2
    A, hist = L.master_equation(t, A0, 4000, g0, 1.0, dnu, l, modulation=(M, f_m),
                                record_every=100)
    P = np.abs(A)**2
    above = t[P >= P.max()/2]
    fwhm = above[-1] - above[0]
    g = hist["gain"][-1]
    assert g == pytest.approx(l, rel=0.02)
    assert fwhm == pytest.approx(L.active_ml_width(g, M, f_m, dnu), rel=0.05)


def test_saturable_loss_hook():
    t = np.linspace(-5, 5, 1024)
    A0 = 0.1*np.exp(-t**2).astype(complex)
    calls = []

    def sam(P):
        calls.append(1)
        return 0.05/(1 + P/0.5)

    A1, _ = L.master_equation(t, A0, 3, 0.2, 10.0, 2.0, 0.1, saturable_loss=sam)
    A2, _ = L.master_equation(t, A0, 3, 0.2, 10.0, 2.0, 0.1, saturable_loss=(0.05, 0.5))
    assert len(calls) == 3
    assert np.allclose(A1, A2)


def test_time_bandwidth():
    assert L.time_bandwidth_product("gaussian") == pytest.approx(0.4413, rel=1E-3)
    assert L.time_bandwidth_product("sech2") == pytest.approx(0.315, rel=1E-2)


# --- diode laser ---

@pytest.fixture(scope="module")
def dfb():
    return L.DiodeLaser()


def test_diode_threshold(dfb):
    assert dfb.tau_p == pytest.approx(1/(7.5E7*8500))
    assert dfb.i_th == pytest.approx(15.4E-3, rel=0.01)
    for I in (5E-3, 30E-3):
        N, S = dfb.steady_state(I)
        dN, dS = dfb.rhs(N, S, I)
        assert abs(dN) < 1E-6*I/L.Q_E
        assert abs(dS) < 1E-6*S/dfb.tau_p
    P, n = dfb.li_curve([10E-3, 40E-3, 41E-3])
    assert P[0] < 1E-4
    slope = (P[2] - P[1])/1E-3
    assert slope == pytest.approx(dfb.slope_efficiency(), rel=0.03)
    assert n[1] == pytest.approx(dfb.n_th, rel=0.02)


def test_diode_modulation_response_against_simulation(dfb):
    I0, dI, f = 40E-3, 0.2E-3, 4E9
    t = np.arange(0, 6E-9, 0.1E-12)
    N, S, _ = dfb.simulate(t, I0 + dI*np.sin(2*np.pi*f*t))
    sel = t > 3E-9
    x = S[sel] - S[sel].mean()
    ph = np.exp(-2j*np.pi*f*t[sel])
    amp = 2*np.abs(np.mean(x*ph))
    N0, S0 = dfb.steady_state(I0)
    dc = np.real(dfb.modulation_response(0.0, I0, normalize=False))[0]
    h = np.abs(dfb.modulation_response(f, I0, normalize=False))[0]
    assert amp/dI == pytest.approx(h, rel=0.02)
    assert np.abs(dfb.modulation_response(f, I0))[0] == pytest.approx(h/dc, rel=1E-9)
    f_R, gamma = dfb.relaxation(I0)
    # textbook estimate f_R = sqrt(G_N S/tau_p)/(2 pi)
    G_N = dfb.confinement*dfb.v_g*dfb.a/dfb.volume
    assert f_R == pytest.approx(np.sqrt(G_N*S0/dfb.tau_p)/(2*np.pi), rel=0.1)
    assert dfb.bandwidth_3db(I0) > f_R


def test_diode_noise_against_langevin(dfb):
    I0 = 30E-3
    t = np.arange(0, 3E-9, 0.2E-12)
    N, S, phi = dfb.simulate(t, I0, noise=True, rng=3, realizations=64)
    _, S0 = dfb.steady_state(I0)
    sel = t > 0.5E-9
    rel_var = np.var(S[:, sel])/S0**2
    f = np.linspace(0, 200E9, 40001)
    rin = dfb.rin(f, I0)
    # one-sided RIN integrated over frequency = relative variance (step 0.2 ps:
    # simulated band up to 2.5 THz, the RIN beyond 200 GHz is negligible)
    assert rel_var == pytest.approx(np.trapezoid(rin, f), rel=0.15)
    # phase diffusion: var(phi(t)) = 2 pi dnu t for t >> 1/gamma
    dphi = phi[:, -1] - phi[:, np.searchsorted(t, 1E-9)]
    dnu = np.var(dphi)/(2*np.pi*(t[-1] - 1E-9))
    assert dnu == pytest.approx(dfb.linewidth(I0), rel=0.3)
    # without the alpha factor only the direct phase noise is left
    plain = L.DiodeLaser(alpha=0.0)
    _, _, phi0 = plain.simulate(t, I0, noise=True, rng=5, realizations=256)
    dphi0 = phi0[:, -1] - phi0[:, np.searchsorted(t, 1E-9)]
    dnu0 = np.var(dphi0)/(2*np.pi*(t[-1] - 1E-9))
    assert dnu0 == pytest.approx(plain.linewidth(I0), rel=0.2)
    assert dfb.linewidth(I0)/plain.linewidth(I0) == pytest.approx(26.0)


def test_diode_nrz_chirp_sign(dfb):
    bits = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0])
    t, A, I = dfb.nrz(bits, 10.0, 20E-3, 40E-3, samples_per_bit=32)
    assert len(t) == len(bits)*32
    P = np.abs(A)**2
    ones, zeros = P[bits.repeat(32) == 1].mean(), P[bits.repeat(32) == 0].mean()
    assert ones > 2*zeros
    nu = -np.gradient(np.unwrap(np.angle(A)), t[1] - t[0])
    dP = np.gradient(P)
    # blue shifted (positive frequency) on rising edges
    # (relative to the mean: the adiabatic chirp shifts the ones to the blue)
    nu = nu - nu.mean()
    assert np.mean(nu[dP > 0.5*dP.max()]) > 0
    assert np.mean(nu[dP < 0.5*dP.min()]) < 0


def test_diode_mode_spectrum(dfb):
    nu, P_fp = dfb.mode_spectrum(30E-3)
    nu, P_dfb = dfb.mode_spectrum(30E-3, selection=(0, 0.05))
    smsr_fp = 10*np.log10(np.sort(P_fp)[-1]/np.sort(P_fp)[-2])
    smsr_dfb = 10*np.log10(np.sort(P_dfb)[-1]/np.sort(P_dfb)[-2])
    assert smsr_dfb > 30
    assert smsr_fp < smsr_dfb
    nu, P_low = dfb.mode_spectrum(5E-3)
    assert np.sort(P_low)[-1]/np.sort(P_low)[-2] < 1.5
