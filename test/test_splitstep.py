#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the split-step solvers against solitons, SPM, linear dispersion
and the Raman self-frequency shift."""

import numpy as np
import pytest

from pyoptics import splitstep as ss
from pyoptics import fibrecomm


T = np.linspace(-20, 20, 2048, endpoint=False)   # ps
B2, G, T0 = -20.0, 1.3, 1.0                      # ps²/km, 1/(W km), ps


def test_spectrum_convention():
    # a carrier offset exp(-i W0 t) is a positive frequency offset W0
    W0 = 2*np.pi*0.5
    A = np.exp(-T**2)*np.exp(-1j*W0*T)
    Om, S = ss.spectrum(T, A)
    assert Om[np.argmax(np.abs(S))] == pytest.approx(W0, abs=0.01)
    # Parseval
    dt = T[1] - T[0]
    assert np.sum(np.abs(S)**2)*(Om[1] - Om[0])/(2*np.pi) == pytest.approx(np.sum(np.abs(A)**2)*dt)


def test_linear_matches_fibrecomm():
    A0 = fibrecomm.gaussian_pulse(T, 1.0, C=-2.0)
    a = ss.nlse(T, A0, 0.07, betas=[B2, 0.1], steps=10)
    b = fibrecomm.propagate(T, A0, 0.07, beta2=B2, beta3=0.1)
    assert np.allclose(a, b, atol=1E-12)
    c = ss.gnlse(T, A0, 0.07, betas=[B2, 0.1], steps=3)
    assert np.allclose(c, b, atol=1E-12)


def test_fundamental_soliton():
    A0 = ss.soliton(T, T0, B2, G)
    z0 = ss.soliton_period(T0, B2)
    for A in (ss.nlse(T, A0, 5*z0, [B2], G, steps=4000), ss.gnlse(T, A0, 5*z0, [B2], G, steps=500)):
        assert np.max(np.abs(np.abs(A) - np.abs(A0))) < 1E-5*np.abs(A0).max()
    # phase gamma P0 z/2 = pi/4 per soliton period
    A = ss.gnlse(T, A0, z0, [B2], G, steps=400)
    assert np.angle(A[1024]/A0[1024]) == pytest.approx(np.pi/4, abs=1E-8)


def test_third_order_soliton_period_and_adaptive():
    A3 = ss.soliton(T, T0, B2, G, N=3)
    z0 = ss.soliton_period(T0, B2)
    A = ss.gnlse(T, A3, z0, [B2], G, steps=2000)
    assert np.max(np.abs(np.abs(A) - np.abs(A3))) < 1E-5*np.abs(A3).max()
    A_ad = ss.gnlse(T, A3, z0, [B2], G, steps=20, tol=1E-8)
    assert np.max(np.abs(A_ad - A)) < 1E-4*np.abs(A3).max()


def test_spm_without_dispersion():
    P0, L = 2.0, 3.0
    A0 = np.sqrt(P0)*np.exp(-T**2/2)
    A = ss.nlse(T, A0, L, gamma=G, steps=7)
    assert np.allclose(A, A0*np.exp(1j*G*np.abs(A0)**2*L))
    phi = ss.spm_phase(P0, G, L)
    Om, S = ss.spectrum(T, A)
    Om0, S0 = ss.spectrum(T, A0)
    rms = lambda w: np.sqrt(np.sum(Om**2*w)/np.sum(w))
    assert rms(np.abs(S)**2)/rms(np.abs(S0)**2) == pytest.approx(ss.spm_broadening(phi), rel=1E-3)


def test_loss_and_effective_length():
    A0 = np.sqrt(1.0)*np.exp(-T**2/2)
    alpha = 0.046
    A = ss.nlse(T, A0, 50.0, gamma=G, alpha=alpha, steps=2000)
    assert np.abs(A[1024])**2 == pytest.approx(np.exp(-alpha*50), rel=1E-9)
    phi = ss.spm_phase(1.0, G, 50.0, alpha)
    assert np.angle(A[1024]*np.exp(-1j*phi)) == pytest.approx(0.0, abs=1E-5)


def test_history():
    A0 = ss.soliton(T, T0, B2, G)
    A, z, hist = ss.nlse(T, A0, 1.0, [B2], G, steps=100, n_save=11)
    assert z.shape == (11,) and hist.shape == (11, len(T))
    assert np.allclose(hist[-1], A)
    A2, z2, hist2 = ss.gnlse(T, A0, 1.0, [B2], G, steps=100, n_save=11)
    assert np.allclose(z2, z) and np.allclose(hist2[-1], A2)


def test_raman_response():
    t = np.linspace(0, 1, 200001)
    h = ss.raman_response(t)
    assert np.trapezoid(h, t) == pytest.approx(1.0, abs=1E-6)
    assert ss.raman_spectrum(0.0) == pytest.approx(1.0)
    # Raman gain peak near 13 THz below the pump
    Om = np.linspace(-2*np.pi*30, 0, 3001)
    f_peak = Om[np.argmax(np.imag(-ss.raman_spectrum(Om)))]/(2*np.pi)
    assert -15 < f_peak < -11


def test_soliton_self_frequency_shift():
    T0s = 0.1
    t = np.linspace(-10, 10, 2**13, endpoint=False)
    A0 = ss.soliton(t, T0s, B2, G)
    L = 20*ss.dispersion_length(T0s, B2)
    A = ss.gnlse(t, A0, L, [B2], G, raman=0.18, steps=2000)
    Om, S = ss.spectrum(t, A)
    w = np.abs(S)**2
    shift = np.sum(Om*w)/np.sum(w)
    tt = np.linspace(0, 1, 200001)
    T_R = 0.18*np.trapezoid(tt*ss.raman_response(tt), tt)
    assert shift < 0
    assert shift == pytest.approx(ss.ssfs_rate(T0s, B2, T_R)*L, rel=0.15)


def test_self_steepening_shifts_peak_to_trailing_edge():
    # 20 fs pulse at 1550 nm (omega0 = 1216 rad/ps), five nonlinear lengths
    t = np.linspace(-1, 1, 4096, endpoint=False)
    T0s, P0 = 0.02, 1000.0
    A0 = np.sqrt(P0)*np.exp(-t**2/(2*T0s**2))
    A = ss.gnlse(t, A0, 5/(G*P0), gamma=G, omega0=1216.0, steps=400)
    assert t[np.argmax(np.abs(A))] > 0.1*T0s
    B = ss.gnlse(t, A0, 5/(G*P0), gamma=G, steps=400)
    assert abs(t[np.argmax(np.abs(B))]) < 1E-3


def test_formulas():
    assert ss.soliton_order(T0, abs(B2)/(G*T0**2)*9, G, B2) == pytest.approx(3.0)
    assert ss.fission_length(T0, B2, 10) == pytest.approx(ss.dispersion_length(T0, B2)/10)
    assert ss.nonlinear_length(2.0, G) == pytest.approx(1/(2*G))
    assert ss.effective_length(1E6, 0.05) == pytest.approx(20.0)


def test_beam_diffraction():
    lam, w0 = 1E-6, 50E-6
    x = np.linspace(-1E-3, 1E-3, 1024, endpoint=False)
    zR = np.pi*w0**2/lam
    E, z, peak = ss.beam_propagate(x, np.exp(-x**2/w0**2), zR, lam, steps=10)
    # 1D: the peak intensity drops by the factor sqrt(2) at z_R
    assert peak[-1] == pytest.approx(1/np.sqrt(2), rel=1E-6)
    xx = (x, x)
    X, Y = np.meshgrid(x[::4], x[::4])
    E2, z2, peak2 = ss.beam_propagate((x[::4], x[::4]), np.exp(-(X**2 + Y**2)/w0**2), zR, lam, steps=5)
    assert peak2[-1] == pytest.approx(0.5, rel=1E-4)
    assert xx[0] is x


def test_beam_kerr_phase_and_mpa():
    lam = 1E-6
    x = np.linspace(-1E-3, 1E-3, 256, endpoint=False)
    E0 = np.full(len(x), 1E3 + 0j)       # plane wave, 1E6 W/m²
    E, z, peak = ss.beam_propagate(x, E0, 0.01, lam, n2=1E-12, max_phase=0.01)
    assert np.angle(E[0]) == pytest.approx(np.angle(np.exp(1j*2*np.pi/lam*1E-12*1E6*0.01)))
    E, z, peak = ss.beam_propagate(x, E0, 0.01, lam, n2=0.0, mpa=(1E-4, 2), steps=10)
    assert np.abs(E[0])**2 == pytest.approx(1E6/(1 + 1E-4*1E6*0.01), rel=1E-9)
