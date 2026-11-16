#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.fibrecomm.

References
----------
G. P. Agrawal, "Fiber-Optic Communication Systems", 4th ed., Wiley (2010).
"""

import numpy as np

from pyoptics import fibrecomm as fc


def test_prbs():
    b = fc.prbs(7)
    assert len(b) == 127 and b.sum() == 64
    # all 7-bit windows except all zeros appear once
    s = np.concatenate([b, b[:6]])
    words = {tuple(s[i:i + 7]) for i in range(127)}
    assert len(words) == 127 and (0,)*7 not in words


def test_gaussian_broadening_and_chirp():
    t = np.linspace(-300, 300, 8192, endpoint=False)
    b2 = fc.beta2_from_D(17, 1550)
    for C in (0.0, -4.0, 2.0):
        A = fc.propagate(t, fc.gaussian_pulse(t, 10.0, C), 2.0, b2)
        P = np.abs(A)**2
        w = np.sqrt(2*np.sum(P*t**2)/np.sum(P))
        assert np.isclose(w, fc.gaussian_width(2.0, 10.0, b2, C), rtol=1E-6)
    # energy and loss
    A = fc.propagate(t, fc.gaussian_pulse(t, 10.0), 10.0, b2, alpha_db=0.2)
    assert np.isclose(np.sum(np.abs(A)**2)/np.sum(np.abs(fc.gaussian_pulse(t, 10.0))**2), 10**-0.2)


def test_group_delay_sign():
    # a tone at +Omega (exp(-i Omega t)) is delayed by beta2 Omega z
    t = np.linspace(-500, 500, 8192, endpoint=False)
    W = 2*np.pi*40/1000.0
    A = fc.gaussian_pulse(t, 20.0)*np.exp(-1j*W*t)
    B = fc.propagate(t, A, 1.0, beta2=20.0)
    P = np.abs(B)**2
    assert np.isclose(np.sum(P*t)/np.sum(P), 20.0*W, rtol=1E-3)


def test_ber_q():
    assert np.isclose(fc.ber(6.0), 9.866E-10, rtol=1E-3)
    assert np.isclose(fc.q_from_ber(1E-9), 5.998, rtol=1E-3)


def test_sensitivity_and_apd():
    P_pin = fc.sensitivity(6, 7.5)
    M, P_apd = fc.optimum_apd_gain(6, 7.5)
    assert -21 < fc.dbm(P_pin) < -17
    assert 2 < M < 20 and P_apd < P_pin
    # thermal limited PIN: Q sigma_T/R
    s_shot, s_T = fc.receiver_noise(P_pin, 7.5)
    assert s_shot < 0.2*s_T
    assert np.isclose(fc.excess_noise(1.0), 1.0)


def test_eye_and_noise_free_link():
    bits = fc.prbs(7)
    t, A = fc.nrz(bits, 10.0, 32, rise_time=20.0)
    I = fc.detect(t, A, 20.0, noise=False)
    tau, traces = fc.eye_diagram(t, I, 10.0, offset=50.0)
    assert traces.shape[1] == 65
    op = fc.eye_opening(traces)
    assert op > 0.8
    # dispersion closes the eye, more with a chirped directly modulated laser
    b2 = fc.beta2_from_D(17, 1550)
    ops = []
    for ah in (0.0, 5.0):
        t, A = fc.nrz(bits, 10.0, 32, alpha_h=ah)
        I = fc.detect(t, fc.propagate(t, A, 60.0, b2), 7.5, noise=False)
        ops.append(fc.eye_opening(fc.eye_diagram(t, I, 10.0, offset=50.0)[1]))
    assert ops[1] < ops[0] < op


def test_budgets():
    b = fc.power_budget(0.0, -20.0, 40.0, 0.2, splices=10, splice_db=0.1, connectors=2,
                        connector_db=0.5)
    assert np.isclose(b["received_dbm"], -10.0) and np.isclose(b["margin_db"], 10.0)
    assert np.isclose(fc.rise_time_budget(3, 4), 5)
    assert fc.dispersion_penalty(1.0, 10, 17, 0.1) < 0.1
    assert np.isinf(fc.dispersion_penalty(10.0, 100, 17, 1.0))
    assert np.isclose(fc.mw(fc.dbm(2.5)), 2.5)


def test_detect_filter_and_noise():
    # a sine at the 3 dB bandwidth passes with |H|**2 = 1/2
    n, dt = 4096, 1.0
    t = np.arange(n)*dt
    f3 = 64/(n*dt)*1E3                       # GHz, an exact FFT bin
    P = 1 + 0.5*np.cos(2*np.pi*f3*1E-3*t)
    I = fc.detect(t, np.sqrt(P), f3, noise=False)
    assert np.isclose((I.max() - I.min())/2, 0.5/np.sqrt(2), rtol=1E-6)
    # dark receiver: the sample variance is the thermal noise in the noise bandwidth
    I = fc.detect(t, np.zeros(n), f3, rng=1)
    s_th = fc.receiver_noise(0.0, fc.noise_bandwidth(f3))[1]
    assert np.isclose(I.std(), s_th, rtol=0.05)
    assert np.isclose(fc.noise_bandwidth(1.0), 1.0645, atol=1E-4)


def test_rin_noise():
    # RIN of -130 dB/Hz in 10 GHz: relative rms 1E-13*1E10 -> 0.0316
    assert np.isclose(fc.rin_noise(2.0, 10.0, -130.0, 0.5), 2.0*0.5*np.sqrt(1E-3))
    n, dt, f3 = 8192, 1.0, 5.0
    t = np.arange(n)*dt
    I0 = fc.detect(t, np.ones(n), f3, noise=True, rng=2)
    I1 = fc.detect(t, np.ones(n), f3, noise=True, rng=2, rin_db_hz=-120.0)
    nbw = fc.noise_bandwidth(f3)
    s_sh, s_th = fc.receiver_noise(1.0, nbw)
    s_rin = fc.rin_noise(1.0, nbw, -120.0)
    assert np.isclose(I0.std(), np.hypot(s_sh, s_th), rtol=0.05)
    assert np.isclose(I1.std(), np.sqrt(s_sh**2 + s_th**2 + s_rin**2), rtol=0.05)
