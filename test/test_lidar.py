#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the LiDAR module: link budget, detectors, eye safety, FMCW,
coherent detection and beam scanners."""

import numpy as np
import pytest

from pyoptics import lidar


def test_lambertian_return_inverse_square_and_formula():
    P = lidar.lambertian_return(1.0, [10.0, 20.0], 0.5, 0.02)
    assert P[0]/P[1] == pytest.approx(4.0)
    assert P[0] == pytest.approx(0.5/np.pi*np.pi*0.01**2/100)
    assert lidar.lambertian_return(1.0, 10.0, 0.5, 0.02, incidence=np.pi/3,
                                   transmission=0.9) == pytest.approx(P[0]*0.5*0.81)


def test_extinction_kim():
    # clear air: q = 1.6; fog: no wavelength dependence
    a = lidar.extinction_coefficient(60.0, [550.0, 1100.0])
    assert a[0] == pytest.approx(3.912/60)
    assert a[0]/a[1] == pytest.approx(2**1.6)
    f = lidar.extinction_coefficient(0.2, [905.0, 1550.0])
    assert f[0] == pytest.approx(f[1])


def test_solar_irradiance_magnitude():
    E = lidar.solar_irradiance([905.0, 1550.0])
    assert 0.6 < E[0] < 1.0 and 0.15 < E[1] < 0.3
    # above the atmosphere the model integrates to the solar constant
    lam = np.linspace(100, 1e5, 400000)
    E0 = lidar.solar_irradiance(lam, airmass=0.0)
    assert np.trapezoid(E0, lam) == pytest.approx(1361.0, rel=2e-3)


def test_background_independent_of_range_and_scales_with_fov():
    P1 = lidar.background_power(1.0, 0.3, 0.02, 1e-3, 10.0)
    P2 = lidar.background_power(1.0, 0.3, 0.02, 2e-3, 10.0)
    assert P2/P1 == pytest.approx(4.0, rel=1e-5)
    omega = np.pi*(0.5e-3)**2
    assert P1 == pytest.approx(10*0.3/np.pi*np.pi*1e-4*omega, rel=1e-5)


def test_photons():
    assert lidar.photons(lidar.H*lidar.C/1e-6, 1000.0) == pytest.approx(1.0)


def test_apd_snr_limits():
    # PIN without shot noise: SNR = R P/(i_n sqrt(B))
    s = lidar.apd_snr(1e-6, 1e8, 0.5, gain=1.0, input_noise=1e-12)
    expect = 0.5e-6/np.sqrt(2*lidar.Q_E*0.5e-6*1e8 + 1e-24*1e8)
    assert s == pytest.approx(expect)
    # an optimum gain exists with background light
    M = np.logspace(0, 3, 200)
    snr = lidar.apd_snr(1e-8, 1e8, 0.5, M, k_A=0.3, P_bg=1e-7)
    assert 0 < np.argmax(snr) < len(M) - 1


def test_detection_and_false_alarms():
    assert lidar.detection_probability(5.0, 5.0) == pytest.approx(0.5)
    assert lidar.detection_probability(10.0, 5.0) > 0.999
    r = lidar.false_alarm_rate([0.0, 4.0], 1e8)
    assert r[0] == pytest.approx(1e8/np.sqrt(3))
    assert r[1]/r[0] == pytest.approx(np.exp(-8))


def test_rice_rate_monte_carlo():
    rng = np.random.default_rng(0)
    n, B = 2**20, 0.05           # band limited white noise, unit sample rate
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    X[f > B] = 0
    x = np.fft.irfft(X, n)
    x /= x.std()
    T = 2.0
    up = np.count_nonzero((x[:-1] < T) & (x[1:] >= T))
    assert up/n == pytest.approx(lidar.false_alarm_rate(T, B), rel=0.05)


def test_coates_round_trip_and_monte_carlo():
    n = np.full(50, 0.05)
    n[30] += 1.0
    p = lidar.coates_histogram(n)
    assert p.sum() == pytest.approx(1 - np.exp(-n.sum()))
    assert lidar.coates_correction(p*1000, 1000) == pytest.approx(n)
    # Monte Carlo of first photons
    rng = np.random.default_rng(1)
    N = 200000
    counts = rng.poisson(n, size=(N, len(n)))
    first = np.argmax(counts > 0, axis=1)[counts.any(axis=1)]
    h = np.bincount(first, minlength=len(n))
    assert h/N == pytest.approx(p, abs=3e-3)


def test_max_unambiguous_range():
    assert lidar.max_unambiguous_range(1e6) == pytest.approx(149.896, rel=1e-4)


def test_eye_safety_known_values():
    # class 1 CW power: about 1 mW at 905 nm, 10 mW at 1550 nm (7 and 3.5 mm)
    assert lidar.ael_class1(905.0, 100.0)/100 == pytest.approx(3.9e-4*10**(0.002*205), rel=0.01)
    assert lidar.ael_class1(1550.0, 100.0)/100 == pytest.approx(1e-2)
    assert lidar.ael_class1(905.0, 5e-9) == pytest.approx(7.7e-8*10**0.41)
    assert lidar.ael_class1(1550.0, 5e-9) == pytest.approx(8e-3)
    # continuous at the breakpoints
    for lam, tb in ((905.0, 5e-6), (1064.0, 1.3e-5), (905.0, 10.0), (1550.0, 0.35), (1550.0, 10.0)):
        a, b = lidar.ael_class1(lam, tb*(1 - 1e-9)), lidar.ael_class1(lam, tb*(1 + 1e-9))
        assert a == pytest.approx(b, rel=0.05)
    assert lidar.c7(1100.0) == 1.0 and lidar.c4(1200.0) == 5.0
    with pytest.raises(ValueError):
        lidar.ael_class1(1450.0, 1.0)


def test_measurement_aperture_and_fraction():
    assert lidar.measurement_aperture(905.0, 1.0) == pytest.approx(7e-3)
    assert lidar.measurement_aperture(1550.0, 1e-3) == pytest.approx(1e-3)
    assert lidar.measurement_aperture(1550.0, 100.0) == pytest.approx(3.5e-3)
    assert lidar.aperture_fraction(7e-3, 7e-3) == pytest.approx(1 - np.exp(-2))


def test_class1_pulse_train_rules():
    E, rules = lidar.class1_pulse_energy(905.0, 5e-9, 1e5, return_rules=True)
    assert E == min(rules.values())
    assert rules["average"] == pytest.approx(lidar.ael_class1(905.0, 100.0)/1e7)
    # a single pulse per T2 is limited by the single pulse rule
    E1 = lidar.class1_pulse_energy(905.0, 5e-9, 0.01)
    assert E1 == pytest.approx(lidar.ael_class1(905.0, 5e-9))
    # no C5 rule at 1550 nm
    _, r = lidar.class1_pulse_energy(1550.0, 5e-9, 1e5, return_rules=True)
    assert np.isinf(r["train"])


def test_fmcw_round_trip():
    B, T, lam = 1e9, 10e-6, 1550e-9
    fu, fd = lidar.fmcw_beat_frequencies(75.0, B, T, velocity=12.0, wavelength=lam)
    R, v = lidar.fmcw_range_velocity(fu, fd, B, T, lam)
    assert R == pytest.approx(75.0) and v == pytest.approx(12.0)
    assert lidar.range_resolution(1e9) == pytest.approx(0.1499, rel=1e-3)


def test_fmcw_beat_signal_frequency():
    B, T, R = 1e9, 10e-6, 60.0
    fs = 2e9
    t = 1e-6 + np.arange(16384)/fs
    b = lidar.fmcw_beat_signal(t, R, B, T, velocity=5.0)
    f = np.fft.fftfreq(len(t), 1/fs)
    peak = f[np.argmax(np.abs(np.fft.fft(b)))]
    fu, _ = lidar.fmcw_beat_frequencies(R, B, T, 5.0)
    assert peak == pytest.approx(fu, abs=fs/len(t))
    bd = lidar.fmcw_beat_signal(t, R, B, T, velocity=5.0, down=True)
    peak_d = f[np.argmax(np.abs(np.fft.fft(bd)))]
    # down chirp: the beat changes sign, magnitude kappa*tau + f_D
    assert abs(peak_d) == pytest.approx(lidar.fmcw_beat_frequencies(R, B, T, 5.0)[1],
                                        abs=fs/len(t))


def test_phase_noise_coherent_fraction():
    B, T, R, lw = 1e9, 10e-6, 15.0, 1e6
    fs = 1e9
    t = 2e-6 + np.arange(4096)/fs
    tau = 2*R/lidar.C
    b0 = lidar.fmcw_beat_signal(t, R, B, T)
    c = [np.mean(np.real(lidar.fmcw_beat_signal(t, R, B, T, linewidth=lw, rng=seed)*np.conj(b0)))
         for seed in range(100)]
    # <exp(i dphi)> = exp(-pi lw tau): the square root of the coherent fraction
    assert np.mean(c) == pytest.approx(np.sqrt(lidar.coherent_fraction(tau, lw)), rel=0.05)


def test_nonlinearity_shifts_beat():
    B, T, R = 1e9, 10e-6, 30.0
    fs = 2e9
    t = 1e-6 + np.arange(8192)/fs
    eps = lambda tt: 1e9*0.01*np.sin(2*np.pi*tt/T)       # 1 % of B
    b = lidar.fmcw_beat_signal(t, R, B, T, nonlinearity=eps)
    b0 = lidar.fmcw_beat_signal(t, R, B, T)
    inst = np.diff(np.unwrap(np.angle(b)))*fs/(2*np.pi)
    inst0 = np.diff(np.unwrap(np.angle(b0)))*fs/(2*np.pi)
    tau = 2*R/lidar.C
    expect = 1e7*(np.sin(2*np.pi*t[1:]/T) - np.sin(2*np.pi*(t[1:] - tau)/T))
    assert np.max(np.abs(inst - inst0 - expect)) < 0.02*np.max(np.abs(expect))


def test_coherent_snr_shot_noise_limit():
    P_s, B, Rd = 1e-12, 1e6, 0.8
    s = lidar.coherent_snr(P_s, 1e-2, B, Rd, input_noise=0.0, rin_db=-200)
    assert s == pytest.approx(Rd*P_s/(lidar.Q_E*B), rel=1e-6)
    # RIN of a strong LO hurts a single detector, not a balanced pair
    single = lidar.coherent_snr(P_s, 1e-2, B, Rd, rin_db=-140, balanced=False)
    bal = lidar.coherent_snr(P_s, 1e-2, B, Rd, rin_db=-140, balanced=True, cmrr_db=30)
    assert bal > 5*single


def test_resolvable_spots():
    assert lidar.resolvable_spots(0.2, 1e-3, 1e-6) == pytest.approx(200)


def test_galvo_xy():
    x, y = lidar.galvo_xy(0.0, 0.0, 0.02, 1.0)
    assert abs(x) < 1e-12 and abs(y) < 1e-12
    x, y = lidar.galvo_xy(np.radians(5), 0.0, 0.02, 1.0)
    assert x == pytest.approx(np.tan(np.radians(10))*1.02, rel=1e-6)
    x, _ = lidar.galvo_xy(np.radians(5), np.radians([0, 5]), 0.02, 1.0)
    assert x[1] > x[0]          # bow: the x lever grows with the y angle
    _, y = lidar.galvo_xy(0.0, np.radians(5), 0.02, 1.0)
    assert abs(y) == pytest.approx(np.tan(np.radians(10)), rel=1e-6)


def test_polygon_scan():
    rot = np.radians(np.linspace(-25, 25, 11))
    a, p = lidar.polygon_scan(rot, 6, 0.02)
    assert np.allclose(np.unwrap(a), 2*rot, atol=1e-9)
    a, _ = lidar.polygon_scan(np.radians([29.0, 31.0]), 6, 0.02)
    # facet change: the scan jumps back by 4 pi/N
    assert np.degrees(a[0] - a[1]) == pytest.approx(116.0, abs=1e-6)


def test_lissajous():
    x, y = lidar.lissajous(np.array([0.0, 0.25]), 1.0, 2.0, 2.0, 3.0)
    assert x[1] == pytest.approx(2.0) and y[0] == pytest.approx(3.0)


def test_opa():
    s = np.linspace(-1, 1, 4001)
    I = lidar.opa_array_factor(s, 32, 2e-6, 1e-6, steer=0.1)
    assert I[np.argmin(np.abs(s - 0.1))] == pytest.approx(1.0, rel=1e-3)
    # grating lobe at sin = 0.1 - lambda/d = -0.4
    i = np.argmin(np.abs(s + 0.4))
    assert I[i] == pytest.approx(1.0, rel=1e-3)
    # half wavelength pitch: no grating lobe
    I2 = lidar.opa_array_factor(s, 32, 0.5e-6, 1e-6, steer=0.5)
    assert I2[np.abs(s - 0.5) > 0.1].max() < 0.1
    # phase errors lower the peak by about exp(-sigma**2) (Ruze)
    peaks = [lidar.opa_array_factor(0.0, 256, 2e-6, 1e-6, phase_error=0.5, rng=k) for k in range(40)]
    assert np.mean(peaks) == pytest.approx(np.exp(-0.25) + (1 - np.exp(-0.25))/256, rel=0.05)
    # quantization to 2 bits
    q = lidar.opa_array_factor(0.1, 64, 2e-6, 1e-6, steer=0.1, phase_bits=2)
    assert 0.7 < q < 0.9           # about sinc**2(1/4) = 0.81


def test_grating_emission_angle():
    th = lidar.grating_emission_angle(1550e-9, 2.8, 600e-9)
    assert np.sin(th) == pytest.approx(2.8 - 1550/600)
    th = lidar.grating_emission_angle(np.array([1500e-9, 1600e-9]), lambda l: 2.8 - 0*l, 600e-9)
    assert th[0] > th[1]
    assert np.isnan(lidar.grating_emission_angle(1550e-9, 3.0, 2000e-9))


def test_risley():
    w, n = np.radians(5), 1.5
    d = lidar.risley_direction([0.0, 0.0], [0.0, np.pi], w, n)
    dev = np.degrees(np.arcsin(np.hypot(d[:, 0], d[:, 1])))
    assert dev[0] == pytest.approx(2*np.degrees((n - 1)*w), rel=0.03)
    assert dev[1] < 1e-6
    assert d[0, 0] < 0                     # towards the thick end
    assert np.allclose(np.linalg.norm(d, axis=1), 1)
