#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.detectors.

References
----------
B. E. A. Saleh and M. C. Teich, "Fundamentals of Photonics", 3rd ed.,
Wiley (2019).

R. Loudon, "The Quantum Theory of Light", 3rd ed., Oxford University Press
(2000).
"""

import numpy as np
import pytest

from pyoptics import detectors as d
from pyoptics import fibrecomm as fc
from pyoptics import lidar


def test_responsivity_and_cutoff():
    # an ideal detector at 1240 nm gives 1 A/W; eta round trip
    assert np.isclose(d.responsivity(d._HC_EV_NM), 1.0)
    assert np.isclose(d.quantum_efficiency(d.responsivity(850, 0.8), 850), 0.8)
    assert np.isclose(d.cutoff_wavelength(1.0), 1239.84, atol=0.01)
    # Si 1.12 eV, InGaAs 0.74 eV at room temperature; the gap grows on cooling
    assert abs(d.bandgap("Si") - 1.12) < 0.01
    assert abs(d.bandgap("InGaAs") - 0.74) < 0.01
    assert d.bandgap("Si", 77) > d.bandgap("Si", 300)
    # HgCdTe: LWIR near x = 0.2 at 77 K, MWIR near x = 0.3; CdTe 1.6 eV
    assert 12000 < d.cutoff_wavelength(d.bandgap("HgCdTe", 77, x=0.2)) < 16000
    assert 4500 < d.cutoff_wavelength(d.bandgap("HgCdTe", 77, x=0.3)) < 5500
    assert abs(d.hgcdte_bandgap(1.0, 0) - 1.65) < 0.05
    with pytest.raises(ValueError):
        d.bandgap("HgCdTe")


def test_absorption():
    # Si: 1 µm absorption depth around 500 nm, 10 µm near 800 nm, transparent beyond 1.2 µm
    assert 0.5E-6 < d.absorption_depth("Si", 500) < 1.5E-6
    assert 8E-6 < d.absorption_depth("Si", 800) < 20E-6
    assert d.absorption_coefficient("Si", 1300) == 0
    lam = np.linspace(400, 1200, 50)
    assert np.all(np.diff(d.absorption_coefficient("Si", lam)) < 0)
    # InGaAs: about 1.5 µm at 1550 nm, strongly falling past the cutoff
    assert 1E-6 < d.absorption_depth("InGaAs", 1550) < 2E-6
    lc = d.cutoff_wavelength(d.bandgap("InGaAs"))
    assert d.absorption_coefficient("InGaAs", lc + 100) < 1E-2*d.absorption_coefficient("InGaAs", 1550)
    # smooth join of square root and Urbach tail
    E1 = d.bandgap("InGaAs") + 0.005
    l1 = d.cutoff_wavelength(E1)
    a = d.absorption_coefficient("InGaAs", l1*np.array([1 - 1E-9, 1 + 1E-9]))
    assert np.isclose(a[0], a[1], rtol=1E-4)
    # HgCdTe: thousands per cm above the gap
    a = d.absorption_coefficient("HgCdTe", 10000, 77, x=0.2)
    assert 1E5 < a < 1E6
    # layer efficiency: thick layer -> 1 - R, thin layer -> alpha d
    assert np.isclose(d.quantum_efficiency_layer(1E6, 1E-3, 0.3), 0.7)
    assert np.isclose(d.quantum_efficiency_layer(1E3, 1E-6), 1E-3, rtol=1E-3)


def test_bandwidth():
    C = d.junction_capacitance(1E-6, 1E-6, 11.7)
    assert np.isclose(C, 8.854E-12*11.7, rtol=1E-3)
    assert np.isclose(d.rc_bandwidth(1E-12, 50), 1/(2*np.pi*50E-12))
    w = np.linspace(0.1E-6, 10E-6, 2000)
    f = d.photodiode_bandwidth(np.pi*(25E-6)**2, w, eps_r=13.9, v_sat=6E4)
    assert np.all(f < d.transit_bandwidth(w, 6E4))
    # optimum where f_RC = f_tr
    i = np.argmax(f)
    f_rc = d.rc_bandwidth(d.junction_capacitance(np.pi*(25E-6)**2, w[i], 13.9))
    assert np.isclose(f_rc, d.transit_bandwidth(w[i], 6E4), rtol=0.02)


def test_noise_and_fibrecomm_wrappers():
    P, B = 1E-5, 7E9
    s, t = d.noise_currents(P, B, 0.8)
    assert np.isclose(s, np.sqrt(2*d.Q_E*0.8*P*B))
    assert np.isclose(t, np.sqrt(4*d.K_B*300*2*B/50), rtol=1E-3)
    # fibrecomm keeps its API (mW, GHz, mA)
    s2, t2 = fc.receiver_noise(P*1E3, B/1E9, 0.8, dark_na=0)
    assert np.isclose(s2, s*1E3) and np.isclose(t2, t*1E3)
    s3, _ = fc.receiver_noise(P*1E3, B/1E9, 0.8, gain=10, k_A=0.7)
    assert np.isclose(s3**2, (s*1E3)**2*100*d.excess_noise(10, 0.7))
    assert np.isclose(fc.sensitivity(6, 7, 0.8), d.sensitivity(6, 7E9, 0.8)*1E3)
    M, Pm = fc.optimum_apd_gain(6, 7, 0.8)
    assert np.isclose(Pm, d.optimum_apd_gain(6, 7E9, 0.8)[1]*1E3)
    assert fc.ber is d.ber and fc.q_factor is d.q_factor and lidar.excess_noise is d.excess_noise
    assert np.isclose(d.ber(d.q_from_ber(1E-9)), 1E-9)
    assert np.isclose(fc.rin_noise(1.0, 10, -140, 0.8), d.rin_noise(1.0, 10E9, -140, 0.8))
    # excess noise limits: 1 at M = 1, k M for large M
    assert np.isclose(d.excess_noise(1, 0.5), 1)
    assert np.isclose(d.excess_noise(1E6, 0.02), 0.02*1E6, rtol=1E-3)


def test_nep_detectivity():
    nep = d.nep(0.5, dark_current=1E-9)
    assert np.isclose(nep, np.sqrt(2*d.Q_E*1E-9)/0.5)
    assert np.isclose(d.nep_from_detectivity(d.specific_detectivity(nep, 1E-6), 1E-6), nep)
    # 1 mm² and NEP 1E-14 W/sqrt(Hz): D* = 0.1 cm/1E-14 = 1E13 Jones
    assert np.isclose(d.specific_detectivity(1E-14, 1E-6), 1E13)
    # BLIP of a 300 K hemisphere: a few 1E10 Jones in the LWIR, rising 1/sin(theta) with a cold shield
    D10 = d.blip_detectivity(10000)
    assert 3E10 < D10 < 8E10
    assert np.isclose(d.blip_detectivity(10000, half_angle=np.radians(30)), 2*D10, rtol=1E-6)
    assert np.isclose(d.blip_detectivity(10000, photoconductive=True), D10/np.sqrt(2))
    assert d.blip_detectivity(3000) > 10*D10
    # Johnson-limited: R0A = 1E3 Ohm cm² at 77 K and 5 A/W
    assert np.isclose(d.r0a_detectivity(5, 1E3), 5*np.sqrt(1E3/(4*d.K_B*77)))
    assert np.isclose(d.quantum_limit_photons(1E-9), 10.0, atol=0.02)


def test_dead_time():
    r, tau = 2E7, 22E-9
    for par in (False, True):
        m = d.dead_time_rate(r, tau, par)
        assert np.isclose(d.true_rate(m, tau, par), r)
    rng = np.random.default_rng(1)
    t = d.poisson_arrivals(r, 0.02, rng)
    for par in (False, True):
        det = d.spad_detections(t, tau, par)
        assert np.isclose(len(det)/0.02, d.dead_time_rate(r, tau, par), rtol=0.01)
    # dead time makes the counts sub-Poissonian
    det = d.spad_detections(t, tau)
    counts = np.histogram(det, bins=2000, range=(0, 0.02))[0]
    assert np.isclose(d.mandel_q(counts) + 1, d.dead_time_fano(r, tau), rtol=0.1)
    assert np.isclose(d.click_probability(1.0, 0.5), 1 - np.exp(-0.5))


def test_photon_number_distributions():
    n = np.arange(400)
    for pmf, Q, g in ((d.poisson_pmf(n, 5.0), 0, 1), (d.thermal_pmf(n, 5.0), 5, 2),
                      (d.thermal_pmf(n, 5.0, modes=4), 5/4, 1.25), (d.fock_pmf(n, 5), -1, 0.8)):
        m, var, q, g2 = d.pmf_moments(n, pmf)
        assert np.isclose(pmf.sum(), 1) and np.isclose(m, 5)
        assert np.isclose(q, Q, atol=1E-9) and np.isclose(g2, g)
    rng = np.random.default_rng(0)
    c = rng.poisson(20.0, 200000)
    assert abs(d.mandel_q(c)) < 0.02 and abs(d.g2(c) - 1) < 0.01
    # binomial thinning: Poisson stays Poisson, Q -> eta Q
    assert abs(d.mandel_q(d.thin(c, 0.3, rng))) < 0.02
    nf = np.full(200000, 20)
    assert np.isclose(d.mandel_q(d.thin(nf, 0.3, rng)), -0.3, atol=0.01)


def test_chaotic_light():
    tau_c, dt = 1.0, 0.05
    I = d.chaotic_intensity(2**20, dt, tau_c, rng=3)
    assert np.isclose(I.mean(), 1) and np.isclose(np.mean(I**2), 2, rtol=0.03)
    # Mandel's coherence time int |g1|**2 = int (g2 - 1)
    Ic = I - 1
    ac = np.real(np.fft.ifft(np.abs(np.fft.fft(Ic))**2))/len(I)
    k = int(10*tau_c/dt)
    assert np.isclose((ac[0] + 2*np.sum(ac[1:k]))*dt, tau_c, rtol=0.05)
    for T in (0.2, 2.0, 10.0):
        counts = d.photon_counts(I, dt, rate=5.0/T, window=T, rng=4)
        assert np.isclose(d.mandel_q(counts), d.mandel_q_chaotic(5.0, T, tau_c), rtol=0.1)
    assert np.isclose(d.mandel_q_chaotic(3.0, 1E-4, 1), 3.0, rtol=1E-6)
    assert np.isclose(d.mandel_q_chaotic(3.0, 1E4, 1), 3E-4, rtol=1E-3)


def test_beam_splitter():
    # Hong-Ou-Mandel: |1,1> -> (|2,0> + |0,2>)/sqrt(2)
    assert np.allclose(d.beam_splitter_fock(1, 1), [0.5, 0, 0.5])
    # general T: coincidences (T - R)**2
    T = 0.3
    assert np.isclose(d.beam_splitter_fock(1, 1, T)[1], (T - (1 - T))**2)
    # |N, 0> splits binomially
    from scipy.stats import binom
    assert np.allclose(d.beam_splitter_fock(5, 0, T), binom.pmf(np.arange(6), 5, T))
    # |2,2> at 50:50: no |1,3> or |3,1>, 3/8, 1/4, 3/8
    assert np.allclose(d.beam_splitter_fock(2, 2), [3/8, 0, 1/4, 0, 3/8])
    # unitarity
    for n1, n2 in ((3, 2), (4, 4)):
        assert np.isclose(d.beam_splitter_fock(n1, n2, 0.37).sum(), 1)
    # HBT: g2 of the input from the cross correlation of binomially split counts
    rng = np.random.default_rng(5)
    for counts, g in ((rng.poisson(2.0, 400000), 1.0), (rng.geometric(1/3, 400000) - 1, 2.0)):
        n3 = d.thin(counts, 0.5, rng)
        assert np.isclose(d.cross_g2(n3, counts - n3), g, rtol=0.02)


def test_hom():
    bw = 3E8/810E-9**2*1E-9   # 1 nm at 810 nm
    assert np.isclose(d.hom_coincidence(0, bw), 0)
    assert np.isclose(d.hom_coincidence(1, bw), 1)
    assert np.isclose(d.hom_coincidence(0, bw, classical=True), 0.5)
    assert np.isclose(d.hom_coincidence(0, bw, overlap=0.9), 0.1)
    # dip at tau = 0 equals the coincidences of beam_splitter_fock relative to T**2 + R**2
    T = 0.3
    assert np.isclose(d.hom_coincidence(0, bw, T)*(T**2 + (1 - T)**2), d.beam_splitter_fock(1, 1, T)[1])
    w = d.hom_dip_fwhm(bw)
    assert np.isclose(d.hom_coincidence(w/2, bw), 0.5)
    assert 1.2E-12 < w < 1.5E-12
