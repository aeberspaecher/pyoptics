#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.electrooptics, pyoptics.liquid_crystals and
pyoptics.acoustooptics."""

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import jv

from pyoptics import acoustooptics as ao
from pyoptics import crystals as cr
from pyoptics import electrooptics as eo
from pyoptics import fibrecomm as fc
from pyoptics import liquid_crystals as lc
from pyoptics import spectroscopy as spc


# --------------------------------------------------------------------------
# electro-optics

def test_linbo3_indices():
    n_o, n_e = eo.linbo3_indices(1550.0)
    assert n_o == pytest.approx(2.211, abs=1e-3) and n_e == pytest.approx(2.138, abs=1e-3)


def test_kdp_index_ellipsoid_rotates_by_45_degrees():
    # field along z: principal axes at 45 degrees, n = n_o -+ n_o**3 r63 E/2
    k = eo.KDP
    eps0 = cr.uniaxial_eps(k["n_o"], k["n_e"])
    E = 1e6
    eps = eo.pockels_eps(eps0, eo.r_42m(k["r41"]*1e-12, k["r63"]*1e-12), (0.0, 0.0, E))
    n, D, _, _ = cr.wave_indices(eps, (0.0, 0.0, 1.0))
    dn = k["n_o"]**3*k["r63"]*1e-12*E/2
    assert n.real == pytest.approx([k["n_o"] - dn, k["n_o"] + dn], rel=1e-9, abs=1e-12)
    assert abs(abs(D[0, 0]) - abs(D[0, 1])) < 1e-6          # along a diagonal
    # the retardance of a longitudinal cell reaches pi at V_pi
    lam = 633e-9
    V_pi = eo.half_wave_voltage(lam, k["n_o"], k["r63"]*1e-12)
    assert 8.5e3 < V_pi < 9.5e3


def test_linbo3_transverse():
    k = eo.LINBO3
    eps0 = cr.uniaxial_eps(k["n_o"], k["n_e"])
    r = eo.r_3m(k["r13"]*1e-12, k["r22"]*1e-12, k["r33"]*1e-12, k["r51"]*1e-12)
    E = 1e6
    eps = eo.pockels_eps(eps0, r, (0.0, 0.0, E))
    # light along y: indices along x (ordinary) and z (extraordinary)
    n = cr.wave_indices(eps, (0.0, 1.0, 0.0))[0].real
    assert n[0] == pytest.approx(k["n_e"] - k["n_e"]**3*k["r33"]*1e-12*E/2, rel=1e-8)
    assert n[1] == pytest.approx(k["n_o"] - k["n_o"]**3*k["r13"]*1e-12*E/2, rel=1e-8)


def test_sidebands_and_kerr():
    n, P = eo.phase_modulation_sidebands(1.2, 15)
    assert P.sum() == pytest.approx(1.0) and P[n == 0][0] == pytest.approx(jv(0, 1.2)**2)
    E = 1/np.sqrt(2*eo.NITROBENZENE_KERR*0.05)
    assert eo.kerr_retardance(E, 0.05) == pytest.approx(np.pi)


def test_mzm_transfer_extinction_and_chirp():
    V = np.linspace(-4.0, 4.0, 801)
    P = eo.mzm_power(V, 4.0, bias=0.0)
    assert P.max() == pytest.approx(1.0) and P[np.argmin(np.abs(V - 4.0))] == pytest.approx(0.0, abs=1e-12)
    # unequal splitters: extinction 1/(2a - 1)**2
    E_null = eo.mzm_field(0.0, 0.0, 4.0, bias=np.pi, split=0.55)
    E_peak = eo.mzm_field(0.0, 0.0, 4.0, bias=0.0, split=0.55)
    assert abs(E_peak/E_null)**2 == pytest.approx(eo.mzm_extinction(0.55))
    assert eo.mzm_chirp(1.0, -1.0) == 0.0 and eo.mzm_chirp(1.0, 0.0) == 1.0
    # small signal: d phi/dt = (alpha/2) d ln P/dt, blue rising edges for alpha > 0
    bits = fc.prbs(7)
    for alpha in (-0.7, 0.0, 1.0):
        t, A = eo.mzm_nrz(bits, 10.0, alpha=alpha, swing=0.2*4.0)
        P = np.abs(A)**2
        phi = np.unwrap(np.angle(A))
        dlnP = np.gradient(np.log(P), t)
        sel = np.abs(dlnP) > 0.3*np.abs(dlnP).max()
        ratio = -np.gradient(phi, t)[sel]/dlnP[sel]       # exp(-i omega t): frequency -dphi/dt
        assert np.median(ratio) == pytest.approx(alpha/2, abs=0.02)


def test_mzm_nrz_levels():
    t, A = eo.mzm_nrz(fc.prbs(7), 10.0, p_in=2.0, split=0.5)
    P = np.abs(A)**2
    assert P.max() == pytest.approx(2.0, rel=1e-6) and P.min() < 1e-6


def test_travelling_wave_bandwidth():
    L = 0.01
    f3 = eo.bandwidth_3db(L)
    assert f3 == pytest.approx(1.392*eo.C0/(np.pi*L*(4.2 - 2.18))/1e9, rel=2e-3)
    assert eo.bandwidth_3db(L, n_m=2.18) == np.inf
    assert eo.bandwidth_3db(L, alpha0_db=0.5) < f3


# --------------------------------------------------------------------------
# liquid crystals

def test_threshold_and_energy_gradient():
    assert lc.threshold_voltage() == pytest.approx(0.947, abs=0.002)
    N = 16
    x = np.random.default_rng(0).uniform(0, 1, 2*(N - 1))
    args = (N, 0.05, 0.05, 0.0, 1.5, 1.0, 0.6, 1.5, 5.2, 13.8, 40.0, True)
    err = check_grad(lambda x: lc._energy(x, *args)[0], lambda x: lc._energy(x, *args)[1], x)
    assert err < 1e-4


def test_freedericksz_transition():
    z, th, _ = lc.director_profile(0.9, pretilt=0.0)
    assert np.abs(th).max() < 1e-4
    z, th, _ = lc.director_profile(1.05, pretilt=0.0)
    assert np.degrees(th.max()) > 5
    tilts = [lc.director_profile(V)[1].max() for V in (1.5, 3.0, 6.0)]
    assert tilts[0] < tilts[1] < tilts[2] < np.pi/2
    # just above threshold the profile is close to sin(pi z)
    z, th, _ = lc.director_profile(1.0, pretilt=0.0, n_nodes=61)
    shape = th/th.max()
    assert np.abs(shape - np.sin(np.pi*z)).max() < 0.03


def test_retardance_and_tn_cell():
    n_o, n_e = lc.e7_indices(550.0)
    z, th, _ = lc.director_profile(0.0)
    d0 = lc.retardance(th, 5.0, 0.55, n_o, n_e)
    assert d0 == pytest.approx(2*np.pi*(n_e - n_o)*5.0/0.55, rel=2e-3)
    V = np.linspace(0, 6, 13)
    ds = [lc.retardance(lc.director_profile(v)[1], 5.0, 0.55, n_o, n_e) for v in V]
    assert np.all(np.diff(ds) < 0)
    assert lc.voltage_for_retardance(ds[6], V, ds) == pytest.approx(V[6])
    # a 90 degree TN cell at 0 V between parallel polarizers: Gooch-Tarry
    for d in (2.0, 3.0, 5.0):
        z, th, ph = lc.director_profile(0.0, pretilt=0.0, twist=np.pi/2)
        t, _ = cr.stack_jones(lc.cell_layers(th, ph, d, n_o, n_e), 0.55, n_in=1.5, n_out=1.5,
                              coherent=False)
        E = t @ np.array([1.0, 0.0])
        T = abs(E[0])**2/np.sum(np.abs(E)**2)
        assert T == pytest.approx(lc.gooch_tarry(lc.mauguin_parameter(d, n_e - n_o, 0.55)), abs=2e-3)


def test_response_times():
    on, off = lc.response_times(5e-6, 5.0)
    assert off == pytest.approx(0.23*25e-12/(np.pi**2*11.1e-12))
    assert on < off
    assert lc.response_times(5e-6, 0.5)[0] == np.inf


# --------------------------------------------------------------------------
# acousto-optics

def test_raman_nath_limit_is_bessel():
    m, E = ao.coupled_orders(2.5, 0.0, alpha=0.0)
    assert np.abs(np.abs(E)**2 - jv(m, 2.5)**2).max() < 1e-12
    assert np.sum(np.abs(E)**2) == pytest.approx(1.0)


def test_bragg_limit():
    m, E = ao.coupled_orders(np.pi, 400.0, alpha=1.0, n_orders=4)
    assert abs(E[m == 1][0])**2 == pytest.approx(1.0, abs=1e-3)
    assert ao.bragg_efficiency(np.pi) == pytest.approx(1.0)
    assert ao.bragg_efficiency(1.0) == pytest.approx(np.sin(0.5)**2)
    # with a mismatch the coupled waves and the two-wave formula agree for large Q
    for alpha in (0.98, 1.02):
        Q = 300.0
        m, E = ao.coupled_orders(1.0, Q, alpha=alpha, n_orders=4)
        mis = Q/2*(1 - alpha)/2          # Delta k L/2 of order +1
        assert abs(E[m == 1][0])**2 == pytest.approx(ao.bragg_efficiency(1.0, mis), abs=2e-3)


def test_acousto_optic_numbers():
    dn = ao.delta_n_from_power(1.0, ao.MATERIALS["TeO2 longitudinal"]["M2"], 0.01, 0.001)
    nu = ao.raman_nath_parameter(dn, 0.01, 633e-9)
    # textbook form sin**2(pi/(sqrt(2) lambda) sqrt(M2 L P/H))
    ref = np.sin(np.pi/(np.sqrt(2)*633e-9)*np.sqrt(34.5e-15*0.01*1.0/0.001))**2
    assert ao.bragg_efficiency(nu) == pytest.approx(ref)
    assert ao.aod_spots(0.01, 617.0, 50e6) == pytest.approx(0.01/617*50e6)
    n_o, n_e = ao.teo2_indices(633.0)
    assert n_o == pytest.approx(2.260, abs=2e-3) and n_e == pytest.approx(2.412, abs=2e-3)


def test_aotf():
    dn = lambda lam: np.subtract(*ao.teo2_indices(np.asarray(lam)*1e9)[::-1])
    f = ao.aotf_frequency(633e-9, dn(633e-9), 617.0, np.radians(20.0))
    assert 80e6 < f < 110e6
    lam = np.linspace(625e-9, 641e-9, 16001)
    T = ao.aotf_transmission(lam, 633e-9, 0.02, np.radians(20.0), dn, nu=0.05)
    fw = spc.fwhm(lam, T/T.max())
    assert fw == pytest.approx(ao.aotf_passband(633e-9, 0.02, np.radians(20.0), dn), rel=0.03)
