#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.ellipsometry."""

import numpy as np
import pytest
from scipy.integrate import quad

from pyoptics import ellipsometry as el
from pyoptics.interfaces import r_s as hecht_r_s, r_p as hecht_r_p


def test_bare_silicon_and_native_oxide():
    # Nebraska convention: Delta just below 180 degrees for c-Si at 633 nm and 70 deg,
    # and a 2 nm oxide lowers it by several degrees
    psi0, delta0 = el.psi_delta(0.6328, 70.0, [], el.silicon_index)
    psi2, delta2 = el.psi_delta(0.6328, 70.0, [(el.sio2_index, 0.002)], el.silicon_index)
    assert 10 < psi0 < 11.2 and 178 < delta0 < 180
    assert 4 < delta0 - delta2 < 8


def test_rho_against_interfaces():
    # a bare interface: rho = r_p/r_s with Born & Wolf's r_p, conjugated (exp(+i omega t))
    n2 = 1.5 + 0.2j
    th = np.radians(55.0)
    r = el.rho(0.5, 55.0, [], n2)
    ref = np.conj(hecht_r_p(th, 1.0, n2)/hecht_r_s(th, 1.0, n2))
    assert r == pytest.approx(ref, rel=1e-10)


def test_brewster_angle():
    n = 1.5
    psi, delta = el.psi_delta(0.5, el.brewster_angle(n) + np.array([-1.0, 1.0]), [], n)
    assert psi[0] < 2 and psi[1] < 2
    assert delta[0] == pytest.approx(180.0) and delta[1] == pytest.approx(0.0, abs=1e-9)


def test_pseudo_dielectric_inverts_bare_substrate():
    for wl in (0.4, 0.6328, 0.9):
        r = el.rho(wl, 65.0, [], el.silicon_index)
        assert el.pseudo_dielectric(r, 65.0) == pytest.approx(el.silicon_index(wl)**2, rel=1e-10)


def test_film_period():
    n, ang, wl = 1.46, 70.0, 0.6328
    D = el.film_period(wl, n, ang)
    for d in (0.05, 0.2):
        a = el.psi_delta(wl, ang, [(n, d)], el.silicon_index)
        b = el.psi_delta(wl, ang, [(n, d + D)], el.silicon_index)
        assert a[0] == pytest.approx(b[0], abs=1e-9)
        assert a[1] == pytest.approx(b[1], abs=1e-9)


def test_tauc_lorentz_kramers_kronig():
    A, E0, C, Eg, einf = 122.0, 3.45, 2.54, 1.2, 1.15

    def e2(x):
        return el.tauc_lorentz(x, A, E0, C, Eg, 0.0).imag

    for E in (0.6, 1.6, 3.0, 4.5):
        main = quad(lambda x: 2/np.pi*x*e2(x)/(x + E), Eg, 80, weight="cauchy", wvar=E,
                    limit=400)[0]
        tail = quad(lambda x: 2/np.pi*x*e2(x)/(x**2 - E**2), 80, np.inf)[0]
        assert el.tauc_lorentz(E, A, E0, C, Eg, einf).real == pytest.approx(einf + main + tail,
                                                                            rel=1e-7)
    assert el.tauc_lorentz(1.0, A, E0, C, Eg).imag == 0.0


def test_lorentz_drude_signs():
    E = np.linspace(0.5, 5, 50)
    assert np.all(el.lorentz(E, 2.0, 3.0, 0.5).imag > 0)
    assert np.all(el.drude(E, 9.0, 0.1).imag > 0)
    # peak of eps_2 of the Lorentz oscillator is A at E0
    assert el.lorentz(3.0, 2.0, 3.0, 0.5).imag == pytest.approx(2.0)


def test_bruggeman():
    ea, eb = 2.1, 15.0 + 0.2j
    assert el.bruggeman(ea, eb, 0.0) == pytest.approx(ea)
    assert el.bruggeman(ea, eb, 1.0) == pytest.approx(eb)
    e = el.bruggeman(ea, eb, 0.3)
    fa, fb = 0.7, 0.3
    assert abs(fa*(ea - e)/(ea + 2*e) + fb*(eb - e)/(eb + 2*e)) < 1e-12
    assert e.imag >= 0 and ea < e.real < eb.real
    assert el.bruggeman(ea, ea, 0.5) == pytest.approx(ea)


def test_index_helpers():
    assert el.sio2_index(0.5876).real == pytest.approx(1.4585, abs=2e-4)
    assert el.silicon_index(0.6).real == pytest.approx(3.94)
    with pytest.raises(ValueError):
        el.silicon_index(0.3)
    assert el.cauchy(1.0, 1.5, 0.01, 0.0, 0.001) == pytest.approx(1.51 + 0.001j)
    N = el.index_from_eps(-10 + 1j)
    assert N.imag > 0 and N**2 == pytest.approx(-10 + 1j)


@pytest.mark.parametrize("psi, delta", [(20.0, 250.0), (35.0, 10.0), (10.0, 175.0)])
def test_rotating_compensator_recovers_psi_delta(psi, delta):
    N, C, S = el.ncs_from_psi_delta(psi, delta)
    comp = np.arange(0, 180, 7.5)
    sig = el.ellipsometer_signal(N, C, S, 45.0, 45.0, comp, retardance=90.0)
    p, d, P = el.psi_delta_from_ncs(*el.fit_ncs(sig, 45.0, 45.0, comp, retardance=90.0))
    assert p == pytest.approx(psi) and d == pytest.approx(delta) and P == pytest.approx(1.0)


def test_rotating_analyzer_loses_sign_of_delta():
    psi, delta = 20.0, 250.0
    N, C, S = el.ncs_from_psi_delta(psi, delta)
    A = np.arange(0, 180, 5.0)
    sig = el.ellipsometer_signal(N, C, S, 45.0, A)
    n_, c_, s_ = el.fit_ncs(sig, 45.0, A)
    assert s_ is None
    p, d, _ = el.psi_delta_from_ncs(n_, c_)
    assert p == pytest.approx(psi) and d == pytest.approx(360.0 - delta)
    # Fourier coefficients
    a, b = el.rotating_analyzer_coefficients(psi, delta, 45.0)
    X = np.c_[np.ones_like(A), np.cos(2*np.radians(A)), np.sin(2*np.radians(A))]
    c = np.linalg.lstsq(X, sig, rcond=None)[0]
    assert c[1]/c[0] == pytest.approx(a) and c[2]/c[0] == pytest.approx(b)


def test_depolarization():
    # incoherent mix of two samples: degree of polarization below 1
    n1 = np.array(el.ncs_from_psi_delta(20.0, 100.0))
    n2 = np.array(el.ncs_from_psi_delta(20.0, 140.0))
    N, C, S = 0.5*(n1 + n2)
    comp = np.arange(0, 180, 10.0)
    sig = el.ellipsometer_signal(N, C, S, 45.0, 45.0, comp)
    assert el.psi_delta_from_ncs(*el.fit_ncs(sig, 45.0, 45.0, comp))[2] < 0.99


def test_fit_film_thickness_and_cauchy():
    wl = np.linspace(0.4, 1.0, 61)
    angles = np.array([65.0, 70.0, 75.0])
    W, Aa = np.meshgrid(wl, angles, indexing="ij")
    true = (1.45, 0.0040, 0.250)

    def model(p):
        return el.psi_delta(W, Aa, [(lambda w: el.cauchy(w, p[0], p[1]), p[2])],
                            el.silicon_index)

    rng = np.random.default_rng(1)
    psi, delta = model(true)
    psi = psi + 0.02*rng.standard_normal(psi.shape)
    delta = delta + 0.04*rng.standard_normal(delta.shape)
    fit = el.fit_psi_delta(model, (1.5, 0.0, 0.2), psi, delta, sigma_psi=0.02,
                           sigma_delta=0.04)
    assert fit.params == pytest.approx(true, rel=2e-3)
    assert 0.7 < fit.mse < 1.3
    assert np.all(np.abs(fit.params - true) < 5*fit.std)
    assert fit.correlation.shape == (3, 3)
    assert "MSE" in repr(fit)
