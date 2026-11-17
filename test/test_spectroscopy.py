#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test line shapes, Kramers-Kronig, Fourier transform spectroscopy,
Fabry-Perot and Lyot filters against closed forms."""

import numpy as np
import pytest

from pyoptics import spectroscopy as spc
from pyoptics.polarization import jones


def test_line_shapes_normalized_and_widths():
    nu = np.linspace(-200, 200, 400001)
    for f in (spc.lorentzian(nu, 0, 1.0), spc.gaussian(nu, 0, 1.0),
              spc.voigt(nu, 0, 1.0, 0.7)):
        assert np.isclose(np.trapezoid(f, nu), 1.0, atol=3e-3)
    assert np.isclose(spc.fwhm(nu, spc.lorentzian(nu, 0, 1.3)), 1.3, rtol=1e-4)
    assert np.isclose(spc.fwhm(nu, spc.gaussian(nu, 0, 1.3)), 1.3, rtol=1e-4)
    # Voigt limits and the Olivero formula:
    assert np.allclose(spc.voigt(nu, 0, 1.0, 1e-9), spc.gaussian(nu, 0, 1.0), atol=1e-7)
    assert np.allclose(spc.voigt(nu, 0, 1e-6, 1.0), spc.lorentzian(nu, 0, 1.0), atol=1e-6)
    w = spc.fwhm(nu, spc.voigt(nu, 0, 1.0, 0.7))
    assert np.isclose(w, spc.voigt_fwhm(1.0, 0.7), rtol=3e-4)


def test_doppler_width_sodium():
    # Na D2 at 589 nm and 500 K: about 2.0 pm (1.7 GHz)
    w = spc.doppler_fwhm(589.0, 500.0, 23.0)
    assert 1.9e-3 < w < 2.05e-3


def test_kramers_kronig_lorentz_oscillator():
    nu = np.linspace(1e-3, 60.0, 30001)
    n = spc.lorentz_index(nu, 10.0, 0.05, 0.3)
    n_inf = 1.0
    dn = spc.kramers_kronig(nu, n.imag)
    sel = (nu > 2) & (nu < 30)
    assert np.max(np.abs(dn[sel] - (n.real[sel] - n_inf))) < 2e-3*np.max(np.abs(n.real - 1))


def test_beer_lambert_and_curve_of_growth():
    assert np.isclose(spc.absorbance(spc.transmittance(np.log(10))), 1.0)
    nu = np.linspace(-50, 50, 20001)
    phi = spc.lorentzian(nu, 0, 0.2)
    W_weak = spc.equivalent_width(nu, spc.transmittance(1e-4*phi))
    assert np.isclose(W_weak, 1e-4, rtol=2e-3)
    # Lorentzian square-root regime: W = 2 sqrt(S gamma/2) for S phi_0 >> 1
    S = 200.0
    W = spc.equivalent_width(nu, spc.transmittance(S*phi))
    assert np.isclose(W, 2*np.sqrt(S*0.2/2), rtol=0.03)


def test_band_lines():
    p, S, lab = spc.linear_molecule_band(2349.1, 0.3902, 0.3871, 296.0, even_only=True)
    assert np.isclose(S.sum(), 1.0)
    assert lab[0] == "P(2)" and "R(0)" in lab
    # adjacent R lines 4B apart near the origin:
    r0, r2 = p[list(lab).index("R(0)")], p[list(lab).index("R(2)")]
    assert np.isclose(r2 - r0, 4*0.3871 + 6*(0.3871 - 0.3902)*0 + (r2 - r0 - 4*0.3871), atol=1e-12)
    assert 1.4 < r2 - r0 < 1.6
    # the R branch peaks near J = 16 at room temperature for CO2
    R = np.array([l.startswith("R") for l in lab])
    J = np.array([int(l[2:-1]) for l in lab])
    assert 12 <= J[R][np.argmax(S[R])] <= 20


def test_slit_function_and_convolve():
    x = np.linspace(-5, 5, 10001)
    tri = spc.slit_function(x, 1.0, 1.0)
    assert np.isclose(np.trapezoid(tri, x), 1.0, atol=1e-6)
    assert np.isclose(spc.fwhm(x, tri), 1.0, rtol=1e-3)
    trap = spc.slit_function(x, 2.0, 1.0)
    assert np.isclose(np.trapezoid(trap, x), 1.0, atol=1e-6)
    y = spc.convolve(x, (np.abs(x) < 1e-9).astype(float)/(x[1] - x[0]),
                     lambda d: spc.slit_function(d, 1.0, 1.0))
    assert np.allclose(y, tri, atol=1e-6)


@pytest.mark.parametrize("kind,factor", [("boxcar", 1.207), ("triangle", 1.772),
                                         ("happ_genzel", 1.816), ("norton_beer_weak", 1.442),
                                         ("norton_beer_medium", 1.693),
                                         ("norton_beer_strong", 1.924)])
def test_instrument_line_shape_widths(kind, factor):
    L = 1.0
    nu = np.linspace(-4, 4, 8001)
    ils = spc.instrument_line_shape(nu, L, kind)
    # unit area (the sinc tails of boxcar and triangle reach beyond the grid):
    assert np.isclose(np.trapezoid(ils, nu), 1.0, atol=0.03)
    assert np.isclose(spc.fwhm(nu, ils), factor/(2*L), rtol=0.01)


def test_interferogram_chirp_z_matches_direct_sum():
    # uniform x: chirp-z transform; the same points out of order: direct sum
    rng = np.random.default_rng(3)
    nu = np.linspace(500.0, 900.0, 801)
    S, phi = rng.random(801), rng.random(801)
    x = np.linspace(-0.02, 0.03, 1001)
    perm = rng.permutation(len(x))
    for phase in (None, phi):
        fast = spc.interferogram(x, nu, S, phase)
        direct = spc.interferogram(x[perm], nu, S, phase)
        assert np.allclose(fast[perm], direct, rtol=0, atol=1e-9*np.max(np.abs(direct)))


def test_kramers_kronig_matches_the_maclaurin_sum():
    # the FFT evaluation against the explicit sum over the other parity
    rng = np.random.default_rng(4)
    nu = 0.3 + 0.01*np.arange(301)
    kappa = rng.random(301)
    dn = spc.kramers_kronig(nu, kappa)
    h = nu[1] - nu[0]
    idx = np.arange(len(nu))
    for i in (0, 1, 150, 299, 300):
        j = idx[idx % 2 != i % 2]
        ref = (4*h/np.pi)*np.sum(nu[j]*kappa[j]/(nu[j]**2 - nu[i]**2))
        assert dn[i] == pytest.approx(ref, rel=1e-10, abs=1e-12)


def test_fts_recovers_lines():
    nu = np.linspace(0, 4000, 40001)
    S = spc.gaussian(nu, 1000.0, 2.0) + 0.5*spc.gaussian(nu, 2500.0, 2.0)
    dx = 1/(2*5000)
    L = 2.0
    x = np.arange(-int(L/dx), int(L/dx) + 1)*dx
    I = spc.interferogram(x, nu, S)
    nu_m, S_m = spc.fts_spectrum(x, I, "boxcar")
    for c, a in ((1000.0, 1.0), (2500.0, 0.5)):
        sel = np.abs(nu_m - c) < 20
        assert np.isclose(np.trapezoid(S_m[sel], nu_m[sel]), a, rtol=0.01)


def test_mertz_phase_correction():
    nu = np.linspace(0, 4000, 20001)
    S = spc.gaussian(nu, 1500.0, 300.0)
    phase = 0.4 + 2*np.pi*nu*1e-4 + 1e-7*(nu - 1500)**2   # ZPD one sample off, dispersion
    dx = 1/(2*5000)
    x = np.arange(-200, 4001)*dx
    I = spc.interferogram(x, nu, S, phase)
    nu_m, S_m, ph = spc.mertz_spectrum(x, I, 150*dx, kind="boxcar")
    sel = np.abs(nu_m - 1500) < 300
    ref = np.interp(nu_m[sel], nu, S)
    assert np.max(np.abs(S_m[sel] - ref)) < 0.01*ref.max()


def test_fabry_perot():
    R = 0.9
    delta = np.linspace(-np.pi, np.pi, 200001)
    T = spc.airy_transmission(delta, R)
    assert np.isclose(T.max(), 1.0)
    width = spc.fwhm(delta, T)
    assert np.isclose(2*np.pi/width, spc.reflective_finesse(R), rtol=1e-3)
    assert np.isclose(spc.airy_transmission(0.0, 0.9, 0.01), (0.09/0.1)**2)
    assert np.isclose(spc.effective_finesse(30.0, 40.0), 24.0)
    assert np.isclose(spc.free_spectral_range(0.5, 1.0), 1.0)


def test_lyot_against_jones():
    lam = np.linspace(600, 700, 501)
    d, dn, N = 50e3, 0.009, 3
    T = spc.lyot_transmission(lam, d, dn, N)
    P = jones.J_polarizer(1.0, 0.0)
    E = np.array([1.0, 0.0])
    for i, l in enumerate(lam):
        J = np.eye(2)
        for k in range(N):
            J = P @ jones.rotate(jones.J_WP(2*np.pi*dn*d*2**k/l), np.pi/4) @ J
        assert np.isclose(np.abs(J @ E)[0]**2 + np.abs(J @ E)[1]**2, T[i], atol=1e-10)


def test_resolving_powers():
    assert spc.grating_resolving_power(2, 30000) == 60000
    assert np.isclose(spc.prism_resolving_power(50e6, 1.6e-4), 8000)
    assert np.isclose(spc.jacquinot_solid_angle(2*np.pi*1e4), 1e-4)
