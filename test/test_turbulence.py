#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.turbulence against the classical Kolmogorov results.

References
----------
R. J. Noll, "Zernike polynomials and atmospheric turbulence", J. Opt. Soc. Am.
66, 207-211 (1976).

F. Roddier, "The effects of atmospheric turbulence in optical astronomy",
Progress in Optics 19, 281-376 (1981).
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0

from pyoptics import turbulence


def test_constants():
    assert np.isclose(turbulence.KOLMOGOROV_SF, 6.88, atol=0.01)
    assert np.isclose(turbulence.KOLMOGOROV_PSD, 0.023, atol=0.0002)


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_structure_function_from_psd():
    # D(r) = 2 int 2 pi f Phi(f) (1 - J0(2 pi f r)) df, Kolmogorov and von Karman
    r0 = 0.1
    for L0 in (np.inf, 5.0):
        for r in (0.03, 0.3, 3.0):
            def g(f):
                return 4*np.pi*f*turbulence.phase_psd(f, r0, L0)*(1 - j0(2*np.pi*f*r))
            num = sum(quad(g, a, b, limit=400)[0] for a, b in
                      [(0, 1/r), (1/r, 10/r), (10/r, 1000/r)]) + quad(g, 1000/r, np.inf, limit=400)[0]
            assert np.isclose(num, turbulence.structure_function_theory(r, r0, L0), rtol=2e-3)


def test_von_karman_limits():
    r0, L0 = 0.1, 20.0
    # the leading correction is relative (r/L0)**(1/3): 12 % at r = L0/2000
    small = turbulence.structure_function_theory(0.01, r0, L0)
    assert 0.85 < small/turbulence.structure_function_theory(0.01, r0) < 0.9
    tiny = turbulence.structure_function_theory(0.01, r0, 1e5)
    assert np.isclose(tiny, turbulence.structure_function_theory(0.01, r0), rtol=0.01)
    sat = turbulence.structure_function_theory(1e4, r0, L0)
    assert np.isclose(sat, 0.1726*(L0/r0)**(5/3), rtol=1e-3)


def test_noll_numbers():
    # Noll's table (with his rounded constant: 1.0299, 0.582, 0.134, 0.111, ...)
    assert np.isclose(turbulence.noll_variance(1), 0.449, atol=0.002)
    delta = turbulence.noll_residual([1, 2, 3, 4, 6, 10, 21])
    noll = [1.0299, 0.582, 0.134, 0.111, 0.0648, 0.0401, 0.0208]
    assert np.allclose(delta, noll, rtol=0.03)
    J = np.array([50, 200, 1000])        # Noll's asymptotic fit, good to about 10 %
    assert np.allclose(turbulence.noll_residual(J), 0.2944*J**(-np.sqrt(3)/2), rtol=0.1)
    assert np.isclose(turbulence.noll_residual(3, 10.0), 0.1347*10**(5/3), rtol=0.01)


def test_screen_structure_function():
    r0, dx = 0.1, 0.01
    screens = [turbulence.phase_screen(256, dx, r0, rng=i) for i in range(30)]
    r, D = turbulence.structure_function(screens, dx, 60)
    ratio = D/turbulence.structure_function_theory(r, r0)
    assert np.all(np.abs(ratio[4:] - 1) < 0.12)
    # without subharmonics the large scales are missing
    plain = [turbulence.phase_screen(256, dx, r0, subharmonics=0, rng=i) for i in range(30)]
    r, Dp = turbulence.structure_function(plain, dx, 60)
    assert Dp[-1] < 0.8*D[-1]


def test_screen_shape_and_seed():
    a = turbulence.phase_screen(64, 0.01, 0.1, rng=3, M=128)
    b = turbulence.phase_screen(64, 0.01, 0.1, rng=3, M=128)
    assert a.shape == (64, 128) and np.array_equal(a, b)
    assert abs(a.mean()) < 1e-10


def test_layers_and_integrals():
    wl = 0.5e-6
    h, c = turbulence.layers_from_profile(turbulence.hufnagel_valley,
                                          np.array([0, 1000, 5000, 10000, 20000]))
    assert np.all(np.diff(h) > 0)
    # HV 5/7: r0 about 5 cm and theta0 about 7 urad at 500 nm
    assert np.isclose(turbulence.fried_parameter(c, wl), 0.05, rtol=0.05)
    assert 5e-6 < turbulence.isoplanatic_angle(h, c, wl) < 9e-6
    # r0 of layers combine with the power -5/3
    r0s = np.array([turbulence.fried_parameter(ci, wl) for ci in c])
    assert np.isclose(np.sum(r0s**(-5/3))**(-3/5), turbulence.fried_parameter(c, wl))
    assert np.isclose(turbulence.r0_at_wavelength(0.1, 0.5e-6, 2.2e-6), 0.1*4.4**1.2)
    # tau0 = 0.314 r0 / v for a single layer
    tau = turbulence.coherence_time([1e-13], [10.0], wl)
    assert np.isclose(tau, 0.314*turbulence.fried_parameter(1e-13, wl)/10, rtol=1e-3)


def test_scintillation_index():
    wl = 0.5e-6
    h, c = np.array([2000., 10000.]), np.array([2e-13, 1e-13])
    point = turbulence.scintillation_index(h, c, wl)
    assert np.isclose(point, 19.12*wl**(-7/6)*np.sum(c*h**(5/6)))
    # the numerical integral agrees with the closed form
    assert np.isclose(turbulence.scintillation_index(h, c, wl, f_max=1e5), point, rtol=0.01)
    # aperture averaging: monotonic, D**(-7/3) for large apertures
    s = [turbulence.scintillation_index(h, c, wl, D=D) for D in (0.01, 0.1, 1.0, 2.0)]
    assert np.all(np.diff(s) < 0)
    assert np.isclose(s[3]/s[2], 2**(-7/3), rtol=0.1)


def test_split_step_scintillation():
    # one thin layer: the simulated scintillation index matches the Rytov integral
    wl, N, dx, h = 0.5e-6, 512, 4e-3, 5000.0
    x = (np.arange(N) - N//2)*dx
    c = 3e-13
    s2 = []
    for i in range(4):
        phi = turbulence.phase_screen(N, dx, turbulence.fried_parameter(c, wl), subharmonics=0,
                                      rng=i)
        u = turbulence.split_step(np.ones((N, N)), x, x, [phi], [h], wl)
        I = np.abs(u)**2
        s2.append(I.var()/I.mean()**2)
    theory = turbulence.scintillation_index([h], [c], wl, f_max=1/(2*dx))
    assert np.isclose(np.mean(s2), theory, rtol=0.15)


def test_atmospheric_otf_and_seeing():
    r0, wl = 0.1, 0.5e-6
    nu = r0/wl                                  # lambda nu = r0
    assert np.isclose(turbulence.atmospheric_otf(nu, r0, wl), np.exp(-3.44), rtol=0.01)
    assert turbulence.atmospheric_otf(nu, r0, wl, D=1.0) > turbulence.atmospheric_otf(nu, r0, wl)
    assert np.isclose(turbulence.seeing_fwhm(r0, wl), 0.976*wl/r0)
    assert turbulence.seeing_fwhm(r0, wl, L0=25) < turbulence.seeing_fwhm(r0, wl)
    assert np.isclose(turbulence.lucky_probability(5.0), 5.6*np.exp(-0.1557*25))


def test_frozen_flow():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((64, 64))
    flow = turbulence.FrozenFlow([s], 1.0, [(2.0, -1.0)], (16, 16), origins=[(10.0, 20.0)])
    # integer shifts reproduce the samples exactly
    assert np.allclose(flow(0.0), s[20:36, 10:26])
    assert np.allclose(flow(3.0), s[17:33, 16:32])
    # two layers add
    flow2 = turbulence.FrozenFlow([s, s], 1.0, [(0, 0), (0, 0)], (16, 16),
                                  origins=[(0.0, 0.0), (0.0, 0.0)])
    assert np.allclose(flow2(1.0), 2*s[:16, :16])


@pytest.mark.parametrize("n", [1, 2, 5])
def test_noll_variance_matches_integral(n):
    from scipy.special import jv
    R = 0.5
    f = lambda x: 2*np.pi*x*turbulence.phase_psd(x, 1.0)*(n + 1)*(jv(n + 1, 2*np.pi*x*R)/(np.pi*x*R))**2
    num = quad(f, 0, 5, limit=500)[0] + quad(f, 5, np.inf, limit=500)[0]
    assert np.isclose(num, turbulence.noll_variance(n), rtol=1e-4)
