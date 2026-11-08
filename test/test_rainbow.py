#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.rainbow, analytical.mie_amplitudes(), the water and air
indices of pyoptics.materials and pyoptics.colour.

References
----------
H. C. van de Hulst, "Light Scattering by Small Particles", Wiley (1957).

C. F. Bohren and D. R. Huffman, "Absorption and Scattering of Light by
Small Particles", Wiley (1983).

M. Daimon and A. Masumura, Appl. Opt. 46, 3811-3820 (2007).
"""

import numpy as np
import pytest

from pyoptics import colour, rainbow, raytrace
from pyoptics.analytical import mie_amplitudes, mie_efficiencies
from pyoptics.materials import air_index, water_index
from pyoptics.raytrace import Rays


DEG = np.pi/180


def test_water_and_air_index():
    # textbook n_D of water at 20 deg C relative to air, and its dispersion
    n_D = water_index(589.29)/air_index(589.29)
    assert abs(n_D - 1.33299) < 5e-5
    assert abs(air_index(589.29) - 1.000277) < 2e-6
    assert water_index(404.66) > water_index(486.13) > water_index(656.27)
    assert abs((water_index(486.13) - water_index(656.27)) - 0.00598) < 2e-4


def test_descartes_angles():
    n = 4/3
    _, _, th1 = rainbow.descartes(n, 1)
    _, _, th2 = rainbow.descartes(n, 2)
    assert abs(180 - th1/DEG - 42.03) < 0.02
    assert abs(180 - th2/DEG - 50.96) < 0.02
    with pytest.raises(ValueError):
        rainbow.descartes(2.5, 1)


@pytest.mark.parametrize("k", [-1, 0, 1, 2, 3])
def test_trace_matches_deviation_formula(k):
    n = 1.334
    b = np.linspace(0.01, 0.99, 50)
    tr = rainbow.trace_drop(0*b, b, n, k)
    assert np.allclose(tr.theta, rainbow.scattering_angle(np.arcsin(b), n, k), atol=1e-11)
    assert tr.points.shape == (k + 4 if k >= 0 else 3, 50, 3)
    # all intersections on the sphere
    assert np.allclose(np.linalg.norm(tr.points[1:-1], axis=-1), 1.0)


def test_rainbow_ray_is_extremum_of_traced_rays():
    n = 1.3343
    b = np.linspace(0.5, 0.999, 200001)
    for k in (1, 2):
        th = rainbow.trace_drop(0*b, b, n, k).theta
        ext = th.min() if k == 1 else th.max()
        assert abs(ext - rainbow.descartes(n, k)[2]) < 1e-9


def test_envelope_theorem():
    n, dn = 1.334, 1e-6
    for k in (1, 2):
        _, r_c, _ = rainbow.descartes(n, k)
        fd = abs(rainbow.descartes(n + dn, k)[2] - rainbow.descartes(n - dn, k)[2])/(2*dn)
        assert np.isclose(fd, 2*(k + 1)*np.tan(r_c)/n, rtol=1e-6)


def test_airy_parameters_against_traced_rays():
    n = 1.3343
    th_c, b_c, th2 = rainbow.airy_parameters(n, 1, 2.0)
    h = 1e-4
    b = b_c + np.array([-h, 0, h])
    th = rainbow.trace_drop(0*b, b, n, 1, radius=2.0).theta
    assert np.isclose((th[0] - 2*th[1] + th[2])/h**2, th2, rtol=1e-4)
    assert th2 > 0 and rainbow.airy_parameters(n, 2)[2] < 0


def test_geometric_cross_section_monte_carlo_and_energy():
    n = 1.334
    rng = np.random.default_rng(0)
    N = 400_000
    b = np.sqrt(rng.uniform(0, 1, N))
    edges = np.linspace(100, 170, 36)*DEG
    centres = (edges[:-1] + edges[1:])/2
    d_omega = 2*np.pi*(np.cos(edges[:-1]) - np.cos(edges[1:]))
    tr = rainbow.trace_drop(0*b, b, n, -1)
    hist, _ = np.histogram(tr.theta, edges, weights=tr.eps_s)
    s_s, _ = rainbow.geometric_cross_section(n, -1, theta=centres)
    assert np.allclose(hist*np.pi/N/d_omega, s_s, rtol=0.05)
    # the cross sections of all orders integrate to (almost) the geometric cross section pi
    theta = np.linspace(1e-4, np.pi - 1e-4, 200001)
    total = sum(sum(rainbow.geometric_cross_section(n, k, theta=theta))/2 for k in range(-1, 12))
    integral = np.trapezoid(total*2*np.pi*np.sin(theta), theta)
    assert abs(integral/np.pi - 1) < 0.01


@pytest.mark.parametrize("use_numba", [True, False])
def test_drop_system_matches_direct_trace(use_numba):
    old = raytrace.use_numba
    raytrace.use_numba = use_numba
    try:
        for k in (0, 1, 2):
            for i_band in (15*DEG, 55*DEG, 80*DEG):
                i = i_band + np.linspace(-5, 5, 9)*DEG
                b = np.sin(i)
                system = rainbow.drop_system(1.0, water_index, k, i_band)
                rays = Rays(np.stack([0*b, b, np.full_like(b, -2.0)], -1),
                            np.tile([0, 0, 1.0], (len(b), 1)), 0*b)
                bundle = system.trace(rays, clip=False)
                direct = rainbow.trace_drop(0*b, b, water_index(550.0), k)
                assert np.all(bundle.valid)
                assert np.allclose(bundle.directions[-2], direct.directions, atol=1e-10)
    finally:
        raytrace.use_numba = old


def test_ray_starting_on_far_branch_of_sphere():
    # a ray that starts exactly on the far hemisphere of a sphere surface must
    # find the vertex branch (it used to give NaN for t = 0)
    s = raytrace.Surface(1.0)
    P = np.array([[0.0, 0.6, 1.8]])      # on the sphere (centre at z = 1), far branch
    d = np.array([[0.0, 0.0, -1.0]])
    t = s.intersect(P, d)
    Q = P + t[:, None]*d
    assert np.isfinite(t[0]) and t[0] > 0
    assert np.isclose(Q[0, 2], s.sag(0.36))


def test_mie_amplitudes_optical_theorem_and_symmetry():
    for a, n in ((0.3e-3, 1.33), (2e-3, 1.5 + 0.01j)):
        wl = 0.5e-3
        x = 2*np.pi*a/wl
        S1, S2 = mie_amplitudes(np.array([0.0, np.pi]), a, n, wl)
        q_ext, q_sca = mie_efficiencies(a, n, wl)
        assert np.isclose(S1[0], S2[0])
        assert np.isclose(S1[1], -S2[1])
        assert np.isclose(4/x**2*S1[0].real, q_ext, rtol=1e-8)
        # scattering efficiency from the integral of the cross section
        theta = np.linspace(0, np.pi, 20001)
        S1, S2 = mie_amplitudes(theta, a, n, wl)
        I = (np.abs(S1)**2 + np.abs(S2)**2)/2
        q = np.trapezoid(I*2*np.pi*np.sin(theta), theta)/x**2/np.pi
        assert np.isclose(q, q_sca, rtol=1e-4)


def test_airy_theory_against_mie():
    # a = 0.25 mm: the main maximum and the first supernumeraries of Airy's
    # theory against the Mie series (s polarization), ripple smoothed
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    n, wl, a = 1.3343, 550e-6, 0.25
    k0 = 2*np.pi/wl
    theta = np.linspace(137.5, 141.5, 1500)*DEG
    S1, _ = mie_amplitudes(theta, a, n, wl)
    mie = gaussian_filter1d(np.abs(S1)**2/k0**2, 2.0/(k0*a)/(theta[1] - theta[0]))
    airy, _ = rainbow.airy_cross_section(theta, a, wl, n)
    pm, _ = find_peaks(mie, prominence=0.1*mie.max())
    pa, _ = find_peaks(airy)
    # main maximum within 0.07 deg, supernumeraries within 0.035 deg
    assert np.allclose(theta[pm][:3], theta[pa][:3], atol=[0.07*DEG, 0.035*DEG, 0.035*DEG])
    assert np.isclose(mie.max(), airy.max(), rtol=0.1)


def test_airy_far_from_bow_approaches_geometric():
    # averaged over many fringes, Ai^2 gives the geometric intensity of the
    # two branches of the cubic wavefront, b eps/sin(theta) sqrt(2/(|theta''| dtheta))
    n, wl, a = 1.3343, 550e-6, 5.0
    th_c, b_c, th2 = rainbow.airy_parameters(n, 1, a)
    eps_s, _ = rainbow.fresnel_factors(np.arcsin(b_c/a), n, 1)
    theta = th_c + np.linspace(0.3, 0.6, 200001)*DEG
    airy, _ = rainbow.airy_cross_section(theta, a, wl, n)
    cubic = b_c*eps_s/np.sin(theta)*np.sqrt(2/(abs(th2)*(theta - th_c)))
    assert np.isclose(airy.mean(), cubic.mean(), rtol=0.05)
    geo, _ = rainbow.geometric_cross_section(n, 1, a, theta=theta)
    assert np.isclose(airy.mean(), geo.mean(), rtol=0.15)


def test_colour():
    lam = np.arange(380.0, 781.0, 5.0)
    xyz = colour.spectrum_to_xyz(lam, np.ones_like(lam))
    assert np.isclose(xyz[1], 1.0)
    # equal energy white is nearly neutral
    x, y = xyz[0]/xyz.sum(), xyz[1]/xyz.sum()
    assert abs(x - 1/3) < 0.01 and abs(y - 1/3) < 0.01
    rgb = colour.spectrum_to_rgb(lam, np.exp(-0.5*((lam - 650)/10)**2), gain=10)
    assert rgb[0] > 0.9 and rgb[1] < 0.5 and rgb[2] < 0.5
    T = colour.planck(np.array([450.0, 560.0, 700.0]), 5800)
    assert np.isclose(T[1], 1.0) and T[0] > T[2]
