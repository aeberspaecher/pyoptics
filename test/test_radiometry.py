#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the radiometry tools: Planck against Stefan-Boltzmann and Wien, the
fits of V(lambda) and V'(lambda) against samples of the CIE tables, black
body efficacies, view factors (closed forms, reciprocity and Monte Carlo),
the integrating sphere (Monte Carlo against the sphere multiplier) and the
camera equation.
"""

import numpy as np
import pytest

from pyoptics import radiometry as rad
from pyoptics import illumination as il


# samples of the CIE tables (CIE 018:2019): photopic V (1924) and scotopic V' (1951)
CIE_V = {450: 0.038, 500: 0.323, 510: 0.503, 555: 1.000, 600: 0.631, 610: 0.503, 650: 0.107}
CIE_V_SCOT = {450: 0.455, 480: 0.793, 507: 1.000, 550: 0.481, 600: 0.03315}


def test_planck_integrates_to_stefan_boltzmann():
    lam = np.geomspace(50, 1e7, 400001)
    for T in (300.0, 2850.0, 5772.0):
        total = np.trapezoid(rad.planck_radiance(lam, T), lam)
        assert total == pytest.approx(rad.blackbody_radiance(T), rel=1e-4)


def test_wien_peak_and_photons():
    T = 5772.0
    lam = np.linspace(300, 1000, 70001)
    assert lam[np.argmax(rad.planck_radiance(lam, T))] == pytest.approx(rad.wien_peak(T), abs=0.02)
    q = rad.planck_radiance(550.0, T, photons=True)
    assert q == pytest.approx(rad.planck_radiance(550.0, T)/rad.photon_energy(550.0))


def test_luminous_efficiency_fits():
    for lam, v in CIE_V.items():
        assert rad.V_photopic(lam) == pytest.approx(v, abs=0.007)
    for lam, v in CIE_V_SCOT.items():
        assert rad.V_scotopic(lam) == pytest.approx(v, abs=0.006)


def test_efficacies():
    lam = np.linspace(540, 570, 3001)
    # a narrow line at 555 nm: close to K_m
    S = np.exp(-0.5*((lam - 555)/0.5)**2)
    assert rad.luminous_efficacy(lam, S) == pytest.approx(rad.K_M, rel=0.005)
    T = np.linspace(4000, 10000, 601)
    e = rad.blackbody_efficacy(T)
    assert T[np.argmax(e)] == pytest.approx(6600, abs=150)
    assert e.max() == pytest.approx(95, abs=1.5)
    assert rad.blackbody_efficacy(2856.0) == pytest.approx(16.5, abs=1.0)


def test_solar_constant_from_radiance():
    E = rad.blackbody_radiance(il.SUN_TEMPERATURE)*np.pi*np.sin(il.SUN_ANGULAR_RADIUS)**2
    assert E == pytest.approx(rad.SOLAR_CONSTANT, rel=0.005)


def test_solid_angles():
    assert rad.solid_angle_cone(np.pi/2) == pytest.approx(2*np.pi)
    assert rad.projected_solid_angle_cone(np.pi/2) == pytest.approx(np.pi)
    assert rad.solid_angle_cone(1e-3) == pytest.approx(rad.projected_solid_angle_cone(1e-3), rel=1e-5)


def test_view_factors():
    # reciprocity and limits
    r1, r2, d = 1.0, 2.5, 1.7
    assert r1**2*rad.view_factor_parallel_disks(r1, r2, d) == pytest.approx(
        r2**2*rad.view_factor_parallel_disks(r2, r1, d))
    assert rad.view_factor_parallel_disks(1e-4, 1.0, 1.0) == pytest.approx(
        rad.view_factor_point_disk(1.0, 1.0), rel=1e-6)
    assert rad.view_factor_parallel_disks(5.0, 5.0, 1e-3) == pytest.approx(1.0, abs=1e-3)
    # Monte Carlo: Lambertian emission from disk 1
    rng = np.random.default_rng(3)
    n = 400000
    P = il.disk_points(n, r1, rng)
    D = il.lambertian_directions(n, rng=rng)
    t = d/D[:, 2]
    hit = np.hypot(P[:, 0] + t*D[:, 0], P[:, 1] + t*D[:, 1]) <= r2
    F = rad.view_factor_parallel_disks(r1, r2, d)
    assert np.mean(hit) == pytest.approx(F, abs=4*np.sqrt(F*(1 - F)/n))


def test_integrating_sphere_against_multiplier():
    rho, R = 0.95, 1.0
    a_in, a_det = np.deg2rad(15.0), np.deg2rad(10.0)
    f_in, f_det = (1 - np.cos(a_in))/2, (1 - np.cos(a_det))/2
    n = 40000
    # a beam enters through the input port (at -z) and hits the opposite wall first
    P = np.tile([0.0, 0.0, -0.999*R], (n, 1))
    D = np.tile([0.0, 0.0, 1.0], (n, 1))
    res = rad.sphere_trace(P, D, R, rho, ports=[([0, 0, -1], a_in), ([1, 0, 0], a_det)],
                           power=1.0/n, rng=5)
    expected = rad.sphere_gain(rho, f_in + f_det)*f_det
    assert res.port_power[1] == pytest.approx(expected, rel=0.03)
    total = res.port_power.sum() + res.absorbed + res.lost
    assert total == pytest.approx(1.0, rel=1e-9)


def test_sphere_first_bounce_is_uniform():
    # light reflected once from a Lambertian point on the wall lands uniformly on the sphere
    n = 200000
    res = rad.sphere_trace(np.tile([0, 0, -0.999], (n, 1)), np.tile([0, 0, 1.0], (n, 1)), 1.0, 1.0,
                           max_bounces=1, keep_hits=2, rng=1)
    Q = res.hits[1][0]
    # uniform on a sphere: z uniform in [-1, 1]
    assert np.mean(Q[:, 2] < 0) == pytest.approx(0.5, abs=0.005)
    assert np.mean(Q[:, 2] > 0.5) == pytest.approx(0.25, abs=0.005)


def test_camera_equation():
    L = 1000.0
    E = rad.camera_irradiance(L, 2.0)
    assert E == pytest.approx(np.pi*L/16)
    # 1:1 with a symmetric lens: working F-number doubles, a quarter of the light
    assert rad.camera_irradiance(L, 2.0, magnification=-1.0) == pytest.approx(E/4)
    assert rad.working_f_number(2.0, -1.0, 2.0) == pytest.approx(3.0)
    assert rad.camera_irradiance(L, 2.0, field_angle=np.pi/4) == pytest.approx(E/4)


def test_exposure():
    assert rad.exposure_value(16.0, 1/125) == pytest.approx(14.97, abs=0.01)
    # sunny 16: a mean scene luminance of about 4000 cd/m^2 at ISO 100 and f/16 gives about 1/100 s
    t = rad.metered_exposure(4000.0, 100, 16.0)
    assert 1/t == pytest.approx(125, rel=0.3)
    n = rad.photons_per_pixel(1e-3, 550.0, 4e-6, 0.01)
    assert n == pytest.approx(1e-3*16e-12*0.01/(6.62607015e-34*299792458.0/550e-9))
