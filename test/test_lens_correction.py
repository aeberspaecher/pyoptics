#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the lens correction module
(notebook 161, distortion, lateral colour and lens shading correction)."""

import numpy as np

from pyoptics import lens_correction as lc
from pyoptics import raytrace as rt
from pyoptics import sensor as se


def _ideal(f=5.0, field=np.radians(20.0)):
    return rt.System([rt.IdealLens(f, thickness=f, semi_diameter=1.0)], 0, 550e-6, field_angle=field)


def test_ideal_lens_is_rectilinear():
    s = _ideal()
    th = np.linspace(0, s.max_field, 7)
    h, ideal = lc.radial_mapping(s, th, wavelengths=[500e-6, 600e-6])
    assert np.allclose(h, ideal[None], atol=1e-9)
    assert np.allclose(np.degrees(lc.chief_ray_angles(s, th)), np.degrees(th), atol=1e-6)


def test_brown_fit_recovers_coefficients():
    r = np.linspace(0, 1, 50)
    k = (-0.12, 0.05, -0.01)
    kf, res = lc.fit_brown_radial(r, lc.brown_radial(r, k), n_terms=3)
    assert np.allclose(kf, k, atol=1e-10)
    assert np.max(np.abs(res)) < 1e-12


def test_warp_undoes_a_radial_distortion():
    # a smooth image distorted by r_d = r (1 + k r^2) is restored by the warp
    n, p = 101, 1.0
    yy, xx = np.indices((n, n)) - 50.0
    scene = lambda y, x: np.cos(0.11*x) + np.sin(0.07*y)
    k = 2e-5
    r_u = np.linspace(0, 80, 400)
    r_d = r_u*(1 + k*r_u**2)
    # the sensor image at radius r_d shows the scene point at r_u
    ru_of_rd = np.interp(np.hypot(yy, xx), r_d, r_u)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(np.hypot(yy, xx) > 0, ru_of_rd/np.hypot(yy, xx), 1.0)
    sensor = scene(yy*m, xx*m)
    rows, cols = lc.warp_coordinates((n, n), p, r_u, r_d)
    out = lc.remap(sensor, rows, cols, order=3)
    inner = np.hypot(yy, xx) < 35
    assert np.max(np.abs(out - scene(yy, xx))[inner]) < 5e-3


def test_ir_cut_moves_to_the_blue():
    lam = np.array([600.0, 640.0, 650.0, 660.0])
    T0 = lc.ir_cut_transmission(lam)
    T30 = lc.ir_cut_transmission(lam, np.radians(30.0))
    assert np.isclose(T0[2], 0.5)
    assert np.all(T30 <= T0)
    lc30 = 650*np.sqrt(1 - 0.25/1.8**2)
    assert np.isclose(lc.ir_cut_transmission([lc30], np.radians(30.0))[0], 0.5)
    a = lc.cone_angles(np.radians(20.0), 2.0, n=4000)
    # from the chief ray angle -/+ the half cone angle arcsin(1/(2N)) (in direction sines)
    assert np.isclose(np.sin(a.min()), np.sin(np.radians(20.0)) - 0.25, atol=0.01)
    assert np.isclose(np.sin(a.max()), np.sin(np.radians(20.0)) + 0.25, atol=0.01)


def test_shading_of_an_ideal_lens_follows_cos4():
    s = _ideal(field=np.radians(30.0))
    th = np.radians([10.0, 20.0, 30.0])
    lam = np.arange(450, 651, 50)
    S = np.ones((1, len(lam)))
    sh, parts = lc.channel_shading(s, th, lam, S, n_rays=20000)
    assert np.allclose(sh[0], np.cos(th)**4, atol=0.02)


def test_microlens_shift_restores_the_pixel():
    st = se.pixel_stack(pitch=0.001)
    cra = np.radians([0.0, 25.0])
    plain = lc.pixel_efficiency(st, cra, 1.8, n_rays=5000)
    shifted = lc.pixel_efficiency(st, cra, 1.8, shifts=[se.microlens_shift(st, c) for c in cra], n_rays=5000)
    assert plain[1] < 0.5*plain[0]
    assert shifted[1] > 0.6*shifted[0] and shifted[1] > 2*plain[1]


def test_gain_table_and_radial_map():
    m = lc.radial_map((9, 13), 1.0, [0.0, 10.0], [1.0, 2.0])
    assert np.isclose(m[4, 6], 1.0) and np.isclose(m[4, 0], 1.6)
    t = lc.gain_table(m, grid=(5, 7))
    assert t.shape == (5, 7) and np.isclose(t[0, 0], m[0, 0]) and np.isclose(t[-1, -1], m[-1, -1])
    img, g = lc.apply_gain_table(np.ones((9, 13)), t)
    assert np.allclose(g[[0, 0, -1, -1], [0, -1, 0, -1]], m[[0, 0, -1, -1], [0, -1, 0, -1]])
    assert np.max(np.abs(g - m)) < 0.1
