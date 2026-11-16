#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.deflectometry."""

import numpy as np
import pytest

from pyoptics import deflectometry as df
from pyoptics.structured_light import Pinhole, look_at


R = 2000.0


def surf(X, Y):
    return -(X**2 + Y**2)/(2*R) + 0.0005*np.cos(2*np.pi*X/10)


def slopes(X, Y):
    return -X/R - 0.0005*2*np.pi/10*np.sin(2*np.pi*X/10), -Y/R


def setup():
    cam = Pinhole((-120, 0, 600), look_at((-120, 0, 600), (0, 0, 0)), 1800, (160, 160))
    scr = df.Screen((120, 0, 600), look_at((120, 0, 600), (0, 0, 0)), 0.25, (1920, 1080))
    return cam, scr


def test_reflection_and_normals():
    n = df.normals_from_slopes(0.1, -0.2)
    assert np.linalg.norm(n) == pytest.approx(1.0)
    assert df.slopes_from_normals(n) == pytest.approx((0.1, -0.2))
    d = np.array([0.3, 0.1, -1.0])
    r = df.reflect(d, n)
    assert np.linalg.norm(r) == pytest.approx(np.linalg.norm(d))
    assert np.dot(r, n) == pytest.approx(-np.dot(d, n))


def test_screen_round_trip():
    _, scr = setup()
    u, v = np.array([10.0, 900.0]), np.array([1000.0, 3.5])
    P = scr.to_world(u, v)
    o = np.array([0.0, 0.0, 0.0])
    uu, vv = scr.intersect(np.broadcast_to(o, P.shape), P - o)
    assert uu == pytest.approx(u) and vv == pytest.approx(v)
    assert np.all(scr.contains(u, v)) and not scr.contains(-1.0, 0.0)


def test_slopes_from_true_points_are_exact():
    cam, scr = setup()
    P, us, vs = df.trace(cam, scr, surf, slopes)
    ok = np.isfinite(us)
    assert ok.mean() > 0.9
    gx, gy = df.slopes_from_points(cam.position, P[ok], scr.to_world(us[ok], vs[ok]))
    tx, ty = slopes(P[ok][:, 0], P[ok][:, 1])
    assert np.max(np.abs(gx - tx)) < 1e-9 and np.max(np.abs(gy - ty)) < 1e-9


def test_reconstruction_converges():
    cam, scr = setup()
    _, us, vs = df.trace(cam, scr, surf, slopes)
    x = np.linspace(-30, 30, 61)
    y = np.linspace(-30, 30, 61)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z, gx, gy, hist = df.reconstruct(cam, scr, us, vs, x, y, anchor=(0.0, 0.0, surf(0, 0)),
                                     n_iter=4)
    err = [np.nanstd(H - surf(X, Y)) for H in hist]
    assert np.isnan(Z[0, 0]) and np.isfinite(Z[30, 30])
    assert err[0] > 10*err[-1]
    assert err[-1] < 3e-5  # 30 nm on a 1 um ripple and 0.6 mm sag


def test_sensitivity_helpers():
    assert df.slope_resolution(0.01, 500.0) == pytest.approx(1e-5)
    assert df.screen_uncertainty(5.0, 2*np.pi/1000) == pytest.approx(0.005)
    assert df.defocus_contrast(5.0, 0.0) == 1.0
    assert df.defocus_contrast(5.0, 5.0*3.8317/np.pi) == pytest.approx(0.0, abs=1e-4)
    assert df.blur_diameter(10.0, 600.0, 600.0) == pytest.approx(10.0)
