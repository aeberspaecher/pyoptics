#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the structured light module: pinhole cameras, triangulation,
fringe projection and the dot projector."""

import numpy as np
import pytest

from pyoptics import structured_light as sl
from pyoptics.interferometry import psi_n_step


Z0 = 500.0


def _bump(X, Y):
    return Z0 - 30*np.exp(-(X**2 + Y**2)/(2*40.0**2))


def _cam_proj():
    cam = sl.Pinhole([0, 0, 0], sl.look_at([0, 0, 0], [0, 0, Z0]), 600, (160, 120))
    proj = sl.Pinhole([150, 0, 0], sl.look_at([150, 0, 0], [0, 0, Z0]), 900, (640, 480))
    return cam, proj


def test_pinhole_round_trip():
    R = sl.look_at([10, 5, -20], [0, 0, 100])
    assert R @ R.T == pytest.approx(np.eye(3))
    cam = sl.Pinhole([10, 5, -20], R, 700, (200, 100))
    u, v = np.array([3.0, 150.0]), np.array([90.0, 7.0])
    d = cam.rays(u, v)
    P = cam.position + 123.0*d
    uu, vv = cam.project(P)
    assert uu == pytest.approx(u) and vv == pytest.approx(v)
    assert cam.project(cam.position - d)[0][0] != cam.project(cam.position - d)[0][0]  # NaN behind


def test_intersect_depth():
    cam, _ = _cam_proj()
    u, v = cam.pixel_grid()
    P = sl.intersect_depth(cam.position, cam.rays(u, v), _bump)
    assert np.max(np.abs(P[..., 2] - _bump(P[..., 0], P[..., 1]))) < 1e-9


def test_triangulation():
    f, b, th = 25.0, 100.0, np.radians(10)
    Z = np.array([300.0, 500.0])
    x = f*(b/Z - np.tan(th))
    assert sl.triangulation_depth(x, b, f, th) == pytest.approx(Z)
    h = 1e-6
    num = (sl.triangulation_depth(x + h, b, f, th) - sl.triangulation_depth(x - h, b, f, th))/(2*h)
    assert -num == pytest.approx(sl.triangulation_sensitivity(Z, b, f), rel=1e-5)
    assert sl.speckle_depth_uncertainty(0.6e-3, 0.01, np.pi/2) == pytest.approx(0.6e-3/(2*np.pi*0.01))


def test_fringe_projection_exact():
    cam, proj = _cam_proj()
    u, v = cam.pixel_grid()
    P = sl.intersect_depth(cam.position, cam.rays(u, v), _bump)
    up, _ = proj.project(P)
    periods = [700.0, 100.0, 16.0]
    phases = [psi_n_step(sl.fringe_patterns(up, p, 4))[0] for p in periods]
    uu = sl.temporal_unwrap(phases, periods)
    assert np.max(np.abs(uu - up)) < 1e-9
    Q = sl.triangulate_column(cam, proj, u, v, uu)
    assert np.max(np.abs(Q - P)) < 1e-6


def test_gamma_harmonics():
    # a projector gamma adds harmonics; 3 steps alias the 2nd harmonic,
    # more steps suppress it
    u = np.linspace(0, 64, 257)
    err = {}
    for N in (3, 4, 8):
        phi = psi_n_step(sl.fringe_patterns(u, 32.0, N, gamma=2.2))[0]
        e = np.angle(np.exp(1j*(phi - 2*np.pi*u/32)))
        err[N] = np.max(np.abs(e))
    assert err[3] > 0.05 and err[4] > 1e-2 and err[8] < 1e-3


def test_fanout_ifta():
    orders = np.array([(i, j) for i in range(-2, 3) for j in range(-2, 3)])
    phase, eta = sl.fanout_ifta(32, orders, 150, rng=0)
    assert eta.sum() > 0.8
    assert (eta.max() - eta.min())/(eta.max() + eta.min()) < 0.02
    assert eta == pytest.approx(sl.order_efficiencies(phase, orders))
    phase, eta = sl.fanout_ifta(32, orders, 150, levels=2, rng=0)
    assert set(np.round(np.unique(phase), 9)) <= {0.0, round(np.pi, 9)}
    assert eta.sum() > 0.45
    assert (eta.max() - eta.min())/(eta.max() + eta.min()) < 0.1


def test_dot_directions():
    em = np.array([[0.0, 0.0], [0.1, 0.0]])
    k = sl.dot_directions(em, 1.0, np.array([[0, 0], [1, 0]]), 10e-6, 1e-6)
    assert k.shape == (4, 2)
    assert k[0] == pytest.approx([0, 0])
    assert k[1, 0] == pytest.approx(-0.1/np.sqrt(1.01))
    assert k[2, 0] == pytest.approx(0.1)
    assert sl.dot_directions(em, 1.0, np.array([[20, 0]]), 10e-6, 1e-6).shape == (0, 2)


def test_dots_block_matching_depth():
    rng = np.random.default_rng(3)
    k = rng.uniform(-0.25, 0.25, (3000, 2))
    cam = sl.Pinhole([0, 0, 0], np.eye(3), 400, (200, 150))
    src = [30.0, 0.0, 0.0]
    ref, _ = sl.render_dots(cam, src, k, lambda X, Y: 600 + 0*X)
    plane, uv = sl.render_dots(cam, src, k, lambda X, Y: 450 + 0*X)
    d, score = sl.block_match(plane, ref, 11, 30, -2)
    expect = 400*30*(1/450 - 1/600)
    inner = d[20:-20, 40:-20]
    assert np.median(inner) == pytest.approx(expect, abs=0.05)
    Z = sl.depth_from_disparity(np.median(inner), 30, 400, 600)
    assert Z == pytest.approx(450, rel=2e-3)
    assert np.median(score[20:-20, 40:-20]) > 0.9
