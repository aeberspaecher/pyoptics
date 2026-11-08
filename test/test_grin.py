#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for the gradient-index ray tracer pyoptics.grin, against the
closed form ray paths of the Luneburg lens, Maxwell's fish-eye, the
parabolic and the hyperbolic secant rod and the exponential layer.

References
----------
A. Sharma, D. V. Kumar and A. K. Ghatak, "Tracing rays through
graded-index media: a new method", Appl. Opt. 21, 984-987 (1982).

M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), section 4.2 (the fish-eye).

A. L. Mikaelian, "Self-focusing media with variable index of refraction",
Progress in Optics 17, 279-345 (1980).
"""

import numpy as np
import pytest

from pyoptics import grin


def _fan(angles_deg, start=(0.0, 0.0)):
    a = np.radians(angles_deg)
    return np.tile(start, (len(a), 1)), np.c_[np.sin(a), np.cos(a)]


def test_parabolic_rod_exact_and_fourth_order():
    rod = grin.ParabolicRod(1.6, 2*np.pi/1000, n_min=0.5)
    r0, d = _fan([2, 10, 20, 30])
    errs = []
    for dt in (20.0, 10.0, 5.0):
        rays = grin.trace(rod, r0, d, dt, t_max=800)
        assert np.allclose(rays.t, 800)
        errs.append(np.abs(rays.position - rod.exact(r0, 1.6*d, rays.t)).max())
    order = np.log2(np.array(errs[:-1])/errs[1:])
    assert np.all(order > 3.8)
    assert errs[-1] < 1e-6


def test_parabolic_rod_pitch_and_skew_rays():
    rod = grin.ParabolicRod(1.5, 0.01, n_min=0.5)
    # pitch of a meridional ray: the first return to the axis after two crossings
    th = np.radians(25)
    r0, d = _fan([25])
    beta = 1.5*np.cos(th)
    half = grin.trace(rod, r0 + [1e-9, 0], d, 0.5, stop=lambda r: r[:, 0])  # first crossing of x = 0
    assert np.isclose(half.position[0, 1], rod.pitch(beta)/2, rtol=1e-9)
    # 3D skew ray against the exact solution, n_min is not reached
    r0 = np.array([[20.0, 0.0, 0.0]])
    T0 = np.array([[0.0, 0.2, 0.0]])
    T0[0, 2] = np.sqrt(rod.n2(r0)[0] - 0.2**2)
    rays = grin.trace(rod, r0, T0, 1.0, t_max=900)
    assert np.abs(rays.position - rod.exact(r0, T0, rays.t)).max() < 1e-6
    # the invariant T_z and the angular momentum x T_y - y T_x
    assert np.isclose(rays.T[0, 2], T0[0, 2])
    L = rays.position[0, 0]*rays.T[0, 1] - rays.position[0, 1]*rays.T[0, 0]
    assert np.isclose(L, 20*0.2)


def test_sech_rod_every_meridional_ray_has_the_same_pitch():
    rod = grin.SechRod(1.6, 2*np.pi/1000)
    x0 = np.array([0.0, 0.0, 50.0, 100.0, -80.0])
    th = np.radians([5, 30, 10, 0, -20])
    r0 = np.c_[x0, 0*x0]
    d = np.c_[np.sin(th), np.cos(th)]
    for z_end in (300.0, 1000.0):
        rays = grin.trace(rod, r0, d, 2.0, stop=lambda r: r[:, 1] - z_end)
        assert rays.stopped.all() and np.allclose(rays.position[:, 1], z_end, atol=1e-10)
        assert np.allclose(rays.position[:, 0], rod.exact_meridional(x0, np.tan(th), z_end), atol=1e-6)
    # after one pitch every ray is back where it started, with its slope
    assert np.allclose(rays.position[:, 0], x0, atol=1e-6)
    assert np.allclose(rays.direction[:, 0], d[:, 0], atol=1e-8)


def test_luneburg_focus_and_ellipses():
    lens = grin.Luneburg(2.0, center=(1.0, 0.5))
    b = np.linspace(-0.95, 0.95, 9)*2.0
    r0 = np.c_[1.0 - np.sqrt(4 - b**2)*(1 - 1e-12), 0.5 + b]
    d = np.tile([1.0, 0.0], (len(b), 1))
    rays = grin.trace(lens, r0, d, 0.02,
                      stop=lambda r: np.sum((r - [1.0, 0.5])**2, axis=1) - 4.0)
    assert rays.stopped.all()
    # every ray of the bundle meets the opposite point of the surface at t = pi R/2
    assert np.allclose(rays.position, [3.0, 0.5], atol=1e-8)
    assert np.allclose(rays.t, np.pi, atol=1e-8)
    assert np.allclose(rays.direction[:, 0]*0 + np.linalg.norm(rays.T, axis=1), 1.0)
    # the recorded path lies on the exact ellipses
    k = 30
    assert np.allclose(rays.path[k], lens.exact(r0, d, 0.02*k), atol=1e-9)
    # all optical paths from the entrance plane x = -1 to the focus are equal (Fermat)
    opl = rays.opl + (r0[:, 0] - (-1.0))
    assert np.ptp(opl) < 1e-8


def test_fisheye_circles_image_and_equal_paths():
    eye = grin.MaxwellFishEye(1.0)
    r0 = np.array([0.5, 0.0])
    a = np.radians([60, 90, 120, 150, 170, -100])
    d = np.c_[np.cos(a), np.sin(a)]
    img = eye.image(r0)
    assert np.allclose(img, [-2.0, 0.0])
    # the circles through r0 and the image cross the x axis only there: stop on the axis
    side = np.sign(d[:, 1])
    rays = grin.trace(eye, np.tile(r0, (len(a), 1)), d, 0.002, stop=lambda r: side*r[:, 1] + 1e-12)
    assert rays.stopped.all()
    assert np.allclose(rays.position, img, atol=1e-8)
    c, R = eye.circle(np.tile(r0, (len(a), 1)), d)
    assert np.abs(np.linalg.norm(rays.path - c, axis=-1) - R).max() < 1e-8
    assert np.ptp(rays.opl) < 1e-8


def test_exponential_layer_paths():
    air = grin.ExponentialLayer(1.000270, 6.6e-5, 0.05)
    # rays down from an eye 1.6 m above the road at small depression angles
    a = -np.radians([0.38, 0.41, 0.44])
    r0 = np.tile([0.0, 1.6], (len(a), 1))
    d = np.c_[np.cos(a), np.sin(a)]
    rays = grin.trace(air, r0, d, 0.5, stop=lambda r: r[:, 0] - 600.0)
    beta = air.n(r0)*np.cos(a)
    y_t = air.turning_height(beta)
    assert np.all(y_t > 0)
    # lowest point and its horizontal position against the closed form
    i_low = np.argmin(rays.path[:, :, 1], axis=0)
    y_low = rays.path[i_low, np.arange(len(a)), 1]
    assert np.allclose(y_low, y_t, atol=2e-3)
    x_t = air.horizontal_distance(beta, 1.6)
    yy = np.linspace(0.3, 1.5, 5)
    for j in range(len(a)):
        p = rays.path[:, j]
        up = np.arange(len(p)) > i_low[j]
        x_up = np.interp(yy, p[up, 1], p[up, 0])
        assert np.allclose(x_up, x_t[j] + air.horizontal_distance(beta[j], yy), rtol=1e-6)
    # the invariant n cos(elevation)
    assert np.allclose(rays.T[:, 0], beta, rtol=1e-12)


def test_numerical_gradient_matches_analytic():
    eye = grin.MaxwellFishEye(1.3, n0=1.8, center=(0.2, -0.1, 0.4))
    generic = grin.IndexField(n=eye.n, h=1e-6)
    r = np.random.default_rng(1).normal(size=(20, 3))
    assert np.allclose(generic.force(r), eye.force(r), atol=1e-7)
    rays_a = grin.trace(eye, r[:3], np.ones((3, 3)), 0.01, t_max=1.0)
    rays_n = grin.trace(generic, r[:3], np.ones((3, 3)), 0.01, t_max=1.0)
    assert np.allclose(rays_a.position, rays_n.position, atol=1e-7)


def test_trace_arguments_and_index_map():
    rod = grin.SechRod(1.5, 0.1)
    with pytest.raises(ValueError):
        grin.trace(rod, [0.0, 0.0], [0.0, 1.0], 0.1)
    rays = grin.trace(rod, [0.0, 0.0], [0.0, 1.0], 0.1, t_max=1.0, record=False)
    assert rays.path is None and np.allclose(rays.position, [[0.0, 1.5]])
    assert np.isclose(rays.opl[0], 1.5**2)    # n ds = n^2 dt
    x, z = np.linspace(-1, 1, 5), np.linspace(0, 2, 3)
    n = grin.index_map(rod, x, z)
    assert n.shape == (5, 3) and np.allclose(n[:, 0], 1.5/np.cosh(0.1*x))
    # rays that start on the stop surface are stopped at once
    rays = grin.trace(rod, [[0.0, 0.0], [0.1, 0.0]], [0.0, 1.0], 0.1, stop=lambda r: r[:, 0])
    assert rays.stopped[0] and rays.t[0] == 0
