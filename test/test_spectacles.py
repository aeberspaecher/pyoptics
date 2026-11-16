#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the spectacle lens tools (notebook 129, progressive spectacle lenses)."""

import numpy as np
import pytest

from pyoptics import spectacles as sp


def test_accommodation_and_add():
    assert np.isclose(sp.accommodation_amplitude(40, "mean"), 18.5 - 12.0)
    assert np.isclose(sp.accommodation_amplitude(40, "min"), 5.0)
    assert sp.accommodation_amplitude(80, "min") == 0.0
    adds = sp.required_add(np.array([45, 50, 55, 60, 70]))
    assert np.allclose(adds, [0.75, 1.25, 2.0, 2.5, 2.5])
    assert np.all(np.diff(adds) >= 0)
    assert np.allclose(adds*4, np.round(adds*4))


def test_add_profile():
    y = np.array([10.0, 4.0, -5.0, -14.0, -20.0])
    P = sp.add_profile(y, 2.0)
    assert np.allclose(P[[0, 1]], 0.0) and np.allclose(P[[3, 4]], 2.0)
    assert np.isclose(P[2], 1.0)


def test_surface_power_of_sphere_and_cylinder():
    x = np.linspace(-10, 10, 81)
    X, Y = np.meshgrid(x, x, indexing="ij")
    R, n = 100.0, 1.5
    sphere = R - np.sqrt(R**2 - X**2 - Y**2)
    mean, cyl = sp.surface_power(x, x, sphere, n)
    c = (slice(30, 51), slice(30, 51))
    assert np.allclose(mean[c], 1000*(n - 1)/R, rtol=2e-3)
    assert np.allclose(cyl[c], 0, atol=5e-3)
    cylinder = R - np.sqrt(R**2 - X**2)
    mean, cyl = sp.surface_power(x, x, cylinder, n)
    assert np.allclose(mean[c], 500*(n - 1)/R, rtol=2e-3)
    assert np.allclose(cyl[c], 1000*(n - 1)/R, rtol=2e-3)


def test_sphero_cylinder():
    a = np.deg2rad(30)
    # +2 sphere, -1 cyl at 30 degrees: P = S I + C n n^T, n normal to the axis
    nn = np.array([-np.sin(a), np.cos(a)])
    P = 2*np.eye(2) - 1*np.outer(nn, nn)
    s, c, ax = sp.sphero_cylinder(P)
    assert np.isclose(s, 2) and np.isclose(c, -1) and np.isclose(ax, 30)


@pytest.fixture(scope="module")
def designs():
    x = np.linspace(-32, 32, 65)
    out = {}
    for kind in ("hard", "soft"):
        wm, wa = sp.default_weights(kind)
        d = sp.progressive_departure(x, 2.0, n=1.498, mean_weight=wm, astig_weight=wa)
        out[kind] = (x, d, sp.surface_power(x, x, d, 1.498))
    return out


def test_progressive_design(designs):
    x, d, (mean, cyl) = designs["hard"]
    i0 = np.argmin(np.abs(x))
    j = lambda y: np.argmin(np.abs(x - y))
    # no departure, no prism at the fitting cross (0, 4)
    assert abs(d[i0, j(4.0)]) < 1e-6
    # distance zone at zero power, near zone at the add (at the inset near point)
    assert abs(mean[i0, j(12.0)]) < 0.15
    assert abs(mean[np.argmin(np.abs(x + 2.5)), j(-16.0)] - 2.0) < 0.2
    # clear corridor, astigmatism beside it
    assert cyl[np.argmin(np.abs(x + 1.0)), j(-5.0)] < 0.5
    inside = np.add.outer(x**2, x**2) <= 25**2
    peak_hard = np.nanmax(np.where(inside, cyl, np.nan))
    peak_soft = np.nanmax(np.where(inside, designs["soft"][2][1], np.nan))
    assert peak_hard > 3.0
    assert peak_soft < peak_hard


def test_minkwitz(designs):
    # cylinder grows sideways at about twice the power gradient along the corridor
    x, d, (mean, cyl) = designs["hard"]
    j = np.argmin(np.abs(x + 5.0))
    k = np.argmin(np.abs(x - sp.umbilic_line(-5.0)))
    dPdy = -np.gradient(mean[k], x)[j]
    side = (cyl[k + 3, j] + cyl[k - 3, j])/2 - cyl[k, j]
    assert np.isclose(side/(3*(x[1] - x[0])), 2*dPdy, rtol=0.35)


def test_single_vision_traced_as_worn():
    x = np.linspace(-30, 30, 61)
    lens = sp.ProgressiveLens(x, None, front_power=5.0, distance_power=2.0, n=1.498)
    P, prism = lens.power(0.0, 0.0)
    s, c, _ = sp.sphero_cylinder(P)
    assert abs(s - 2.0) < 0.05 and abs(c) < 0.02
    assert np.allclose(prism, 0, atol=0.02)
    # looking down: base-up prism (Prentice: 0.1 * 2 D * 10 mm = 2 prism dioptres)
    P, prism = lens.power(0.0, -10.0)
    assert prism[1] > 1.5
    s, c, _ = sp.sphero_cylinder(P)
    assert abs(c) < 0.2
