#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the phone camera lens helpers (notebook 159,
phone camera lens with Forbes Q polynomials)."""

import numpy as np

from pyoptics import phone_lens as pl
from pyoptics import qaspheres as qa
from pyoptics import raytrace as rt


def test_design_first_order_data():
    s = pl.phone_lens()
    p = s.paraxial()
    assert np.isclose(p.efl, pl.EFL, atol=5e-3)
    assert np.isclose(p.efl/(2*p.ep_radius), pl.F_NUMBER, atol=1e-3)
    assert np.isclose(np.degrees(s.max_field), 36.87, atol=0.01)
    assert all(isinstance(s.surfaces[i], qa.QbfsSurface) for i in pl.LENS_SURFACES)
    assert np.sum([S.thickness for S in s.surfaces]) <= 5.3 + 1e-6


def test_design_meets_sensor_limits():
    s = pl.phone_lens()
    assert np.all(pl.chief_ray_angle()(s) < 35.0)
    assert np.all(np.abs(pl.distortion()(s)) < 0.045)
    rims = pl.clear_apertures(s)
    assert pl.min_thickness([9], rims)(s)[0] > 0.045  # L5 clears the IR filter
    assert np.all(pl.min_thickness([0, 2, 4, 6, 8], rims)(s) > 0.24)
    assert np.all(pl.min_thickness([1, 3, 5, 7], rims)(s) > 0.025)
    assert s.rms_opd(0.0, N=32) < 0.2


def test_design_has_no_holes_between_fields():
    # with six fields in the merit function, an earlier design hid up to
    # 1.7 waves between them; scan the field densely at both distances
    s = pl.phone_lens()
    near, _ = pl.close_focus(s)
    for x in (s, near):
        w = [x.rms_opd(f*x.max_field, N=24) for f in np.linspace(0, 1, 41)]
        assert np.max(w) < 0.35


def test_operands_on_a_paraxial_lens():
    # a thin ideal lens with the stop on it: no distortion, chief ray angle = field angle
    s = rt.System([rt.IdealLens(10.0, thickness=10.0, semi_diameter=1.0)], 0, 550e-6,
                  field_angle=np.radians(20.0))
    assert np.allclose(pl.distortion((0.5, 1.0))(s), 0.0, atol=1e-9)
    assert np.allclose(pl.chief_ray_angle((0.5, 1.0))(s), [10.0, 20.0], atol=1e-6)


def test_min_thickness_continues_as_flange():
    s = rt.System([rt.Surface(5.0, 1.0, n_after=1.5, semi_diameter=2.0), rt.Surface(-5.0, 10.0, n_after=1.0)],
                  0, 550e-6)
    rims = [2.0, 1.0]
    t = pl.min_thickness([0], rims, n=200)(s)[0]
    edge = 1.0 + s.surfaces[1].sag(np.array([1.0]))[0] - s.surfaces[0].sag(np.array([4.0]))[0]
    assert np.isclose(t, edge, rtol=1e-9)


def test_monomial_refit_keeps_the_surface():
    s = pl.phone_lens()
    m, err = pl.to_monomials(s, surfaces=(0, 4))
    assert np.all(err < 1e-3)   # 1 um
    r = np.linspace(0, 1.1, 20)
    for i in (0, 4):
        assert np.allclose(m.surfaces[i].sag(r**2), s.surfaces[i].sag(r**2), atol=2e-4)


def test_close_focus_moves_the_lens_by_about_the_paraxial_shift():
    lens = pl.phone_lens()
    near, dz = pl.close_focus(lens)
    assert near.object_distance == pl.CLOSE_DISTANCE
    assert np.isclose(near.max_field, pl.CLOSE_DISTANCE*np.tan(lens.max_field))
    assert np.isclose(dz, pl.EFL**2/(pl.CLOSE_DISTANCE - pl.EFL), atol=0.015)
    # designed for both distances: as sharp at 100 mm as at infinity
    for f in (0.0, 0.5, 1.0):
        assert near.rms_opd(f*near.max_field, N=24) < 0.35
    # the earlier design for infinity only is not
    old, _ = pl.close_focus(pl.phone_lens(pl.DESIGN_INFINITY))
    assert old.rms_opd(0.5*old.max_field, N=24) > 0.5
