#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the y-ybar (Delano) diagram against paraxial ray traces, thin and
thick lens formulas and the cardinal points of abcd.
"""

import numpy as np
import pytest

from pyoptics import abcd
from pyoptics import raytrace as rt
from pyoptics.delano import DelanoDiagram, principal_planes, thick_lens_curvatures, thicken


WL = 248.3e-6  # mm


def relay():
    """Double telecentric 4x reduction relay of three thin lenses."""

    h, m, NA = 40.0, -0.25, 0.7
    yb = [h, h, 0.3*h, m*h, m*h]
    y = [0.0, 20.0, 60.0, 15.0, 0.0]

    return DelanoDiagram(yb, y, -NA*abs(m)*h).with_length(1000.0)


def test_singlet_spacings_and_power():
    # thin lens f = 100 imaging from 150 mm to 300 mm, stop 50 mm behind it
    f, s, s1, a = 100.0, 150.0, 300.0, 50.0
    system = rt.System([rt.IdealLens(f, a), rt.Surface(np.inf, s1 - a, semi_diameter=5.0)], 1,
                       0.55e-3, object_distance=s, object_height=2.0)
    d = DelanoDiagram.from_system(system)
    assert np.allclose(d.thickness, [s, a, s1 - a])
    assert np.isclose(d.power[0], 1/f)
    assert np.isclose(d.power[1], 0.0, atol=1e-15)
    assert np.isclose(d.magnification, -s1/s)
    assert np.allclose(d.pupils().stop, [s + a])
    cp = d.cardinal_points()
    assert np.isclose(cp.efl, f)
    assert np.allclose([cp.front_principal, cp.rear_principal], [s, s])
    assert np.allclose([cp.front_focal, cp.rear_focal], [s - f, s + f])


def test_caldwell_diagram_matches_paraxial_trace():
    system = rt.caldwell_krf_lens()
    p = system.paraxial()
    d = DelanoDiagram.from_system(system)
    n_s = system.n*system.sigma
    powers = [s.paraxial_power(n_s[i], n_s[i + 1]) for i, s in enumerate(system.surfaces)]
    assert np.allclose(d.power, powers, rtol=0, atol=1e-13)
    assert np.allclose(d.thickness[1:-1], [s.thickness for s in system.surfaces[:-1]])
    assert np.isclose(d.thickness[0], system.object_distance)
    assert np.isclose(d.thickness[-1], p.image_distance)
    assert np.isclose(d.magnification, p.magnification)
    assert np.isclose(d.NA[1], p.na)
    assert np.isclose(d.NA[0], p.na*abs(p.magnification), rtol=1e-4)
    assert np.isclose(d.H, p.lagrange)
    cp = d.cardinal_points()
    assert np.isclose(cp.efl, p.efl)
    z1 = system.object_distance  # first vertex in the diagram's z
    assert np.isclose(cp.front_principal - z1, p.cardinal_points.H)
    assert np.isclose(cp.front_focal - z1, p.cardinal_points.F)
    pupils = d.pupils()
    assert np.isclose(pupils.entrance - z1, p.ep_z)
    assert np.isclose(pupils.exit - z1, p.xp_z)
    z_stop = z1 + system.vertex_z[system.stop_index]
    assert np.any(np.isclose(pupils.stop, z_stop, atol=1e-6))


def test_cardinal_points_match_abcd():
    # telephoto pair in air
    d_ = DelanoDiagram.from_system(rt.System([rt.IdealLens(80.0, 40.0, semi_diameter=10.0),
                                             rt.IdealLens(-50.0, 60.0)], 0, 0.55e-3,
                                            field_angle=0.01))
    M = abcd.compose(abcd.thin_lens(80.0), abcd.free_space(40.0), abcd.thin_lens(-50.0))
    cp = abcd.cardinal_points(M, 40.0)
    c = d_.cardinal_points()
    assert np.isclose(c.efl, cp.f_prime)
    z1 = d_.z[1]  # first vertex (the first point lies in object space)
    assert d_.labels[0] == "object space"
    assert np.isclose(c.rear_principal - z1, cp.H_prime)
    assert np.isclose(c.front_principal - z1, cp.H)
    assert np.isclose(c.rear_focal - z1, cp.F_prime)


def test_conjugate_on_line_through_origin():
    d = relay()
    # the image of the object point on the last space is the image point
    z, mag = d.conjugate(len(d) - 2, 0)
    assert np.isclose(z, d.length)
    assert np.isclose(mag, d.magnification)


def test_thin_lens_system_from_diagram():
    d = relay()
    assert np.isclose(d.length, 1000.0)
    assert np.allclose(d.NA, [0.175, 0.7])
    system = d.to_system(WL)
    p = system.paraxial()
    assert np.isclose(p.magnification, -0.25)
    assert np.isclose(p.na, 0.7)
    assert np.isclose(system.object_distance + system.image_z, 1000.0)
    assert abs(p.ep_z) > 1e12 and abs(p.xp_z) > 1e12  # telecentric on both sides
    # round trip: the lens points of the system's diagram are those drawn
    back = DelanoDiagram.from_system(system)
    lenses = [i for i, s in enumerate(system.surfaces) if isinstance(s, rt.IdealLens)]
    assert np.allclose(back.points[[0] + [i + 1 for i in lenses] + [-1]], d.points)
    assert np.allclose(back.power[lenses], d.power)


def test_with_length_keeps_invariants():
    d = relay()
    d2 = d.with_length(500.0)
    assert np.isclose(d2.length, 500.0)
    assert np.allclose(d2.NA, d.NA)
    assert np.isclose(d2.magnification, d.magnification)
    assert np.allclose(d2.power, 2*d.power)


def test_negative_space_raises():
    d = DelanoDiagram([10.0, 10.0, 12.0, -2.5, -2.5], [0.0, 5.0, 4.0, 3.0, 0.0], -1.75)
    assert np.any(d.tau < 0)
    with pytest.raises(ValueError):
        d.to_system(WL)


@pytest.mark.parametrize("shape", [-2.0, -1.0, 0.0, 0.7, 3.0])
def test_thick_lens_curvatures(shape):
    n, t, phi = 1.5, 12.0, 1/150.0
    c1, c2 = thick_lens_curvatures(phi, n, t, shape)
    assert np.isclose((c1 + c2)/(c1 - c2), shape)
    M = abcd.thick_lens(1/c1 if c1 else np.inf, 1/c2 if c2 else np.inf, t, n)
    assert np.isclose(-M[1, 0], phi)
    phi_, delta, delta_prime = principal_planes(c1, c2, n, t)
    assert np.isclose(phi_, phi)
    cp = abcd.cardinal_points(M, t)
    assert np.isclose(cp.H, delta)
    assert np.isclose(cp.H_prime - t, delta_prime)


def test_thicken_keeps_first_order_properties():
    d = relay()
    for length in (None, 1000.0):
        system = thicken(d, WL, rt.fused_silica, [20.0, 30.0, 25.0], [0.5, -0.3, 1.2],
                         length=length)
        p = system.paraxial()
        assert np.isclose(p.magnification, -0.25)
        assert np.isclose(p.na, 0.7)
        assert abs(p.ep_z) > 1e12 and abs(p.xp_z) > 1e12
        assert all(s.thickness > 0 for s in system.surfaces)
        total = system.object_distance + system.image_z
        if length is None:
            # the lengths grow by the distances between the principal planes
            assert total > 1000.0
        else:
            assert np.isclose(total, 1000.0)
    # the paraxial rays in the air spaces are those of the diagram: the
    # thick system's diagram goes through the thin lens points' lines, e.g.
    # the object space and image space edges are the same
    back = DelanoDiagram.from_system(thicken(d, WL, rt.fused_silica, 20.0, 0.0))
    assert np.allclose(back.slopes[[0, -1]], d.slopes[[0, -1]])
    assert np.isclose(back.H, d.H)


def test_delano_krf_lens_meets_the_specification():
    system = rt.delano_krf_lens()
    p = system.paraxial()
    assert np.isclose(p.magnification, -0.25, atol=2e-4)
    assert np.isclose(p.na, 0.7, atol=1e-3)
    assert np.isclose(system.object_distance + system.image_z, 1000.0)
    glass = [i for i in range(len(system.surfaces)) if system.n[i + 1] > 1.2]
    assert len(glass) == 20
    assert min(system.surfaces[i].thickness for i in glass) > 9.99
    assert sum(1 for s in system.surfaces if any(s.aspheric)) == 5
    # telecentric on both sides: slopes of the real chief ray at the edge of
    # the field, aimed at the centre of the stop
    rays = system.rays(system.max_field, 0.0, 0.0)
    d = np.reshape(rays.directions, (-1, 3))[0]
    I, D = system._image_hits(system.trace(rays, clip=False))
    assert abs(d[1]/d[2]) < 2e-3
    assert abs(D[0, 1]/D[0, 2]) < 1e-3
    assert abs(I[0, 1] + 0.25*system.max_field) < 5e-5
    assert system.rms_opd(field=system.max_field, N=16) < 0.3
