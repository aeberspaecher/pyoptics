#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the scan lens helpers: the F-theta law and telecentricity of an
ideal lens, the galvo beam through a lens, the inversion of the scan and
the correction table, and the F-theta operand.
"""

import numpy as np
import pytest

from pyoptics import lens_optimization as lo
from pyoptics import lidar
from pyoptics import raytrace as rt
from pyoptics import scan_lenses as sl


F = 100.0
WL = 1064e-6


def ideal_scan_lens(stop_distance=F):
    """Ideal (paraxial) lens with the scanner in front of it; telecentric
    for stop_distance = F."""

    S = [rt.Surface(np.inf, stop_distance, semi_diameter=5.0), rt.IdealLens(F, F)]

    return rt.System(S, 0, WL, field_angle=np.radians(20), entrance_pupil_radius=5.0,
                     ray_aiming=False)


def test_ideal_lens_is_f_tan_theta_and_telecentric():
    s = ideal_scan_lens()
    th = np.radians([5.0, 10.0, 20.0])
    assert np.allclose(sl.chief_ray_heights(s, th), F*np.tan(th), atol=1e-9)
    assert np.allclose(sl.ftheta_error(s, th), F*(np.tan(th) - th), atol=1e-9)
    assert np.allclose(sl.ftheta_distortion(s, th), 100*(np.tan(th)/th - 1), atol=1e-7)
    assert np.allclose(sl.image_telecentricity(s, th), 0.0, atol=1e-12)


def test_telecentricity_of_a_non_telecentric_stop():
    # stop at the lens: the chief rays go through its centre undeviated
    s = ideal_scan_lens(stop_distance=0.0)
    th = np.radians([5.0, 10.0])
    assert np.allclose(sl.image_telecentricity(s, th), th, atol=1e-9)


def test_galvo_beam_matches_galvo_xy():
    tx, ty = np.meshgrid(np.radians([-4, 0, 3]), np.radians([-5, 2]))
    P, D = lidar.galvo_beam(tx, ty, 20.0)
    assert P.shape == tx.shape + (3,)
    assert np.allclose(np.linalg.norm(D, axis=-1), 1.0)
    x, y = lidar.galvo_xy(tx, ty, 20.0, 300.0)
    s = (300.0 - P[..., 2])/D[..., 2]
    assert np.allclose(P[..., 0] + s*D[..., 0], x)
    assert np.allclose(P[..., 1] + s*D[..., 1] - 20.0, y)


def test_galvo_spots_through_a_telecentric_lens():
    s = ideal_scan_lens()
    # one mirror only: the y mirror turns about a point on the axis, an
    # ideal lens maps its optical angle 2*theta to F*tan(2*theta)
    for pupil in (0.0, 0.5, 1.0):
        x, y, _ = sl.galvo_spots(s, 0.0, np.radians([-3, 4]), 10.0, pupil=pupil)
        assert np.allclose(np.abs(y), F*np.tan(np.radians([6, 8])), atol=1e-9)
        assert np.allclose(x, 0.0, atol=1e-12)
    # the x mirror alone: the beam pivots about the x mirror, the stop
    # (front focal plane) is at the x mirror for pupil = 0 -> telecentric
    x, y, D = sl.galvo_spots(s, np.radians(5), 0.0, 10.0, pupil=0.0)
    assert np.isclose(abs(x), F*np.tan(np.radians(10)))
    assert abs(D[0]/D[2]) < 1e-12
    # the line bow: lines of constant x mirror angle bend with y
    tx = np.radians(5.0)
    x, _, _ = sl.galvo_spots(s, tx, np.radians([0.0, 5.0, 10.0]), 10.0)
    assert abs(x[1]) > abs(x[0]) and abs(x[2]) > abs(x[1])


def test_invert_scan_and_correction_table():
    s = ideal_scan_lens()

    def spots(a, b):
        return sl.galvo_spots(s, a, b, 10.0)[:2]

    xt, yt = np.array([10.0, -20.0, 0.0]), np.array([15.0, 5.0, -25.0])
    tx, ty, err = sl.invert_scan(spots, xt, yt)
    assert np.all(err < 1e-8)
    x, y = spots(tx, ty)
    assert np.allclose(x, xt) and np.allclose(y, yt)

    nodes = np.linspace(-30.0, 30.0, 13)
    table = sl.CorrectionTable(spots, nodes, nodes, method="cubic")
    assert np.all(table.error < 1e-8)
    q = np.array([[3.3, -17.1], [22.2, 28.9], [-29.0, 0.4]])
    ax, ay = table(q[:, 0], q[:, 1])
    x, y = spots(ax, ay)
    assert np.max(np.hypot(x - q[:, 0], y - q[:, 1])) < 1e-3
    # linear interpolation is coarser but still well below the bow
    lin = sl.CorrectionTable(spots, nodes, nodes)
    x, y = spots(*lin(q[:, 0], q[:, 1]))
    assert np.max(np.hypot(x - q[:, 0], y - q[:, 1])) < 0.05


def test_ftheta_operand():
    s = ideal_scan_lens()
    op = lo.ftheta(fields=(0.5, 1.0), tolerance=1e-3)
    th = np.array([0.5, 1.0])*s.max_field
    assert np.allclose(op(s), F*(np.tan(th) - th)/1e-3, rtol=1e-6)
    op = lo.ftheta(efl=90.0, fields=(1.0,), tolerance=1.0)
    assert np.isclose(op(s)[0], F*np.tan(s.max_field) - 90.0*s.max_field)


@pytest.mark.parametrize("field_deg", [10.0])
def test_ftheta_optimization_of_a_singlet_pair(field_deg):
    """Two singlets behind a scanner: DLS with the F-theta operand and a
    focal length solve pulls the distortion from f*tan(theta) to f*theta."""

    S = [rt.Surface(np.inf, 20.0, semi_diameter=3.0),
         rt.Surface(-30.0, 4.0, n_after=1.5), rt.Surface(-40.0, 2.0, n_after=1.0),
         rt.Surface(300.0, 6.0, n_after=1.5), rt.Surface(-60.0, 95.0, n_after=1.0)]
    s0 = rt.System(S, 0, WL, field_angle=np.radians(field_deg), entrance_pupil_radius=3.0,
                   ray_aiming=False)
    fields = np.radians([5.0, 10.0])
    before = np.max(np.abs(sl.ftheta_error(s0, fields)))
    opt = lo.DampedLeastSquares(s0, [lo.Curvature(i) for i in (1, 2, 3)],
                                [lo.ftheta(fields=(0.5, 1.0), tolerance=1e-3),
                                 lo.spot_rms(fields=(0.0, 1.0), tolerance=0.05)],
                                solves=[lo.solve_curvature_for_efl(4, F),
                                        lo.solve_paraxial_focus()])
    s = opt.run(max_iterations=30).system
    assert np.isclose(s.paraxial().efl, F, rtol=1e-6)
    after = np.max(np.abs(sl.ftheta_error(s, fields)))
    assert after < 0.1*before
