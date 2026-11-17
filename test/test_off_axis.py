#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of off_axis (off-axis segments, fields, clearance, the Cook TMA,
freeform mirrors and the operands of lens_optimization) and of the new
variables of lens_optimization."""

import numpy as np
import pytest

from pyoptics import lens_optimization as lo
from pyoptics import off_axis as oa
from pyoptics import raytrace as rt


DEG = np.radians(1)


def test_fields():
    F = oa.rectangular_field((0.1, 0.2), (0.3, 0.4), (3, 2))
    assert F.shape == (6, 2)
    assert np.allclose(F[0], [-0.2, -0.2]) and np.allclose(F[-1], [0.4, 0.6])
    R = oa.ring_field(2.0, np.radians(30), width=0.2, n_arc=5, n_radial=3)
    assert R.shape == (15, 2)
    r = np.hypot(R[:, 0], R[:, 1])
    assert np.allclose(sorted(set(np.round(r, 9))), [1.9, 2.0, 2.1])
    a = np.arctan2(R[:, 1], R[:, 0])
    assert np.isclose(a.min(), np.radians(60)) and np.isclose(a.max(), np.radians(120))


@pytest.mark.parametrize("K", [-1.0, 0.0, -2.0])
def test_off_axis_conic_radii(K):
    """Second order of the segment sag against the closed form radii."""
    R, h, a = -1000.0, 200.0, 20.0
    g = oa.off_axis_conic(R, K, h)
    z = oa.segment_zernikes(R, K, h, a, n_terms=9)
    assert np.isclose(z[3], a**2/8*(1/g.R_sagittal + 1/g.R_tangential), rtol=2e-3)
    assert np.isclose(z[4], a**2/4*(1/g.R_sagittal - 1/g.R_tangential), rtol=2e-3, atol=1e-9)
    # the tangent frame: no tilt beyond the one hidden in Z8 (Z3 = 2 Z8 for a pure coma cap)
    assert abs(z[1]) < 1e-9 and np.isclose(z[2], 2*z[7], rtol=0.05, atol=1e-9)
    # sphere: no astigmatism, a cap of the same sphere
    if K == 0:
        assert abs(z[4]) < 1e-9 and np.isclose(g.R_tangential, R)


def test_off_axis_parabola_on_axis_perfect():
    s = oa.off_axis_parabola(500.0, 150.0, 100.0, field_angle=DEG)
    assert s.rms_opd(0.0) < 1e-4
    # the centre of the stop: the chief ray hits the mirror 150 mm off the axis
    b = s.trace(s.rays(0.0, 0.0, 0.0))
    assert np.isclose(b.points[2, 0, 1], 150.0, atol=1e-6)


def test_solve_conics_cook_tma():
    s = oa.cook_tma(efl=600.0, f_number=4.0, field_angle=4*DEG)
    assert np.allclose(s.seidel().total[:3], 0.0, atol=1e-9)
    assert np.isclose(s.paraxial().efl, 600.0)
    # flat field: zero Petzval sum
    assert abs(s.seidel().total[3]) < 1e-9
    assert s.rms_opd(0.0) < 0.05


def test_beam_clearance():
    # aperture offset 2, field bias 3 deg: unobscured
    s = oa.cook_tma(600.0, 4.0, 4*DEG, prescription=(-2/3, 0.2, -1/3, 4/9), aperture_offset=-2.0)
    F = oa.rectangular_field((0, 3*DEG), (1.5*DEG, 0.5*DEG), (3, 3))
    c = oa.beam_clearance(s, F)
    assert c.distance.shape == (len(c.obstacles), len(c.beams))
    assert np.min(c.distance) > 5.0 and np.all(c.blocked == 0)
    # the coaxial parent is obscured by its secondary
    p = oa.cook_tma(600.0, 4.0, 4*DEG, prescription=(-2/3, 0.2, -1/3, 4/9))
    c = oa.beam_clearance(p, oa.rectangular_field((0, 0), (DEG, DEG), (3, 3)))
    k = c.names.index("M2 (stop)")
    assert c.distance[k, 0] < 0 and c.blocked[k, 0] > 0


def test_zernike_surface_centre():
    """A centred Zernike departure equals the same polynomial shifted."""
    s = rt.ZernikeSurface(-100.0, zernike={5: 1e-3, 8: 2e-3}, norm_radius=10.0, centre=(0, 30.0))
    s0 = rt.ZernikeSurface(-100.0, zernike={5: 1e-3, 8: 2e-3}, norm_radius=10.0)
    x, y = np.array([1.0, -2.0, 0.5]), np.array([31.0, 25.0, 38.0])
    assert np.allclose(s.departure(x, y), s0.departure(x, y - 30.0))
    assert np.allclose(s.departure_gradient(x, y), s0.departure_gradient(x, y - 30.0))
    s.set_zernike({5: 0.0, 9: 1e-4})
    assert dict(s.zernike) == {5: 0.0, 8: 2e-3, 9: 1e-4}
    assert "centre" in repr(s)


def test_freeform_mirrors_and_variables():
    s = oa.cook_tma(600.0, 4.0, 4*DEG, prescription=(-2/3, 0.2, -1/3, 4/9), aperture_offset=-2.0)
    F = oa.rectangular_field((0, 3*DEG), (1.5*DEG, 0.5*DEG), (3, 3))
    ff = oa.freeform_mirrors(s, F)
    assert all(isinstance(ff.surfaces[i], rt.ZernikeSurface) for i in (0, 2, 3))
    # nothing changed yet
    assert np.allclose(oa.field_rms(ff, F[:2], N=16), oa.field_rms(s, F[:2], N=16))
    v = lo.zernike_terms(ff, [3], [5])[0]
    v.set(ff, 1e-3)
    assert v.get(ff) == 1e-3 and dict(ff.surfaces[3].zernike)[5] == 1e-3
    with pytest.raises(TypeError):
        lo.zernike_terms(s, [0], [5])
    t = lo.Tilt(0, 0)
    t.set(ff, 1e-3)
    assert ff.surfaces[0].tilt == (1e-3, 0.0, 0.0)
    d = lo.Decenter(2, 1)
    d.set(ff, 0.5)
    assert ff.surfaces[2].decenter == (0.0, 0.5)
    it = lo.ImageTilt(0)
    it.set(ff, 0.01)
    assert it.get(ff) == 0.01 and ff.image_tilt[0] == 0.01


def test_operands_and_a_freeform_step():
    """Damped least squares with Zernike terms improves the off-axis TMA."""
    s = oa.cook_tma(600.0, 4.0, 4*DEG, prescription=(-2/3, 0.2, -1/3, 4/9), aperture_offset=-2.0)
    F = oa.rectangular_field((0, 3*DEG), (1.5*DEG, 0.5*DEG), (3, 3))
    fl = oa.focal_length(600.0, (0, 3*DEG), tolerance=1.0)(s)
    # the image scale at 3 deg off the axis differs from the paraxial focal length
    # (anamorphic, about 2 % in x and 6 % in y):
    assert np.all(np.abs(fl) < 60) and abs(fl[1]) > abs(fl[0])
    w = oa.field_wavefront(F, tolerance=1.0)(s)
    assert w.size == 9*len(oa.pupil_rings((0.5, 0.85, 1.0), 8)[0])
    assert np.all(oa.clearance_penalty(F, minimum=5.0)(s) == 0)
    assert np.any(oa.clearance_penalty(F, minimum=100.0)(s) > 0)
    ff = oa.freeform_mirrors(s, F)
    v = lo.zernike_terms(ff, [3], [5, 8]) + [lo.Thickness(3)]
    opt = lo.DampedLeastSquares(ff, v, [oa.field_wavefront(F, tolerance=0.01)])
    m0 = opt.merit()
    res = opt.run(max_iterations=4)
    assert res.merit < 0.8*m0


def test_footprints_and_layout():
    s = oa.cook_tma(600.0, 4.0, 4*DEG, prescription=(-2/3, 0.2, -1/3, 4/9), aperture_offset=-2.0)
    F = oa.rectangular_field((0, 3*DEG), (1.5*DEG, 0.5*DEG), (3, 3))
    fp = oa.footprints(s, F)
    # the stop (local frame, decentred): the beams fill it, centred
    assert np.allclose(fp[1].centre, 0, atol=1e-6)
    assert np.isclose(fp[1].radius, s.surfaces[1].semi_diameter, rtol=0.02)
    # primary and tertiary on opposite sides of the axis
    assert fp[0].centre[1]*fp[3].centre[1] < 0
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    oa.draw_layout(s, [(0, 3*DEG)], ax=ax)
    Z = oa.field_zernikes(s, F, 9, N=16)
    oa.plot_full_field(ax, F, Z[:, 4] + 1j*Z[:, 5], "astigmatism")
    oa.plot_full_field(ax, F, Z[:, 6] + 1j*Z[:, 7], "coma")
    oa.plot_full_field(ax, F, oa.field_rms(s, F, N=16), "rms")
    plt.close(fig)


def test_pupil_proximity_paraxial_limit_and_ring_field():
    # two mirrors, stop on the first, small aperture: the real-ray proximity
    # of the edge fields +-h is the paraxial |y|/(|y| + |ybar|)
    s = rt.System([rt.Surface(R=-400, thickness=-150, mirror=True),
                   rt.Surface(R=-250, thickness=300, mirror=True)], 0, 550e-6,
                  object_distance=900, object_height=50, entrance_pupil_radius=2.0).at_paraxial_focus()
    p = s.paraxial()
    y, yb = np.abs(p.marginal[:, 0]), np.abs(p.chief[:, 0])
    r = oa.pupil_proximity(s, [(0.0, -50.0), (0.0, 50.0)], rings=(1.0,), spokes=32)
    assert np.allclose(r.p, y/(y + yb), atol=2e-3)
    assert np.isclose(r.p[0], 1.0) and np.allclose(r.px, 1.0)
    # a short arc of a ring field shifts the footprints much less across (y)
    # than along the arc (x): the second mirror is closer to a pupil in y
    arc = oa.ring_field(40.0, np.radians(30), width=2.0, n_arc=3, n_radial=2)
    r = oa.pupil_proximity(s, arc, rings=(1.0,), spokes=32)
    assert r.py[1] > r.px[1] and r.shift[1, 0] > r.shift[1, 1]
    assert 0 < r.p[1] < 1 and np.isclose(r.p[1], r.px[1], rtol=0.05)
    with pytest.raises(ValueError):
        oa.pupil_proximity(s, [(0.0, 0.0)])
