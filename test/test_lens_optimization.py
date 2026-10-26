#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the least squares lens optimizer: the bending of a singlet for
minimum spherical aberration, the solves for length and NA, and the
operands.
"""

import numpy as np

from pyoptics import lens_optimization as lo
from pyoptics import raytrace as rt
from pyoptics.delano import thick_lens_curvatures


WL = 0.55e-3  # mm
N_GLASS = 1.5168


def singlet(shape=0.0, f=100.0, t=4.0, sd=8.0):
    """Singlet in air, object at infinity, stop at the first surface."""

    c1, c2 = thick_lens_curvatures(1/f, N_GLASS, t, shape)
    surfaces = [rt.Surface(1/c1 if c1 else np.inf, t, n_after=N_GLASS, semi_diameter=sd),
                rt.Surface(1/c2 if c2 else np.inf, f, n_after=1.0)]

    return rt.System(surfaces, 0, WL, field_angle=0.0).at_paraxial_focus()


def test_singlet_bending_for_minimum_spherical_aberration():
    system = singlet(-1.0)  # plano-convex the wrong way round
    opt = lo.LensOptimizer(system, [lo.ray_errors(fields=(0.0,), tolerance=1e-4),
                                    lo.paraxial_target("efl", 100.0, 1e-3)],
                           thicknesses=[1], length=None)
    before = opt.cost()
    result = opt.run(max_nfev=200)
    assert result.cost < 0.05*before
    S = result.system.surfaces
    shape = (S[0].c + S[1].c)/(S[0].c - S[1].c)
    # thin lens theory, object at infinity: X = 2 (n**2 - 1)/(n + 2) = 0.74
    # for minimum spherical aberration (the spot also balances it with defocus)
    assert 0.5 < shape < 1.0
    assert np.isclose(result.system.paraxial().efl, 100.0, rtol=1e-4)


def test_length_and_object_na_are_kept():
    surfaces = [rt.Surface(80.0, 6.0, n_after=N_GLASS), rt.Surface(-80.0, 20.0, n_after=1.0),
                rt.Surface(np.inf, 60.0, semi_diameter=5.0)]
    system = rt.System(surfaces, 2, WL, object_distance=100.0, object_height=2.0)
    opt = lo.LensOptimizer(system, [lo.ray_errors(tolerance=1e-3)], thicknesses=[1],
                           object_distance=True, length=200.0, object_na=0.05)
    result = opt.run(stages=[(0.5, 0.5), (1.0, 1.0)], max_nfev=30)
    s = result.system
    assert np.isclose(s.object_distance + s.image_z, 200.0)
    p = s.paraxial()
    assert np.isclose(p.na*abs(p.magnification), 0.05, rtol=1e-6)
    assert s.max_field == 2.0
    assert len(result.stages) == 2


def test_clearances():
    system = singlet(0.0, t=2.0, sd=8.0)
    assert np.all(lo.clearances(min_edge=1.0)(system) == 0)
    # an equiconvex f = 100 lens of 2 mm has an edge of 1.4 mm at 8.4 mm
    assert np.any(lo.clearances(min_edge=2.0)(system) > 0)


def test_telecentricity_and_distortion_operands():
    # 4f relay of ideal lenses, stop in the common focal plane: telecentric
    # and free of distortion with magnification -1
    f = 50.0
    surfaces = [rt.IdealLens(f, f), rt.Surface(np.inf, f, semi_diameter=5.0),
                rt.IdealLens(f, f)]
    system = rt.System(surfaces, 1, WL, object_distance=f, object_height=3.0)
    assert np.allclose(lo.object_telecentricity()(system), 0)
    assert np.allclose(lo.image_telecentricity()(system), 0, atol=1e-6)
    assert np.allclose(lo.distortion(-1.0)(system), 0, atol=1e-3)
    assert np.all(np.abs(lo.distortion(-0.9)(system)) > 1e3)


def test_aspheres_correct_spherical_aberration():
    # bending alone leaves the spherical aberration of a fast singlet; an
    # r**4 and r**6 asphere on its first surface removes it
    system = singlet(0.7, f=50.0, t=6.0, sd=10.0)
    ops = [lo.ray_errors(fields=(0.0,), tolerance=1e-4)]
    bent = lo.LensOptimizer(system, ops, thicknesses=[1]).run(max_nfev=100)
    opt = lo.LensOptimizer(system, ops, thicknesses=[1], aspheres=[0])
    asph = opt.run(max_nfev=200)
    assert len(opt.x) == 5
    assert asph.cost < 1e-3*bent.cost
    assert asph.system.rms_opd(N=32) < 0.01
