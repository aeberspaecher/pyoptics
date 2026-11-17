#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the tools for the merit function landscape (`lens_landscape`,
notebook 154, global optimization and saddle points): on an analytic
landscape with two minima and a saddle point between them, and the
saddle-point construction on a real lens.
"""

import numpy as np
import pytest

from pyoptics import lens_landscape as ll
from pyoptics import lens_optimization as lo
from pyoptics import raytrace as rt


# --- an analytic landscape driven by the real optimizer --------------------------

class Point:
    """A stand-in for a system: two numbers."""

    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y

    def copy(self):
        return Point(self.x, self.y)


class Coordinate(lo.Variable):
    def __init__(self, name, bounds=(-3.0, 3.0)):
        super().__init__(bounds)
        self.name = name

    def get(self, system):
        return getattr(system, self.name)

    def set(self, system, value):
        setattr(system, self.name, value)

    def typical(self, system):
        return 1.0


def double_well(p):
    """Merit (x^2 - 1)^2 + (y - 0.2 x^2)^2 + 0.09 (x + 1)^2: minima at
    x = -1 (merit 0) and near x = +1 (higher), a saddle point near x = 0."""

    return np.array([p.x**2 - 1.0, p.y - 0.2*p.x**2, 0.3*(p.x + 1.0)])


def well_optimizer(x=0.5, y=0.5):
    return lo.DampedLeastSquares(Point(x, y), [Coordinate("x"), Coordinate("y")], [double_well])


def test_local_search_and_counting():
    opt = well_optimizer()
    run = ll.local_search(opt, [0.8, 0.3])
    assert run.success
    assert run.x[0] == pytest.approx(1.0, abs=0.1)
    assert run.merit > 0.1
    assert run.evaluations > run.iterations
    assert opt.x == pytest.approx([0.5, 0.5])  # the optimizer is not changed


def test_multistart_and_clusters():
    opt = well_optimizer()
    starts = ll.random_starts(opt, 20, bounds=[(-2, 2), (-2, 2)], rng=1)
    assert starts.shape == (20, 2)
    runs = ll.multistart(opt, starts)
    minima = ll.cluster_minima(runs, opt, xtol=0.01)
    assert len(minima) == 2
    assert sum(m.count for m in minima) == 20
    assert minima[0].merit < minima[1].merit
    assert minima[0].x[0] < 0 < minima[1].x[0]


def test_stationarity_minimum_and_saddle():
    opt = well_optimizer()
    low = ll.local_search(opt, [-1.2, 0.0])
    st = ll.stationarity(opt, low.x)
    assert st.morse_index == 0
    assert np.max(np.abs(st.gradient)) < 1e-3
    x, st = ll.refine_stationary(opt, [0.05, 0.02], iterations=30)
    assert st.morse_index == 1
    assert np.max(np.abs(st.gradient)) < 1e-6
    assert abs(x[0]) < 0.2


def test_eigenvector_following_onto_saddle():
    opt = well_optimizer()
    # from far off the saddle point, where Newton's method would not get there
    x, st = ll.refine_stationary(opt, [0.5, -0.4], iterations=60, index=1, max_step=0.2)
    assert st.morse_index == 1
    assert np.max(np.abs(st.gradient)) < 1e-6
    saddle, _ = ll.refine_stationary(opt, [0.05, 0.02], iterations=30)
    assert x == pytest.approx(saddle, abs=1e-5)


def test_descent_from_saddle_reaches_both_minima():
    opt = well_optimizer()
    x, st = ll.refine_stationary(opt, [0.05, 0.02], iterations=30)
    runs, _ = ll.descend_from_saddle(opt, x, step=0.1, stationary=st)
    assert sorted(np.sign([r.x[0] for r in runs])) == [-1, 1]


def test_minimum_energy_path_climbs_to_saddle():
    opt = well_optimizer()
    a = ll.local_search(opt, [-1.0, 0.2]).x
    b = ll.local_search(opt, [1.0, 0.2]).x
    path = ll.minimum_energy_path(opt, a, b, images=9, iterations=300, step=0.05)
    saddle, _ = ll.refine_stationary(opt, [0.05, 0.02], iterations=30)
    k = path.saddle
    assert 0 < k < 8
    assert path.x[k] == pytest.approx(saddle, abs=0.05)
    assert path.merit[k] == pytest.approx(ll.merit_at(opt, saddle), rel=0.02)
    assert path.merit[k] > max(path.merit[0], path.merit[-1])


def test_basin_hopping_finds_lower_minimum():
    opt = well_optimizer()
    res = ll.basin_hopping(opt, [1.0, 0.0], hops=20, step=1.5, rng=3)
    assert res.x[0] < 0
    assert res.merit == pytest.approx(min(r.merit for r in res.runs))
    assert len(res.current) == len(res.accepted) == len(res.evaluations) == 21
    assert np.all(np.diff(res.evaluations) > 0)


# --- the saddle-point construction on a real lens -----------------------------------

WL = 587.5618e-6


def two_lens_optimizer(system):
    ops = [lo.ray_errors(fields=(0.0, 1.0), rings=(0.7, 1.0), spokes=4, tolerance=0.01)]
    solves = [lo.solve_curvature_for_efl(len(system.surfaces) - 1, 100.0),
              lo.solve_paraxial_focus()]
    n = len(system.surfaces)
    variables = [lo.Curvature(i, (-0.08, 0.08)) for i in range(n - 1)]

    return lo.DampedLeastSquares(system, variables, ops, (), solves)


def singlet_minimum():
    S = [rt.Surface(60.0, 5.0, material=lo.GLASSES["N-BK7"]),
         rt.Surface(-300.0, 95.0, n_after=1.0)]
    system = rt.System(S, 0, WL, field_angle=np.deg2rad(2.0), entrance_pupil_radius=6.0,
                       ray_aiming=False)
    opt = two_lens_optimizer(system)
    run = ll.local_search(opt, opt.x)
    opt.system = opt.system_at(run.x)

    return opt


def test_insert_null_lens_changes_nothing():
    opt = singlet_minimum()
    s = opt.system
    new, (i, j) = ll.insert_null_lens(s, 0, lo.GLASSES["N-F2"], 0.02, distance=2.0)
    assert len(new.surfaces) == 4 and (i, j) == (1, 2)
    assert new.surfaces[0].thickness == 2.0
    assert new.surfaces[2].thickness == pytest.approx(3.0)
    assert lo.DampedLeastSquares(new, [], opt.operands).merit() == pytest.approx(opt.merit(),
                                                                                 rel=1e-9)
    p_old, p_new = s.paraxial(), new.paraxial()
    assert p_new.efl == pytest.approx(p_old.efl, rel=1e-12)


def test_contact_construction_is_a_saddle_point():
    opt = singlet_minimum()
    assert ll.stationarity(opt).morse_index == 0
    # a lens of the medium in front (air) in contact with the first
    # surface: an air gap of zero thickness that splits the singlet
    new, (i, j) = ll.insert_null_lens(opt.system, 0)
    assert new.surfaces[i].c == new.surfaces[0].c
    ext = two_lens_optimizer(new)
    st = ll.stationarity(ext)
    scaled = np.abs(st.gradient)/np.max(np.abs(st.hessian))
    assert np.max(scaled) < 1e-3
    assert st.morse_index == 1


def test_null_lens_curvatures_are_stationary():
    opt = singlet_minimum()
    new, ij = ll.insert_null_lens(opt.system, 0, lo.GLASSES["N-F2"], 0.0, distance=3.0)
    ext = two_lens_optimizer(new)
    roots = ll.null_lens_curvatures(ext, ij, np.linspace(-0.08, 0.08, 9))
    assert len(roots) >= 1
    for c in roots:
        assert abs(ll.null_lens_gradient(ext, ij, [c])[0]) < 1e-6*np.max(
            np.abs(ll.null_lens_gradient(ext, ij, [-0.08, 0.08])))
