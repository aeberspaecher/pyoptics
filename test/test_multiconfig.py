#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the optimization of several configurations of one lens
(`lens_optimization.MultiConfiguration`,
notebook 153, multi-configuration optimization): overrides, shared and per
configuration variables, operands, solves and constraints in some
configurations, and that one configuration gives the same as the plain
optimizer.
"""

import numpy as np
import pytest

from pyoptics import lens_optimization as lo
from pyoptics import raytrace as rt
from pyoptics import zoom
from pyoptics.delano import thick_lens_curvatures


WL = 0.55e-3  # mm
N_GLASS = 1.5168
F_LINE, D_LINE, C_LINE = 486.1327e-6, 587.5618e-6, 656.2725e-6


def singlet(shape=0.0, f=100.0, t=4.0, sd=8.0, object_distance=np.inf):
    c1, c2 = thick_lens_curvatures(1/f, N_GLASS, t, shape)
    surfaces = [rt.Surface(1/c1 if c1 else np.inf, t, n_after=N_GLASS, semi_diameter=sd),
                rt.Surface(1/c2 if c2 else np.inf, f, n_after=1.0)]
    system = rt.System(surfaces, 0, WL, object_distance=object_distance, field_angle=0.0)
    lo.solve_paraxial_focus()(system)

    return system


def doublet(f=100.0):
    crown, flint = (1.5168, 64.17), (1.62005, 36.43)
    (n1, V1), (n2, V2) = crown, flint
    p1, p2 = V1/(V1 - V2)/f, -V2/(V1 - V2)/f
    c1 = p1/(n1 - 1)/2
    c2 = -c1
    c3 = c2 - p2/(n2 - 1)
    S = [rt.Surface(1/c1, 5.0, material=crown), rt.Surface(1/c2, 2.5, material=flint),
         rt.Surface(1/c3, 95.0, n_after=1.0)]
    system = rt.System(S, 0, D_LINE, field_angle=np.deg2rad(1.0), entrance_pupil_radius=10.0)

    return system.at_paraxial_focus()


def shape_factor(system):
    c1, c2 = system.surfaces[0].c, system.surfaces[1].c

    return (c1 + c2)/(c1 - c2)


def test_configure_overrides():
    s = singlet(0.5)
    c = lo.configure(s, {1: 50.0, "wavelength": 0.6e-3, "object_distance": 500.0,
                         "max_field": 2.0, "entrance_pupil_radius": 3.0,
                         "surfaces": {0: {"conic": -0.5}}})
    assert c.surfaces[1].thickness == 50.0
    assert c.wavelength == 0.6e-3 and c.object_distance == 500.0 and c.max_field == 2.0
    assert c.surfaces[0].conic == -0.5
    assert np.isclose(c.surfaces[0].semi_diameter, 3.0)  # stop in front: EP = stop
    # the original is untouched
    assert s.surfaces[1].thickness != 50.0 and s.surfaces[0].conic == 0.0
    c2 = lo.configure(s, lambda system: setattr(system, "wavelength", 0.7e-3))
    assert c2.wavelength == 0.7e-3 and s.wavelength == WL
    with pytest.raises(ValueError):
        lo.configure(s, {"colour": 1})


def test_one_configuration_is_the_plain_optimizer():
    system = doublet()
    ops = [lo.seidel("W040", 0.0, 0.01), lo.seidel("W131", 0.0, 0.01)]
    efl = lo.Constraint(lo.paraxial_value("efl"), 100.0, 100.0, 1e-3, "efl")
    plain = lo.DampedLeastSquares(system, lo.curvatures(system), ops, [efl],
                                  solves=[lo.solve_paraxial_focus()])
    multi = lo.DampedLeastSquares(lo.MultiConfiguration([system]), lo.curvatures(system), ops,
                                  [efl], solves=[lo.solve_paraxial_focus()])
    assert all(isinstance(v, lo.Shared) for v in multi.variables)
    assert np.allclose(plain.residuals(), multi.residuals())
    J0, A0 = plain.jacobian()[:2]
    J1, A1 = multi.jacobian()[:2]
    assert np.allclose(J0, J1, rtol=1e-6) and np.allclose(A0, A1, rtol=1e-6)
    r0, r1 = plain.run(max_iterations=10), multi.run(max_iterations=10)
    assert np.allclose(r0.x, r1.x) and np.isclose(r0.merit, r1.merit)
    assert isinstance(r1.system, lo.MultiConfiguration)


def test_shared_and_per_configuration_variables():
    mc = lo.MultiConfiguration.from_overrides(singlet(0.5), [{}, {1: 90.0}], names=["a", "b"])
    shared = lo.Shared(lo.Curvature(0))
    shared.set(mc, 0.02)
    assert all(s.surfaces[0].c == 0.02 for s in mc)
    t0, t1 = lo.per_configuration(lo.Thickness(1), 2)
    t1.set(mc, 80.0)
    assert mc[1].surfaces[1].thickness == 80.0 and mc[0].surfaces[1].thickness != 80.0
    assert t1.name == "t1@1" and t1.get(mc) == 80.0
    # the column of a variable of one configuration is zero for the residuals of the other
    opt = lo.DampedLeastSquares(mc, [lo.Curvature(0), t0, t1],
                                [lo.ray_errors(fields=(0.0,), tolerance=1e-3)])
    J = opt.jacobian()[0]
    n = J.shape[0]//2
    assert np.all(J[:n, 2] == 0) and np.any(J[n:, 2] != 0)
    assert np.all(J[n:, 1] == 0) and np.any(J[:n, 1] != 0)
    assert np.all(np.any(J[:n, :1] != 0, axis=0)) and np.all(np.any(J[n:, :1] != 0, axis=0))


def test_operands_in_some_configurations_and_weights():
    mc = lo.MultiConfiguration.from_overrides(singlet(0.5), [{}, {1: 90.0}], weights=[1.0, 3.0])
    op = lo.ray_errors(fields=(0.0,), tolerance=1e-3)
    single = op(mc[1])
    only_b = lo.DampedLeastSquares(mc, [lo.Curvature(0)], [lo.in_configs(op, [1])])
    assert np.allclose(only_b.residuals(), 3*single)
    both = lo.DampedLeastSquares(mc, [lo.Curvature(0)], [op])
    assert np.allclose(both.residuals(), np.concatenate([op(mc[0]), 3*single]))
    m = both.merit_by_configuration()
    assert np.isclose(m.sum(), both.merit()) and np.isclose(m[1], 9*np.sum(single**2))
    assert "config 2" in both.report()
    # constraints are not weighted, and a plain one holds in every configuration
    t = lo.Constraint(lo.center_thickness(1), upper=200.0, name="t1")
    opt = lo.DampedLeastSquares(mc, [lo.Curvature(0)], [op], [t])
    assert np.allclose(opt.constraint_values()[0], [mc[0].surfaces[1].thickness, 90.0])
    # solves in one configuration
    mc = lo.MultiConfiguration.from_overrides(singlet(0.5), [{1: 95.0}, {1: 90.0}])
    opt = lo.DampedLeastSquares(mc, [lo.Curvature(0)], [op],
                                solves=[lo.in_configs(lo.solve_paraxial_focus(), [1])])
    s0, s1 = opt.system.configs
    assert np.isclose(s1.surfaces[1].thickness, s1.paraxial().image_distance)
    assert s0.surfaces[1].thickness == 95.0


def test_two_object_distances_share_one_bending():
    # spherical aberration of a singlet focused at infinity and at 400 mm: the bending
    # that is best for both lies between the two best bendings
    def best(mc):
        opt = lo.DampedLeastSquares(mc, [lo.Curvature(0)], [lo.seidel("W040", 0.0, 1.0)],
                                    solves=[lo.solve_curvature_for_efl(1, 100.0),
                                            lo.solve_paraxial_focus()])
        result = opt.run(max_iterations=30)
        return result, opt

    far, near = singlet(0.0), singlet(0.0, object_distance=400.0)
    r_far = best(lo.MultiConfiguration([far]))[0]
    r_near = best(lo.MultiConfiguration([near]))[0]
    r_both, opt = best(lo.MultiConfiguration([far, near]))
    X_far, X_near, X_both = (shape_factor(r.system[0]) for r in (r_far, r_near, r_both))
    assert X_near < X_both < X_far
    # the same lens in both configurations, each focused for its object
    a, b = r_both.system
    assert a.surfaces[0].c == b.surfaces[0].c and a.surfaces[1].c == b.surfaces[1].c
    assert b.surfaces[1].thickness > a.surfaces[1].thickness
    for s in (a, b):
        assert np.isclose(s.paraxial().image_distance, s.surfaces[1].thickness)
    # each configuration is worse than the lens designed for it alone
    m = opt.merit_by_configuration()
    assert m[0] > r_far.merit and m[1] > r_near.merit


def test_zoom_positions_round_trip():
    Z = zoom.olympus_zoom_us7583450(glasses="model")
    angles = np.arctan(10.8/Z.focal_lengths)
    mc = lo.MultiConfiguration.from_zoom(Z, field_angle=angles)
    assert len(mc) == 3 and mc.zoom_gaps == sorted(Z.variable)
    for k, s in enumerate(mc):
        assert np.isclose(s.paraxial().efl, Z.focal_lengths[k], atol=0.02)
        assert np.isclose(s.max_field, angles[k])
    table = mc.thickness_table(mc.zoom_gaps)
    for i, v in Z.variable.items():
        assert np.allclose(table[i], v)
    Z2 = mc.to_zoom(Z)
    for k in range(3):
        assert np.isclose(Z2.system(k).paraxial().efl, mc[k].paraxial().efl)
        assert np.isclose(Z2.system(k).paraxial().f_number, Z.system(k).paraxial().f_number)


def test_zoom_spacings_per_position():
    # a two group zoom of thick singlets: one shared bending of the rear lens, the
    # separation per position held at the focal length by a constraint
    S = [rt.Surface(-60.0, 2.0, n_after=N_GLASS), rt.Surface(60.0, 30.0, n_after=1.0),
         rt.Surface(np.inf, 0.0, semi_diameter=3.0),
         rt.Surface(30.0, 4.0, n_after=N_GLASS), rt.Surface(-30.0, 40.0, n_after=1.0)]
    system = rt.System(S, 2, WL, field_angle=0.05, entrance_pupil_radius=2.0)
    mc = lo.MultiConfiguration.from_overrides(system, [{1: 30.0}, {1: 12.0}])
    targets = [mc[0].paraxial().efl, mc[1].paraxial().efl]
    gaps = lo.per_configuration(lo.Thickness(1, bounds=(1.0, None)), 2)
    efl = [lo.Constraint(lo.in_configs(lo.paraxial_value("efl"), [k]), targets[k], targets[k],
                         1e-4, "efl{}".format(k)) for k in range(2)]
    opt = lo.DampedLeastSquares(mc, [lo.Curvature(3), lo.Curvature(4)] + gaps,
                                [lo.ray_errors(fields=(0.0, 1.0), tolerance=1e-3)], efl,
                                solves=[lo.solve_paraxial_focus()])
    before = opt.merit()
    result = opt.run(max_iterations=10)
    assert result.merit < before
    for k, s in enumerate(result.system):
        assert np.isclose(s.paraxial().efl, targets[k], atol=1e-6)
    a, b = result.system
    assert a.surfaces[3].c == b.surfaces[3].c
    assert a.surfaces[1].thickness != b.surfaces[1].thickness


def test_lens_optimizer_with_configurations():
    system = singlet(0.0)
    configs = [{}, {"object_distance": 400.0, 1: 140.0}]
    op = lo.ray_errors(fields=(0.0,), tolerance=1e-3)
    opt = lo.LensOptimizer(system, [op], curvatures=[0], thicknesses=[], configurations=configs)
    single = lo.LensOptimizer(system, [op], curvatures=[0], thicknesses=[])
    assert len(opt.residuals()) == 2*len(single.residuals())
    near = opt.configuration_systems()[1]
    assert near.object_distance == 400.0 and near.surfaces[1].thickness == 140.0
    # a thickness that a configuration sets is no default variable
    assert 1 not in lo.LensOptimizer(system, [op], configurations=configs).thicknesses
    weighted = lo.LensOptimizer(system, [op], curvatures=[0], thicknesses=[],
                                configurations=configs, configuration_weights=[0.0, 1.0])
    assert np.allclose(weighted.residuals()[:len(single.residuals())], 0.0)
    before = opt.cost()
    result = opt.run(max_nfev=50)
    assert result.cost < before


def test_snap_glasses_in_every_configuration():
    system = doublet()
    efl = lo.Constraint(lo.paraxial_value("efl"), 100.0, 100.0, 1e-3, "efl")
    mc = lo.MultiConfiguration.from_overrides(system, [{}, {"wavelength": F_LINE}])
    opt = lo.DampedLeastSquares(mc, lo.curvatures(system) + lo.glass(0), [lo.seidel("W040", 0.0, 0.1)],
                                [efl, lo.glass_map_constraint(0)],
                                solves=[lo.solve_paraxial_focus()])
    opt.run(max_iterations=5)
    chosen = opt.snap_glasses()
    assert set(chosen) == {0}
    assert all(s.surfaces[0].label == chosen[0] for s in opt.system)
    assert not any(isinstance(lo._unwrap(v), (lo.GlassIndex, lo.GlassAbbe)) for v in opt.variables)


def test_a_variable_at_its_bound_keeps_the_constraints():
    # the operand pulls the thickness far below its bound: the step puts it on the
    # bound and solves the curvatures for the focal length with that thickness
    system = singlet(0.5, t=5.05)
    efl = lo.Constraint(lo.paraxial_value("efl"), 100.0, 100.0, 1e-3, "efl")
    opt = lo.DampedLeastSquares(system, [lo.Curvature(0), lo.Curvature(1),
                                         lo.Thickness(0, bounds=(5.0, None))],
                                [lambda s: np.array([(s.surfaces[0].thickness - 2.0)/1e-2]),
                                 lo.seidel("W040", 0.0, 1.0)], [efl])
    result = opt.run(max_iterations=1)
    s = result.system
    assert s.surfaces[0].thickness == 5.0
    assert abs(s.paraxial().efl - 100.0) < 0.05  # to first order


def test_analytic_jacobian_of_several_configurations():
    # shared curvatures and a focus per configuration: the analytic Jacobian
    # agrees with forward differences, also with weights and an operand in
    # one configuration only
    mc = lo.MultiConfiguration.from_overrides(doublet(), [{}, {"object_distance": 2000.0}],
                                              weights=[1.0, 2.0])
    V = lo.curvatures(mc[0]) + lo.per_configuration(lo.Thickness(2), 2)
    ops = [lo.wavefront(fields=(0.0, 1.0), wavelengths=(F_LINE, C_LINE), tolerance=0.05),
           lo.in_configs(lo.ray_errors(fields=(0.0, 0.7), tolerance=1e-3), [1])]
    J = {}
    for jac in ("2-point", "analytic"):
        opt = lo.DampedLeastSquares(mc, V, ops, jac=jac)
        J[jac], _, r, _ = opt.jacobian()
    assert np.allclose(r, opt.residuals())
    scale = np.abs(J["2-point"]).max(axis=0)
    assert np.allclose(J["analytic"]/scale, J["2-point"]/scale, atol=2e-4)
    # the focus of one configuration does not act on the other
    n0 = len(ops[0](mc[0]))
    assert np.all(J["analytic"][:n0, -1] == 0) and np.any(J["analytic"][:n0, -2] != 0)
    assert np.all(J["analytic"][n0:, -2] == 0) and np.any(J["analytic"][n0:, -1] != 0)
