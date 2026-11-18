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


def test_real_chief_ray_operands():
    # with aim=True the operands use the chief ray through the centre of the
    # stop, as traced with ray aiming, also when the system does not aim
    system = rt.caldwell_krf_lens()
    system.ray_aiming = False
    fields = (0.5, 1.0)
    tol = 1e-3
    tele_img = lo.image_telecentricity(fields, tol, aim=True)(system)
    tele_obj = lo.object_telecentricity(tol, aim=True, fields=fields)(system)
    dist = lo.distortion(-0.25, fields, 1e-5, aim=True)(system)
    aimed = system.copy()
    aimed.ray_aiming = True
    h = system.max_field
    for k, f in enumerate(fields):
        rays = aimed.rays(f*h, 0.0, 0.0)
        d = np.reshape(rays.directions, (-1, 3))[0]
        I, D = aimed._image_hits(aimed.trace(rays, clip=False))
        assert np.isclose(tele_obj[k], d[1]/d[2]/tol, atol=1e-6)
        assert np.isclose(tele_img[k], D[0, 1]/D[0, 2]/tol, atol=1e-6)
        assert np.isclose(dist[k], (I[0, 1] + 0.25*f*h)/1e-5, atol=1e-3)
    # the paraxially aimed chief ray differs (pupil aberration)
    assert not np.allclose(lo.image_telecentricity(fields, tol)(system), tele_img, atol=0.1)


# --- DampedLeastSquares --------------------------------------------------------

F_LINE, D_LINE, C_LINE = 486.1327e-6, 587.5618e-6, 656.2725e-6


def doublet(crown=(1.5168, 64.17), flint=(1.62005, 36.43), f=100.0):
    """Cemented doublet from the thin lens achromat condition."""

    (n1, V1), (n2, V2) = crown, flint
    p1, p2 = V1/(V1 - V2)/f, -V2/(V1 - V2)/f
    c1 = p1/(n1 - 1)/2
    c2 = -c1
    c3 = c2 - p2/(n2 - 1)
    S = [rt.Surface(1/c1, 5.0, material=crown), rt.Surface(1/c2, 2.5, material=flint),
         rt.Surface(1/c3, 95.0, n_after=1.0)]
    system = rt.System(S, 0, D_LINE, field_angle=np.deg2rad(1.0), entrance_pupil_radius=10.0)

    return system.at_paraxial_focus()


def test_dls_achromat_constraints_hold_exactly():
    system = doublet()
    efl = lo.Constraint(lo.paraxial_value("efl"), 100.0, 100.0, 1e-3, "efl")
    colour = lo.Constraint(lambda s: lo.focal_shift(F_LINE, C_LINE, tolerance=1.0)(s)[0],
                           0.0, 0.0, 1e-4, "F-C")
    opt = lo.DampedLeastSquares(system, lo.curvatures(system), [lo.seidel("W040", 0.0, 0.01)],
                                [efl, colour], solves=[lo.solve_paraxial_focus()])
    result = opt.run()
    assert result.success
    s = result.system
    assert np.isclose(s.paraxial().efl, 100.0, atol=1e-6)
    zF = s.with_wavelength(F_LINE).paraxial().image_distance
    zC = s.with_wavelength(C_LINE).paraxial().image_distance
    assert abs(zF - zC) < 1e-7
    # one free variable left, spherical aberration corrected
    assert abs(s.seidel().W040) < 1e-6
    # the image plane follows the paraxial focus
    assert np.isclose(s.surfaces[-1].thickness, s.paraxial().image_distance)


def test_dls_model_glasses_make_an_aplanatic_doublet():
    system = doublet()
    efl = lo.Constraint(lo.paraxial_value("efl"), 100.0, 100.0, 1e-3, "efl")
    colour = lo.Constraint(lambda s: lo.focal_shift(F_LINE, C_LINE, tolerance=1.0)(s)[0],
                           0.0, 0.0, 1e-4, "F-C")
    ops = [lo.seidel("W040", 0.0, 0.01), lo.seidel("W131", 0.0, 0.01)]
    fixed = lo.DampedLeastSquares(system, lo.curvatures(system), ops, [efl, colour],
                                  solves=[lo.solve_paraxial_focus()]).run()
    opt = lo.DampedLeastSquares(system, lo.curvatures(system) + lo.glass(0) + lo.glass(1), ops,
                                [efl, colour, lo.glass_map_constraint(0),
                                 lo.glass_map_constraint(1)],
                                solves=[lo.solve_paraxial_focus()])
    result = opt.run()
    assert fixed.merit > 100  # coma is left with N-BK7/N-F2
    assert result.merit < 1e-8  # aplanatic with model glasses
    for i in (0, 1):
        assert lo.glass_map_distance(*result.system.surfaces[i].material) <= 1e-6
    chosen = opt.snap_glasses()
    assert set(chosen.values()) <= set(lo.GLASSES)
    assert not any(isinstance(v, (lo.GlassIndex, lo.GlassAbbe)) for v in opt.variables)


def test_dls_reduces_to_gauss_newton_on_a_linear_problem():
    # paraxial power is linear in one curvature: one undamped step solves it
    system = singlet(0.0)
    opt = lo.DampedLeastSquares(system, [lo.Curvature(1)],
                                [lo.paraxial_target("efl", 80.0, 1e-3)], damping=0.0)
    result = opt.run(max_iterations=5)
    assert np.isclose(result.system.paraxial().efl, 80.0, rtol=1e-7)


# --- the damped step: stacked least squares instead of the normal equations ---

class _Numbers:
    """A stand-in for a system: a vector of numbers."""

    def __init__(self, x):
        self.x = np.array(x, float)

    def copy(self):
        return _Numbers(self.x)


class _Entry(lo.Variable):
    def __init__(self, k):
        super().__init__((None, None))
        self.k = k

    def get(self, system):
        return system.x[self.k]

    def set(self, system, value):
        system.x[self.k] = value

    def typical(self, system):
        return 1.0


def _step_with(J, r, damping, A=None, g=None, scale=None):
    """The step of DampedLeastSquares for a given Jacobian (equality
    constraints A dx = -g if given)."""

    n = J.shape[1]
    constraints = []
    if A is not None:
        constraints = [lo.Constraint(lambda s, k=k: 0.0, 0.0, 0.0, 1.0, "c{}".format(k))
                       for k in range(len(A))]
    opt = lo.DampedLeastSquares(_Numbers(np.zeros(n)), [_Entry(k) for k in range(n)],
                                [lambda s: np.zeros(len(r))], constraints)
    if scale is not None:
        opt._scale = np.asarray(scale, float)
    A = np.zeros((0, n)) if A is None else A
    g = np.zeros(len(A)) if g is None else g
    opt.jacobian = lambda system=None: (J, A, r, g)

    return opt.step(damping)


def _normal_equations_step(J, r, damping, A=None, g=None):
    """The textbook step: (J^T J + lambda diag(J^T J)) dx = -J^T r, with
    A dx = -g by Lagrange multipliers (the KKT system, solved directly)."""

    H = J.T @ J
    d = np.diag(H)
    d = np.maximum(d, 1e-9*d.max() + 1e-30)
    H = H + damping*np.diag(d)
    if A is None:
        return np.linalg.solve(H, -J.T @ r)
    m = len(A)
    K = np.block([[H, A.T], [A, np.zeros((m, m))]])

    return np.linalg.solve(K, np.concatenate([-J.T @ r, -g]))[:J.shape[1]]


def _random_jacobian(rng, m, n, condition, column_spread):
    """m x n matrix with singular values from 1 to 1/condition, then
    columns scaled from 1 to column_spread."""

    U = np.linalg.qr(rng.normal(size=(m, n)))[0]
    V = np.linalg.qr(rng.normal(size=(n, n)))[0]
    s = np.logspace(0, -np.log10(condition), n)

    return (U*s) @ V.T*np.logspace(0, np.log10(column_spread), n)


def test_dls_step_unchanged_when_well_conditioned():
    rng = np.random.default_rng(3)
    for damping in (0.0, 1e-3, 1.0, 1e3):
        for constrained in (False, True):
            J = _random_jacobian(rng, 30, 5, 10.0, 100.0)
            r = rng.normal(size=30)
            A = rng.normal(size=(2, 5)) if constrained else None
            g = rng.normal(size=2) if constrained else None
            new = _step_with(J, r, damping, A, g)
            old = _normal_equations_step(J, r, damping, A, g)
            np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-12*np.abs(old).max())
            if constrained:
                np.testing.assert_allclose(A @ new, -g, atol=1e-12)


def test_dls_step_accurate_when_ill_conditioned():
    # column norms over five decades and nearly dependent columns: cond(J)
    # ~ 1e12, cond(J^T J) ~ 1e24; the normal equations lose everything
    # beyond 16 - 2*7 digits of the equilibrated problem, the stacked
    # least squares solve only 16 - 7
    rng = np.random.default_rng(4)
    err_new, err_old = [], []
    for _ in range(10):
        J = _random_jacobian(rng, 40, 6, 1e7, 1e5)
        x = rng.normal(size=6)/np.linalg.norm(J, axis=0)
        r = -J @ x  # zero residual at x: the Gauss-Newton step is x
        for step, err in ((_step_with(J, r, 0.0), err_new),
                          (_normal_equations_step(J, r, 0.0), err_old)):
            err.append(np.max(np.abs(step - x)/np.abs(x)))
    assert np.median(err_new) < 1e-6
    assert np.median(err_old) > 1e3*np.median(err_new)


def test_dls_step_power_of_two_column_scaling_is_exact():
    # scaling a column of J by 2**p (a variable in other units) changes no
    # bit of the equilibrated problem: the step changes by exactly 2**-p
    # (with damping, as long as the column norms stay above the floor of the
    # damping, 1e-9 of the largest diag(J^T J))
    rng = np.random.default_rng(5)
    J = _random_jacobian(rng, 20, 4, 1e3, 1.0)
    r = rng.normal(size=20)
    A, g = rng.normal(size=(1, 4)), rng.normal(size=1)
    for damping, p in ((0.0, 2.0**np.array([-30, 7, 0, 41])),
                       (1e-2, 2.0**np.array([-6, 7, 0, 3]))):
        for args in ((None, None), (A, g)):
            a = _step_with(J, r, damping, *args)
            b = _step_with(J*p, r, damping, *(None if args[0] is None else args[0]*p, args[1]))
            assert np.array_equal(b*p, a)


def test_dls_step_huge_and_infinite_damping():
    # full relative accuracy for any lambda (the damping rows on top of the
    # stacked matrix), the scaled steepest descent step -J^T r/(lambda
    # diag(J^T J)) in the limit, and no step at all for infinite damping
    rng = np.random.default_rng(6)
    J, r = _random_jacobian(rng, 10, 3, 10.0, 1e4), rng.normal(size=10)
    for damping in (1e8, 1e16, 1e40):
        np.testing.assert_allclose(_step_with(J, r, damping),
                                   _normal_equations_step(J, r, damping), rtol=1e-12)
    dx = _step_with(J, r, 1e200)
    expected = -(J.T @ r)/(1e200*np.sum(J**2, axis=0))
    np.testing.assert_allclose(dx, expected, rtol=1e-12)
    assert np.array_equal(_step_with(J, r, np.inf), np.zeros(3))


def test_jacobian_columns_are_the_sensitivities():
    system = singlet(0.3)
    op = lo.seidel("SI")
    opt = lo.DampedLeastSquares(system, [lo.Curvature(0), lo.AirSpace(1, 1)], [op])
    J = opt.jacobian()[0]
    c = system.surfaces[0].c
    dR = 1/(c + 1e-7) - 1/c
    expected = (op(system.with_radius_change(0, dR)) - op(system))/1e-7
    assert np.isclose(J[0, 0], expected[0], rtol=1e-3)


def test_inequality_constraint_becomes_active():
    system = singlet(0.0, t=6.0)
    opt = lo.DampedLeastSquares(system, [lo.Curvature(0), lo.Curvature(1), lo.Thickness(0)],
                                [lo.paraxial_target("efl", 100.0, 1e-3),
                                 lo.seidel("W040", 0.0, 10.0),
                                 lambda s: np.array([(s.surfaces[0].thickness - 2.0)/1e-3])],
                                [lo.Constraint(lo.edge_thickness(0, height=8.0), lower=2.5,
                                               tolerance=0.01)])
    result = opt.run()
    edge = lo.edge_thickness(0, height=8.0)(result.system)
    # the lens is made thinner until the edge constraint holds it:
    assert np.isclose(edge, 2.5, atol=1e-5)
    assert 2.5 < result.system.surfaces[0].thickness < 4.0
    assert "edge0" in result.multipliers


def test_glass_map_and_catalogue():
    assert lo.glass_map_distance(1.5168, 64.17) < 0
    assert lo.glass_map_distance(1.5, 30.0) > 0  # no low index glass with such dispersion
    assert lo.nearest_glass(1.517, 64.0)[0] == "N-BK7"


def test_spot_and_wavefront_operands():
    system = singlet(0.7)
    spot = lo.spot_rms(fields=(0.0,), tolerance=1.0)(system)[0]
    ref = system.spot_diagram(0.0, N=31).rms
    assert np.isclose(spot, ref, rtol=0.25)
    W = lo.wavefront(fields=(0.0,), tolerance=1.0)(system)
    assert np.isclose(np.max(np.abs(W)), np.max(np.abs(np.nan_to_num(system.opd(0.0).W))),
                      rtol=0.1)
    assert np.allclose(lo.polychromatic_ray_errors([WL], fields=(0.0,), tolerance=1e-4)(system),
                       lo.ray_errors(fields=(0.0,), tolerance=1e-4)(system))


def test_dls_length_and_object_na_solves():
    surfaces = [rt.Surface(80.0, 6.0, n_after=N_GLASS), rt.Surface(-80.0, 20.0, n_after=1.0),
                rt.Surface(np.inf, 60.0, semi_diameter=5.0)]
    system = rt.System(surfaces, 2, WL, object_distance=100.0, object_height=2.0)
    opt = lo.DampedLeastSquares(system, [lo.Curvature(0), lo.Thickness(1), lo.ObjectDistance()],
                                [lo.ray_errors(fields=(0.0, 1.0), tolerance=1e-3)],
                                solves=[lo.solve_length(200.0), lo.solve_object_na(0.05)])
    result = opt.run(max_iterations=10)
    s = result.system
    assert np.isclose(s.object_distance + s.image_z, 200.0)
    p = s.paraxial()
    assert np.isclose(p.na*abs(p.magnification), 0.05, rtol=1e-6)


def test_real_na_solve_and_paraxial_calibration():
    from pyoptics.litho import euv
    # off-axis ring field: the real NA at the slit centre is ~8 % below the paraxial NA
    system = euv.six_mirror_projector(freeform=False)
    field = (0.0, 120.0)
    assert lo.real_na(system, field) < 0.95*system.paraxial().na
    assert np.isclose(lo.object_na(system), 0.25*0.25, rtol=1e-6)
    solved = system.copy()
    lo.solve_real_na(0.25, field)(solved)
    assert abs(lo.real_na(solved, field) - 0.25) < 1e-6
    # the calibrated paraxial target gives the same stop
    na_obj = lo.paraxial_na_for_real_na(system, 0.25, field)
    paraxial = system.copy()
    lo.solve_object_na(na_obj)(paraxial)
    assert np.isclose(paraxial.surfaces[paraxial.stop_index].semi_diameter,
                      solved.surfaces[solved.stop_index].semi_diameter, rtol=1e-9)


def test_paraxial_focus_solve_in_front_of_a_cover_glass():
    surfaces = [rt.Surface(50.0, 5.0, n_after=N_GLASS), rt.Surface(-50.0, 30.0, n_after=1.0),
                rt.Surface(np.inf, 0.17, n_after=1.523), rt.Surface(np.inf, 0.0, n_after=1.0)]
    system = rt.System(surfaces, 0, WL, entrance_pupil_radius=5.0)
    lo.solve_paraxial_focus(surface=1)(system)
    assert abs(system.paraxial().image_distance) < 1e-9
    assert system.surfaces[-1].thickness == 0.0
    # the cover glass moves the focus away by t (1 - 1/n)
    plain = rt.System([rt.Surface(50.0, 5.0, n_after=N_GLASS), rt.Surface(-50.0, 0.0, n_after=1.0)],
                      0, WL, entrance_pupil_radius=5.0)
    bfd = plain.paraxial().image_distance
    assert np.isclose(system.surfaces[1].thickness, bfd - 0.17/1.523, atol=1e-6)
    # an offset puts the image plane behind the focus, as for the image distance solve
    lo.solve_paraxial_focus(offset=0.01, surface=1)(system)
    assert np.isclose(system.paraxial().image_distance, -0.01, atol=1e-9)


def test_predicted_gain_ranks_new_variables():
    # an on-axis singlet bent for least spherical aberration: an asphere removes the rest,
    # a variable that repeats the bending gains nothing
    system = singlet(0.7)
    opt = lo.DampedLeastSquares(system, [lo.Curvature(0), lo.Thickness(1)],
                                [lo.ray_errors(fields=(0.0,), tolerance=1e-4)],
                                solves=[lo.solve_curvature_for_efl(1, 100.0)])
    opt.run(max_iterations=30)
    rim = 8.0
    gains, base = opt.predicted_gain({"asphere": [lo.Aspheric(0, 4, rim), lo.Aspheric(0, 6, rim)],
                                      "repeat": [lo.Curvature(0)]})
    assert gains["asphere"] > 0.5*base
    assert abs(gains["repeat"]) < 1e-3*base
    # the prediction comes true
    with_asphere = lo.DampedLeastSquares(opt.system, [lo.Curvature(0), lo.Thickness(1),
                                                      lo.Aspheric(0, 4, rim), lo.Aspheric(0, 6, rim)],
                                         opt.operands, solves=opt.solves).run(max_iterations=30)
    assert with_asphere.merit < 0.2*opt.merit()
