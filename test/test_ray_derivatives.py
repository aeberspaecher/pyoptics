#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the analytic derivatives of real ray traces (differential ray
tracing) against central differences of complete traces: mirror systems
with Q-con and Zernike surfaces, tilts and decentres, refracting lenses with
model glasses and aspheres, aimed and unaimed rays, solves, the operands
with analytic Jacobians and the optimizers' jac="analytic"."""

import numpy as np
import pytest

from pyoptics import lens_optimization as lo
from pyoptics import off_axis as oa
from pyoptics import qaspheres as qa
from pyoptics import raytrace as rt


def freeform_tma():
    """Cook three-mirror anastigmat with a Q-con primary and a Zernike
    tertiary."""

    s = oa.cook_tma()
    S = s.surfaces
    S[0] = qa.QconSurface(R=S[0].R, thickness=S[0].thickness, conic=S[0].conic,
                          q=(1e-3, 2e-4), norm_radius=100.0, mirror=True)
    S[2] = rt.ZernikeSurface(R=S[2].R, thickness=S[2].thickness, conic=S[2].conic,
                             zernike={5: 1e-3, 7: 5e-4}, norm_radius=100.0, mirror=True)

    return s


TMA_VARIABLES = [lo.Curvature(0), lo.Conic(0), qa.QCoefficient(0, 1), qa.QCoefficient(0, 3),
                 lo.Curvature(1), lo.Conic(1), lo.ZernikeTerm(2, 5), lo.ZernikeTerm(2, 8),
                 lo.Curvature(2), lo.Thickness(0), lo.Thickness(1), lo.Thickness(2),
                 lo.Tilt(2, 0), lo.Decenter(1, 1), lo.ImageTilt(0)]


def doublet(object_distance=np.inf):
    S = [rt.Surface(60.0, 5.0, material=(1.5168, 64.17), semi_diameter=10.0,
                    aspheric=(1e-7,)),
         rt.Surface(-45.0, 2.5, material=(1.62005, 36.43)),
         rt.Surface(-250.0, 95.0, n_after=1.0)]
    if np.isinf(object_distance):
        return rt.System(S, 0, 0.5e-3, field_angle=np.deg2rad(2.0)).at_paraxial_focus()

    return rt.System(S, 0, 0.5e-3, object_distance=object_distance,
                     object_height=3.0).at_paraxial_focus()


def doublet_variables(system):
    return (lo.curvatures(system) + [lo.Thickness(0), lo.Thickness(1), lo.Aspheric(0, 4, 10.0),
                                     lo.Aspheric(2, 6, 10.0)] + lo.glass(1))


def rel_errors(J, Jfd):
    scale = np.maximum(np.max(np.abs(Jfd), axis=0), 1e-300)

    return np.max(np.abs(J - Jfd), axis=0)/scale


def image_rays(field, px, py):
    def f(system):
        b = system.trace(system.rays(field, px, py), clip=False)
        I, D = system._image_hits(b)
        d0 = np.reshape(system.rays(field, px, py).directions, (-1, 3))
        return np.concatenate([I.ravel(), D.ravel(), b.opl[-1], d0.ravel()])

    return f


def stacked(rd):
    V = len(rd.dI)
    return np.concatenate([rd.dI.reshape(V, -1), rd.dD.reshape(V, -1), rd.dopl,
                           rd.dd0.reshape(V, -1)], axis=1).T


@pytest.mark.parametrize("aim", [True, False])
@pytest.mark.parametrize("focus", [False, True])
def test_mirror_trace_derivatives(aim, focus):
    s = freeform_tma()
    s.ray_aiming = aim
    dls = lo.DampedLeastSquares(s, TMA_VARIABLES, [],
                                solves=[lo.solve_paraxial_focus()] if focus else [])
    lin = dls.linearization()
    assert not np.any(lin.unsupported)
    field = (np.radians(0.5), np.radians(3.0))
    px, py = oa.pupil_rings((0.5, 1.0), 8)
    rd = lin.trace(field, px, py)
    f = image_rays(field, px, py)
    assert np.allclose(np.concatenate([rd.I.ravel(), rd.D.ravel(), rd.opl, rd.d0.ravel()]),
                       f(dls.system), rtol=0, atol=1e-9)
    err = rel_errors(stacked(rd), lin.finite_difference(f))
    assert np.all(err < 1e-5), dict(zip([v.name for v in TMA_VARIABLES], err))


@pytest.mark.parametrize("object_distance", [np.inf, 400.0])
def test_refracting_trace_derivatives(object_distance):
    s = doublet(object_distance)
    solves = ([lo.solve_entrance_pupil(8.0)] if np.isinf(object_distance)
              else [lo.solve_object_na(0.02)])
    variables = doublet_variables(s)
    if not np.isinf(object_distance):
        variables.append(lo.ObjectDistance())
    dls = lo.DampedLeastSquares(s, variables, [], solves=solves + [lo.solve_paraxial_focus()])
    lin = dls.linearization()
    assert not np.any(lin.unsupported)
    field = 0.7*s.max_field
    px, py = oa.pupil_rings((0.5, 1.0), 6)
    rd = lin.trace(field, px, py)
    err = rel_errors(stacked(rd), lin.finite_difference(image_rays(field, px, py)))
    assert np.all(err < 1e-5), dict(zip([v.name for v in variables], err))


def test_operand_jacobians():
    s = freeform_tma()
    fields = np.radians([[0.0, 2.0], [1.0, 3.0], [-1.0, 4.0]])
    ops = [oa.field_wavefront(fields, rings=(0.5, 1.0), spokes=8, tolerance=0.05),
           lo.ray_errors(fields=(0.5, 1.0), tolerance=1e-3),
           lo.wavefront(fields=(0.5, 1.0), tolerance=0.05)]
    dls = lo.DampedLeastSquares(s, TMA_VARIABLES, ops, solves=[lo.solve_paraxial_focus()])
    lin = dls.linearization()
    for op in ops:
        r, J = op.linearize(lin)
        assert np.allclose(r, op(dls.system), rtol=1e-9, atol=1e-9)
        err = rel_errors(J, lin.finite_difference(op))
        assert np.all(err < 1e-4), err


def test_polychromatic_wavefront_jacobian():
    s = doublet()
    op = lo.wavefront(fields=(0.0, 1.0), wavelengths=(0.48613e-3, 0.65627e-3), tolerance=0.05)
    dls = lo.DampedLeastSquares(s, doublet_variables(s), [op])
    lin = dls.linearization()
    r, J = op.linearize(lin)
    assert np.allclose(r, op(dls.system), rtol=1e-9, atol=1e-9)
    assert np.all(rel_errors(J, lin.finite_difference(op)) < 1e-4)


@pytest.mark.parametrize("object_distance", [np.inf, 200.0])
def test_polychromatic_ray_error_and_spot_jacobians(object_distance):
    from pyoptics.ray_derivatives import Linearization

    s = doublet(object_distance)
    F, C = 0.48613e-3, 0.65627e-3
    ops = [lo.polychromatic_ray_errors((F, 0.5876e-3, C), fields=(0.0, 0.7, 1.0), weights=(0.5, 1.0, 0.7)),
           lo.spot_rms(fields=(0.0, 1.0)),
           lo.spot_rms(fields=(0.0, 1.0), wavelengths=(F, C)),
           lo.spot_rms(fields=(0.3, 1.0), wavelengths=(F, C), about="chief")]
    dls = lo.DampedLeastSquares(s, doublet_variables(s), ops, solves=[lo.solve_paraxial_focus()])
    # the RMS spot radius is strongly nonlinear in the aspheric terms: small difference steps
    lin = Linearization(dls.system_at, dls.x, 1e-5*dls._scale, fd_steps=1e-6*dls._scale)
    for op in ops:
        r, J = op.linearize(lin)
        assert np.allclose(r, op(dls.system), rtol=1e-9, atol=1e-9)
        assert np.all(rel_errors(J, lin.finite_difference(op)) < 1e-4)


@pytest.mark.parametrize("aim", [False, True])
def test_chief_ray_and_paraxial_operand_jacobians(aim):
    from pyoptics.ray_derivatives import Linearization

    def errors(J, Jfd):  # relative to the largest entry of the operand's Jacobian
        return np.max(np.abs(J - Jfd))/np.max(np.abs(Jfd))

    s = doublet(200.0)
    s.surfaces[1].thickness = 4.0  # stop inside the lens: aimed and paraxial chief rays differ
    s.surfaces[1].semi_diameter = 8.0
    s.stop_index = 1
    s = s.at_paraxial_focus()
    rays = [lo.distortion(-0.5, fields=(0.5, 1.0), aim=aim),
            lo.image_telecentricity(fields=(0.5, 1.0), aim=aim),
            lo.object_telecentricity(fields=(0.5, 1.0), aim=aim)]
    paraxial = [lo.paraxial_target("magnification", -0.5, 1e-3),
                lo.clearances(min_center=4.5, min_edge=1.0),
                lo.seidel("W040"), lo.chromatic("axial", 0.1e-3),
                lo.focal_shift(0.48613e-3, 0.65627e-3)]
    dls = lo.DampedLeastSquares(s, doublet_variables(s), rays + paraxial,
                                solves=[lo.solve_paraxial_focus()])
    lin = dls.linearization()
    # paraxial data have no noise: small difference steps
    lin_small = Linearization(dls.system_at, dls.x, 1e-5*dls._scale, fd_steps=1e-5*dls._scale)
    for ops, tol in ((rays, 1e-5), (paraxial, 1e-6)):
        for op in ops:
            r, J = op.linearize(lin)
            assert np.allclose(r, op(dls.system), rtol=1e-8, atol=1e-8)
            assert errors(J, (lin if ops is rays else lin_small).finite_difference(op)) < tol
    s = rt.System(doublet().surfaces, 0, 0.5e-3, field_angle=np.deg2rad(5.0)).at_paraxial_focus()
    op = lo.ftheta(fields=(0.5, 1.0), aim=aim)
    dls = lo.DampedLeastSquares(s, doublet_variables(s), [op])
    lin = dls.linearization()
    r, J = op.linearize(lin)
    assert np.allclose(r, op(dls.system), rtol=1e-8, atol=1e-8)
    assert errors(J, lin.finite_difference(op)) < 1e-5


class Wavelength(lo.Variable):
    name = "wavelength"

    def get(self, system):
        return system.wavelength

    def set(self, system, value):
        system.wavelength = value

    def typical(self, system):
        return 1e-3*system.wavelength


def test_unsupported_parameters_and_surfaces_fall_back_to_differences():
    s = doublet()
    op = lo.wavefront(fields=(0.0, 1.0), tolerance=0.05)
    variables = [lo.Curvature(0), Wavelength()]
    dls = lo.DampedLeastSquares(s, variables, [op])
    lin = dls.linearization()
    assert list(lin.unsupported) == [False, True]
    r, J = lin.jacobian(op)
    Jfd = lin.finite_difference(op)
    assert np.allclose(J[:, 1], Jfd[:, 1])
    assert rel_errors(J[:, :1], Jfd[:, :1])[0] < 1e-4
    # an ideal lens has no analytic derivatives: the whole operand by differences
    S = [rt.IdealLens(100.0, 100.0, semi_diameter=10.0)]
    ideal = rt.System(S, 0, 0.5e-3, field_angle=0.01)
    dls = lo.DampedLeastSquares(ideal, [lo.Thickness(0)], [op])
    lin = dls.linearization()
    with pytest.raises(NotImplementedError):
        lin.trace(0.0, 0.5, 0.5)
    r, J = lin.jacobian(op)
    assert np.allclose(J, lin.finite_difference(op))


def test_dls_analytic_jacobian():
    s = freeform_tma()
    fields = np.radians([[0.0, 2.0], [0.0, 4.0]])
    ops = [oa.field_wavefront(fields, rings=(0.5, 1.0), spokes=8, tolerance=0.05),
           lo.paraxial_target("efl", 600.0, 0.1)]
    variables = [lo.Conic(0), qa.QCoefficient(0, 1), lo.Conic(1), lo.ZernikeTerm(2, 5),
                 lo.Thickness(1)]
    efl = lo.Constraint(lo.paraxial_value("efl"), 600.0, 600.0, 1e-3, "efl")
    kw = dict(solves=[lo.solve_paraxial_focus()])
    fd = lo.DampedLeastSquares(s, variables, ops, **kw)
    an = lo.DampedLeastSquares(s, variables, ops, constraints=[efl], jac="analytic", **kw)
    J0, _, r0, _ = fd.jacobian()
    J1, A1, r1, g1 = an.jacobian()
    assert np.allclose(r0, r1, rtol=1e-9, atol=1e-9)
    assert np.all(rel_errors(J1, J0) < 1e-3)  # forward differences
    assert A1.shape == (1, len(variables))
    before = an.merit()
    result = an.run(max_iterations=5)
    assert result.merit < 0.5*before
    with pytest.raises(ValueError):
        lo.DampedLeastSquares(s, variables, ops, jac="3-point")


def test_lens_optimizer_analytic_jacobian():
    system = rt.System([rt.Surface(80.0, 6.0, n_after=1.5168), rt.Surface(-80.0, 20.0, n_after=1.0),
                        rt.Surface(np.inf, 60.0, semi_diameter=5.0)], 2, 0.55e-3,
                       object_distance=100.0, object_height=2.0)
    opt = lo.LensOptimizer(system, [lo.ray_errors(tolerance=1e-3),
                                    lo.paraxial_target("magnification", -1.0, 1e-2)],
                           thicknesses=[1], object_distance=True, length=200.0, object_na=0.05)
    J = opt.jacobian()
    x, h = opt.x, 1e-4*opt._scales()
    Jfd = np.array([(opt.residuals(x + e) - opt.residuals(x - e))/(2*e[k])
                    for k, e in enumerate(np.diag(h))]).T
    assert np.all(rel_errors(J, Jfd) < 1e-4)
    result = opt.run(max_nfev=30, jac="analytic")
    assert result.cost < opt.__class__(system, opt.operands, thicknesses=[1], object_distance=True,
                                       length=200.0, object_na=0.05).cost()
