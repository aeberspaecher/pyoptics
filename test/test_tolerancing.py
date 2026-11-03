#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the Monte Carlo tolerancing: distributions, the tolerances against
the sensitivities of `raytrace.System`, the performance vectors, focus
compensation and the linear model against full traces.
"""

import numpy as np
import pytest

from pyoptics import lens_optimization as lo
from pyoptics import raytrace as rt
from pyoptics import tolerancing as tol


WL_F, WL_d, WL_C = 486.13e-6, 587.56e-6, 656.27e-6


def doublet():
    """Air-spaced achromat N-BK7/N-F2, f about 100 mm, F/5, 1 degree."""

    S = [rt.Surface(55.37, 6.0, material=(1.5168, 64.17), semi_diameter=11.5),
         rt.Surface(-34.73, 0.5, n_after=1.0, semi_diameter=11.5),
         rt.Surface(-34.43, 3.0, material=(1.62005, 36.43), semi_diameter=11.5),
         rt.Surface(-167.8, 94.0, n_after=1.0, semi_diameter=11.5)]

    return rt.System(S, 0, WL_d, field_angle=np.deg2rad(1.0),
                     entrance_pupil_radius=10.0).at_paraxial_focus()


@pytest.mark.parametrize("distribution", ["normal", "uniform", "endpoints"])
def test_distributions(distribution):
    t = tol.Thickness(0, 0.05, distribution=distribution)
    x = t.sample(200_000, np.random.default_rng(0))
    assert np.all(np.abs(x) <= 0.05 + 1e-15)
    assert abs(x.std() - t.sigma) < 0.01*t.sigma
    assert abs(x.mean()) < 0.01*t.sigma


def test_tolerances_match_system_sensitivities():
    s = doublet()
    for t, ref in [(tol.Radius(1, 0.02), s.with_radius_change(1, 0.02)),
                   (tol.Index(2, 1e-3), s.with_index_change(2, 1e-3)),
                   (tol.AxialShift(2, 3, 0.1), s.with_axial_shift(2, 3, 0.1))]:
        other = s.copy()
        t.apply(other, t.limit)
        assert other.paraxial().efl == pytest.approx(ref.paraxial().efl, rel=1e-12)
        assert other.paraxial().image_distance == pytest.approx(ref.paraxial().image_distance,
                                                                rel=1e-12)


def test_power_in_fringes():
    s = doublet()
    t = tol.Power(0, 2.0, 11.5)
    other = s.copy()
    t.apply(other, 2.0)
    h = 11.5

    def sag(R):
        return R - np.sign(R)*np.sqrt(R**2 - h**2)

    d_sag = sag(other.surfaces[0].R) - sag(s.surfaces[0].R)
    assert other.surfaces[0].c - s.surfaces[0].c == pytest.approx(2.0*tol.TEST_WAVELENGTH/h**2)
    # N fringes are N half waves of sag (paraxially; 3 % off at h/R = 0.2)
    assert abs(d_sag) == pytest.approx(2.0*tol.TEST_WAVELENGTH/2, rel=0.05)


def test_decenter_and_abbe():
    s = doublet()
    other = s.copy()
    tol.Decenter(2, 3, 0.01, "x").apply(other, 0.01)
    assert other.surfaces[2].decenter == (0.01, 0.0) and other.surfaces[3].decenter == (0.01, 0.0)
    assert other.surfaces[1].decenter == (0.0, 0.0)
    tol.Abbe(0, 0.01, relative=True).apply(other, 0.01)
    assert other.surfaces[0].material == (1.5168, pytest.approx(64.17*1.01))


def test_wavefront_errors_rms_equals_rms_opd():
    s = doublet()
    perf = tol.WavefrontErrors(fields=(0.0, 1.0), N=24)
    y = perf(s)
    rms = tol._block_rms(y, perf.blocks(len(y)))[0]
    for r, f in zip(rms, (0.0, s.max_field)):
        assert r == pytest.approx(s.rms_opd(f, N=24), rel=1e-6)


def test_spot_errors_rms():
    s = doublet()
    perf = tol.SpotErrors(fields=(0.0,), rings=(0.5, 1.0), spokes=12)
    y = perf(s)
    I = s._image_hits(s.trace(s.rays(0.0, perf.px, perf.py), clip=False))[0][:, :2]
    c = perf.w_pupil @ I
    expected = np.sqrt(perf.w_pupil @ np.sum((I - c)**2, axis=1))
    assert np.sqrt(np.sum(y**2)) == pytest.approx(expected, rel=1e-12)


def test_focus_compensates_axial_errors():
    s = doublet()
    perf = tol.SpotErrors(fields=(0.0, 1.0), wavelengths=[WL_F, WL_d, WL_C])
    tols = [tol.AxialShift(0, 3, 0.1), tol.ImagePlane(0.1, after_compensation=False)]
    ana = tol.ToleranceAnalysis(s, tols, perf, [lo.Thickness(-1)])
    nominal = ana.nominal_rms()
    _, y, dc = ana.perturbed([0.1, 0.05])
    assert np.allclose(tol._block_rms(y, ana.blocks)[0], nominal, rtol=1e-6)
    dc0 = ana.perturbed([0.0, 0.0])[2]
    # the lens moved 0.1 towards the image plane, which moved back 0.05
    assert dc[0] - dc0[0] == pytest.approx(0.05, abs=1e-6)
    # without the compensator the defocus shows
    ana0 = tol.ToleranceAnalysis(s, tols, perf)
    _, y0, _ = ana0.perturbed([0.1, 0.05])
    assert np.all(tol._block_rms(y0, ana0.blocks)[0] > 1.2*nominal)


def test_after_compensation_is_not_compensated():
    s = doublet()
    perf = tol.SpotErrors(fields=(0.0,), wavelengths=[WL_d])
    ana = tol.ToleranceAnalysis(s, [tol.ImagePlane(0.05)], perf, [lo.Thickness(-1)])
    r0 = ana.nominal_rms()[0]
    r = tol._block_rms(ana.perturbed([0.05])[1], ana.blocks)[0, 0]
    assert r > 2*r0


def test_linear_model_against_monte_carlo():
    s = doublet()
    perf = tol.SpotErrors(fields=(0.0, 1.0, (1.0, 0.0)), wavelengths=[WL_F, WL_d, WL_C])
    tols = tol.element_tolerances(s, [(0, 1), (2, 3)], fringes=2.0, thickness=0.03, index=3e-4,
                                  abbe=0.005, decenter=0.01) + [tol.Thickness(1, 0.02), tol.ImagePlane(0.01)]

    def colour(system):
        return lo.focal_shift(WL_F, WL_C, tolerance=1.0)(system)[0]

    ana = tol.ToleranceAnalysis(s, tols, perf, [lo.Thickness(-1)], {"colour": colour})
    sens, model = ana.sensitivities()
    assert sens.share.sum() == pytest.approx(1.0)
    decenter = np.array([t.group == "decenter" for t in tols])
    assert sens.share[decenter].sum() > 0.5
    mc = ana.monte_carlo(15, seed=1)
    lin = model.predict(mc.values)
    assert np.allclose(lin.rms, mc.rms, rtol=0.02)
    assert np.allclose(lin.metrics["colour"], mc.metrics["colour"], atol=2e-4)
    assert np.allclose(lin.compensators, mc.compensators, atol=2e-3)
    # the mean square of the linear model against its own Monte Carlo
    big = model.monte_carlo(40_000, seed=2)
    assert np.mean(big.rms**2) == pytest.approx(model.mean_square(), rel=0.03)
    assert 0.0 <= tol.yield_fraction(big, 0.007, {"colour": 0.03}) <= 1.0


def test_lateral_manipulator_compensates_decenter():
    s = doublet()
    perf = tol.SpotErrors(fields=(0.0,), wavelengths=[WL_d])
    tols = [tol.Decenter(2, 3, 0.02, "y")]
    ana = tol.ToleranceAnalysis(s, tols, perf, [lo.Thickness(-1)])
    worse = tol._block_rms(ana.perturbed([0.02])[1], ana.blocks)[0, 0]
    ana_m = ana.with_compensators([lo.Thickness(-1), tol.LateralShift(0, 1, "y")], iterations=2)
    fixed, y, dc = ana_m.perturbed([0.02])
    assert tol._block_rms(y, ana_m.blocks)[0, 0] < 0.5*worse
    # moving the crown with the flint removes the relative decentre
    assert dc[1] == pytest.approx(0.02, rel=0.1)
