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


# --- tilts, wedges, irregularity, inhomogeneity ---------------------------------

def shell_system():
    """A meniscus whose two surfaces are concentric about z = 50 behind its
    first vertex, focused by an ideal lens."""

    S = [rt.Surface(np.inf, 20.0, n_after=1.0, semi_diameter=10.0),
         rt.Surface(50.0, 5.0, material=(1.5168, 64.17), semi_diameter=10.0),
         rt.Surface(45.0, 20.0, n_after=1.0, semi_diameter=10.0),
         rt.IdealLens(50.0, 50.0, semi_diameter=10.0)]

    return rt.System(S, 0, WL_d, field_angle=np.deg2rad(1.0), entrance_pupil_radius=5.0)


def image_points(system, field=0.3):
    px, py = np.array([0.0, 0.5, -0.5, 0.2]), np.array([0.0, 0.3, 0.7, -0.6])
    b = system.trace(system.rays(field*system.max_field, px, py), clip=False)

    return system._image_hits(b)[0]


@pytest.mark.parametrize("axis", ["x", "y"])
def test_tilt_about_common_centre_changes_nothing(axis):
    s = shell_system()
    other = s.copy()
    tol.Tilt(1, 2, 0.01, axis, pivot=50.0).apply(other, 0.01)
    assert other.surfaces[1].tilt != (0.0, 0.0, 0.0)
    assert np.allclose(image_points(other), image_points(s), atol=1e-10)
    # about another pivot the image moves
    moved = s.copy()
    tol.Tilt(1, 2, 0.01, axis).apply(moved, 0.01)
    assert np.max(np.abs(image_points(moved) - image_points(s))) > 1e-3
    # the surfaces behind the element stay where they were
    assert moved.frames()[0][3][0] == pytest.approx(s.frames()[0][3][0], abs=1e-12)


def test_tilt_of_one_surface_about_its_vertex_is_a_wedge():
    s = doublet()
    a, b = s.copy(), s.copy()
    tol.Tilt(1, 1, 1e-3, "y", pivot=0.0).apply(a, 1e-3)
    tol.Wedge(1, 1e-3, "y").apply(b, 1e-3)
    assert a.surfaces[1].tilt == b.surfaces[1].tilt == (0.0, 1e-3, 0.0)
    assert np.allclose(image_points(a, 1.0), image_points(b, 1.0), atol=1e-12)


def test_element_tilt_compensator_matches_tilt_tolerance():
    s = doublet()
    v = tol.ElementTilt(0, 3, "x", typical=1e-3)
    a = s.copy()
    v.set(a, 2e-3)
    assert v.get(a) == pytest.approx(2e-3)
    b = s.copy()
    tol.Tilt(0, 3, 2e-3, "x").apply(b, 2e-3)
    assert np.allclose(image_points(a, 1.0), image_points(b, 1.0), atol=1e-12)
    v.set(a, 0.0)
    assert np.allclose(image_points(a, 1.0), image_points(s, 1.0), atol=1e-10)


@pytest.mark.parametrize("term, pv", [(5, 2.0), (6, 2.0), (9, 1.5), (7, None)])
def test_irregularity_peak_to_valley(term, pv):
    s = doublet()
    t = tol.Irregularity(1, 0.5, 11.5, term)
    if pv is not None:
        assert tol._fringe_pv(term) == pytest.approx(pv, rel=1e-4)
    other = s.copy()
    t.apply(other, 0.5)
    assert isinstance(other.surfaces[1], rt.ZernikeSurface)
    assert other.surfaces[1].R == s.surfaces[1].R
    r, a = np.meshgrid(11.5*np.sqrt(np.linspace(0, 1, 201)), np.linspace(0, 2*np.pi, 721))
    d = other.surfaces[1].departure(r*np.cos(a), r*np.sin(a))
    assert np.ptp(d) == pytest.approx(0.5*tol.TEST_WAVELENGTH/2, rel=1e-4)
    # a second term adds to the first one
    tol.Irregularity(1, 0.5, 11.5, 9).apply(other, 0.5)
    assert len(dict(other.surfaces[1].zernike)) == (1 if term == 9 else 2)
    # the original stays untouched and the ray trace works
    assert type(s.surfaces[1]) is rt.Surface
    assert np.all(np.isfinite(image_points(other, 1.0)))


def plate():
    S = [rt.Surface(np.inf, 10.0, material=(1.5, 60.0), semi_diameter=10.0),
         rt.Surface(np.inf, 10.0, n_after=1.0, semi_diameter=10.0),
         rt.IdealLens(100.0, 100.0, semi_diameter=10.0)]

    return rt.System(S, 0, WL_d, field_angle=0.0, entrance_pupil_radius=8.0)


@pytest.mark.parametrize("term", [4, 5, 9])
def test_inhomogeneity_adds_its_optical_path(term):
    from pyoptics.bases import fringe_zernike
    p = plate()
    q = p.copy()
    tol.Inhomogeneity(0, 1, 1e-5, 8.0, term).apply(q, 1e-5)
    W = q.opd(0.0, N=33, units="length").W - p.opd(0.0, N=33, units="length").W
    g = np.linspace(-1, 1, 33)
    GX, GY = np.meshgrid(g, g)
    expected = 10.0*1e-5*fringe_zernike(term, GX, GY)/tol._fringe_pv(term)
    ok = np.isfinite(W)
    err = W[ok] - expected[ok]
    assert np.ptp(W[ok]) == pytest.approx(np.ptp(expected[ok]), rel=0.01)
    # (the departure also bends the rays a little: a few per cent for Z9)
    assert np.std(err - err.mean()) < 0.03*np.std(expected[ok])


def test_inhomogeneity_gradient_deviates_the_beam():
    # dn = dn_pv*x/(2 rim) across a plate of thickness t deviates by t*dn_pv/(2 rim)
    p = plate()
    q = p.copy()
    tol.Inhomogeneity(0, 1, 1e-5, 8.0, 2).apply(q, 1e-5)
    shift = image_points(q, 0.0)[0, 0] - image_points(p, 0.0)[0, 0]
    assert shift == pytest.approx(100.0*10.0*1e-5/16.0, rel=0.01)


def test_element_tolerances_new_kinds():
    s = doublet()
    els = [(0, 1), (2, 3)]
    tols = tol.element_tolerances(s, els, tilt=1e-4, wedge=1e-4, irregularity=0.2,
                                  inhomogeneity=2e-6)
    groups = [t.group for t in tols]
    assert groups.count("tilt") == 4 and groups.count("wedge") == 4
    assert groups.count("irregularity") == 4*3 and groups.count("inhomogeneity") == 2*3
    assert {t.surface for t in tols if t.group == "wedge"} == {1, 3}


def test_image_tilt_and_wavelength_compensators():
    s = doublet()
    v = tol.ImageTilt("y")
    a = s.copy()
    v.set(a, 1e-3)
    assert a.image_tilt == (0.0, 1e-3, 0.0) and v.get(a) == 1e-3
    w = tol.Wavelength(typical=1e-6)
    w.set(a, WL_F)
    assert a.wavelength == w.get(a) == WL_F and w.typical(a) == 1e-6


# --- inverse sensitivity and allocation -----------------------------------------

@pytest.fixture(scope="module")
def doublet_model():
    s = doublet()
    perf = tol.WavefrontErrors(fields=(0.0, 1.0, (1.0, 0.0)), N=12)
    tols = tol.element_tolerances(s, [(0, 1), (2, 3)], fringes=1.0, thickness=0.02,
                                  decenter=0.01, tilt=2e-4)

    def colour(system):
        return lo.focal_shift(WL_F, WL_C, tolerance=1.0)(system)[0]

    ana = tol.ToleranceAnalysis(s, tols, perf, [lo.Thickness(-1)], {"colour": colour})

    return ana, ana.linear_model()


def test_inverse_sensitivity(doublet_model):
    ana, model = doublet_model
    rms0 = np.sqrt(model.mean_square(np.zeros(len(ana.tolerances), bool)))
    target = 1.5*rms0 + 0.02
    lim = model.inverse_sensitivity(target)
    assert np.all(lim > 0)
    for k in range(len(lim)):
        if not np.isfinite(lim[k]):
            continue
        rms = []
        for sign in (1.0, -1.0):
            x = np.zeros(len(lim))
            x[k] = sign*lim[k]
            rms.append(np.sqrt(np.mean(model.predict(x).rms**2)))
        # one sign reaches the target, the other stays at or below it
        assert max(rms) == pytest.approx(target, rel=1e-9)
    lim_c = model.inverse_sensitivity(1e-3, metric="colour")
    g = np.abs(model._metric_jacobian("colour"))
    finite = np.isfinite(lim_c)
    assert np.allclose(g[finite]*lim_c[finite], 1e-3)
    # a target below the nominal merit leaves nothing
    assert np.all(model.inverse_sensitivity(0.5*rms0) == 0)


def test_allocate_meets_the_budget(doublet_model):
    ana, model = doublet_model
    rms0 = np.sqrt(model.mean_square(np.zeros(len(ana.tolerances), bool)))
    target = 2*rms0
    lim = model.allocate(target)
    new = model.with_limits(lim)
    assert np.sqrt(new.mean_square()) == pytest.approx(target, rel=1e-9)
    # every tolerance takes the same share
    sig = np.array([t.sigma for t in new.analysis.tolerances])
    share = sig**2*np.sum(new.B**2, axis=0)
    assert np.allclose(share, share.mean(), rtol=1e-9)
    # the original analysis keeps its limits
    assert ana.tolerances[0].limit == 1.0
    # weights: twice the share, fixed tolerances keep their limits
    w = np.ones(len(lim))
    w[0] = 2.0
    fixed = np.zeros(len(lim), bool)
    fixed[-1] = True
    lim2 = model.allocate(target, weights=w, fixed=fixed)
    new2 = model.with_limits(lim2)
    sig2 = np.array([t.sigma for t in new2.analysis.tolerances])
    share2 = sig2**2*np.sum(new2.B**2, axis=0)
    assert share2[0] == pytest.approx(2*share2[1], rel=1e-9)
    assert lim2[-1] == ana.tolerances[-1].limit
    assert np.sqrt(new2.mean_square()) == pytest.approx(target, rel=1e-9)
    # a metric budget: the standard deviation of the colour
    lim3 = model.allocate(1e-3, metric="colour")
    mc = model.with_limits(np.where(np.isfinite(lim3), lim3, 1.0)).monte_carlo(40_000, seed=1)
    assert np.std(mc.metrics["colour"]) == pytest.approx(1e-3, rel=0.03)
    with pytest.raises(ValueError):
        model.allocate(0.5*rms0)


def test_monte_carlo_vectors(doublet_model):
    ana, model = doublet_model
    lin = model.monte_carlo(50, seed=1, vectors=True)
    assert lin.vectors.shape == (50, len(ana.y0))
    assert np.allclose(tol._block_rms(lin.vectors, ana.blocks), lin.rms)
    assert model.monte_carlo(5, seed=1).vectors is None
    full = ana.monte_carlo(2, seed=1, vectors=True)
    assert np.allclose(tol._block_rms(full.vectors, ana.blocks), full.rms)


def test_compensator_ranges_damp_the_strokes(doublet_model):
    ana, model = doublet_model
    X = ana.sample(200, seed=5)
    free = model.predict(X).compensators
    wide = model.with_compensators(ana.compensators, ana.C, ana.G_comp, ranges=[1e6])
    assert np.allclose(wide.predict(X).compensators, free, rtol=1e-6, atol=1e-12)
    r = 0.1*np.std(free)
    tight = model.with_compensators(ana.compensators, ana.C, ana.G_comp, ranges=[r])
    # damped least squares: the stroke shrinks, the merit grows
    assert np.std(tight.predict(X).compensators) < 0.5*np.std(free)
    assert tight.mean_square() > model.mean_square()
    # the full compensation uses the same damping
    ana_t = ana.with_compensators(ana.compensators, ana.C, ana.G_comp, ranges=[r])
    dc = ana_t.perturbed(X[0])[2]
    assert dc == pytest.approx(tight.predict(X[:1]).compensators[0], rel=0.05)
