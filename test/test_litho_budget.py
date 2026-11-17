#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the chain from lens tolerances to wafer errors (pyoptics.litho.budget):
the aberration vector and its units, the scanner correction, the
lithographic errors against the sensitivities, and the linear model against
full traces with compensators.
"""

import numpy as np
import pytest

from pyoptics import lens_optimization as lo
from pyoptics import tolerancing as tol
from pyoptics.litho import (Illumination, LithoSimulation, MaskLayout, Projector, ThinMask,
                            budget, lens)
from pyoptics.raytrace import caldwell_krf_lens


X_SLIT = (-10.0, 0.0, 10.0)


@pytest.fixture(scope="module")
def setup():
    system = caldwell_krf_lens()
    ab = budget.SlitAberrations(system, X_SLIT, terms=range(4, 17), N=12)
    sim = LithoSimulation(MaskLayout.lines_spaces(500.0, 250.0), ThinMask.binary(),
                          Illumination.conventional(0.6, n_samples=11),
                          Projector(ab.NA, ab.wavelength, magnification=4.0))
    feat = budget.slit_sensitivities(sim, ab, "dense", dose=sim.dose_to_size(threshold=0.3), N=201)

    return system, ab, feat


def test_nominal_vector_is_zero(setup):
    system, ab, _ = setup
    y = ab(system)
    assert y.shape == (len(X_SLIT)*ab.block_size,)
    assert np.allclose(y, 0.0, atol=1e-12)
    assert len(ab.blocks(len(y))) == len(X_SLIT)
    # the nominal Zernikes are those of lens_aberrations
    z = lens.lens_aberrations(system, (10.0, 0.0), ab.n_terms, ab.N, ab.NA)
    assert ab.nominal_aberrations[2][9] == pytest.approx(z.zernikes[9])


def test_placement_units_and_scanner_correction(setup):
    system, ab, _ = setup
    shifted = system.copy()
    shifted.image_decenter = (2e-6, 0.0)  # 2 nm in mm
    raw = budget.SlitAberrations(system, X_SLIT, terms=range(4, 17), N=12, scanner_correction=False)
    dZ, dP = raw.unpack(raw(shifted))
    assert np.allclose(dP[0, :, 0], -2.0, atol=1e-6)
    assert np.allclose(dZ, 0.0, atol=1e-9)
    # the placement in the vector is the equivalent tilt in waves
    assert raw(shifted)[raw.block_size - 2] == pytest.approx(-2.0*ab.NA/ab.wavelength, rel=1e-6)
    # a translation is what a scanner corrects
    assert np.allclose(ab.unpack(ab(shifted))[1], 0.0, atol=1e-6)


def test_litho_errors_are_the_linear_sensitivities(setup):
    system, ab, feat = setup
    le = budget.LithoErrors(ab, [feat], cd=5.0, placement=10.0, focus=50.0, zernike_weight=0.1,
                            quadratic=False)
    y = np.zeros(len(X_SLIT)*ab.block_size)
    k, j = 1, 9
    i = list(ab.terms).index(j)
    y[k*ab.block_size + i] = 0.01
    v = le.from_aberrations(y)
    n = len(v)//len(X_SLIT)
    s = feat.sensitivities[k]
    js = list(s.terms).index(j)
    assert v[k*n:k*n + 3] == pytest.approx([s.linear["cd"][js]*0.01/5.0,
                                            s.linear["placement"][js]*0.01/10.0,
                                            s.linear["best_focus"][js]*0.01/50.0])
    assert np.allclose(le.unpack(v), y)
    # placement of the lens enters the placement row
    y = np.zeros_like(y)
    y[ab.block_size - 2] = ab.NA/ab.wavelength  # 1 nm along x at the first slit position
    assert le.from_aberrations(y)[1] == pytest.approx(1.0/10.0)
    # wafer errors: the quadratic prediction plus the placement
    y = np.zeros_like(y)
    y[k*ab.block_size + i] = 0.01
    y[k*ab.block_size + ab.block_size - 2] = 3.0*ab.NA/ab.wavelength
    w = budget.wafer_errors(le, le.from_aberrations(y)[None, :])["dense"]
    p = lens.predict(s, {j: 0.01})
    assert w.cd[0, k] == pytest.approx(p["cd"])
    assert w.best_focus[0, k] == pytest.approx(p["best_focus"])
    assert w.placement[0, k] == pytest.approx(p["placement"] + 3.0)
    assert np.allclose(w.cd[0, [0, 2]], 0.0)
    # the same from the plain aberration vector
    w2 = budget.wafer_errors(ab, y[None, :], [feat])["dense"]
    assert np.allclose(w2.cd, w.cd) and np.allclose(w2.placement, w.placement)
    S = budget.wafer_summary({"dense": w})["dense"]
    assert S["focus_range"] == pytest.approx(np.ptp(w.best_focus))
    assert budget.wafer_yield({"dense": w}, placement=3.1) == 1.0
    assert budget.wafer_yield({"dense": w}, placement=2.9) == 0.0


def test_linear_model_against_full_traces_with_compensators(setup):
    system, ab, feat = setup
    le = budget.LithoErrors(ab, [feat], cd=5.0, placement=10.0, focus=50.0)
    tols = [tol.Tilt(14, 15, 1e-5, "y"), tol.Irregularity(29, 0.1, 50.0, 9),
            tol.Thickness(17, 2e-3), tol.Decenter(8, 9, 1e-3, "x")]
    ana = tol.ToleranceAnalysis(system, tols, le, [lo.Thickness(-1), tol.ImageTilt("y")],
                                iterations=2)
    model = ana.linear_model()
    X = ana.sample(3, seed=4)
    full = ana.monte_carlo(values=X, vectors=True)
    lin = model.predict(X, vectors=True)
    wf, wl = budget.wafer_errors(le, full.vectors), budget.wafer_errors(le, lin.vectors)
    for key in ("cd", "placement", "best_focus"):
        a, b = getattr(wf["dense"], key), getattr(wl["dense"], key)
        assert np.max(np.abs(a - b)) < 0.05*np.max(np.abs(a)) + 1e-3
    assert np.allclose(lin.compensators, full.compensators, rtol=0.05, atol=1e-9)
    # the compensators make it better
    none = tol.LinearModel(ana.with_compensators([], np.zeros((len(ana.y0), 0)),
                                                    np.zeros((0, 0))), model.J, model.G)
    assert model.mean_square() < none.mean_square()


def test_quadratic_weights(setup):
    system, ab, feat = setup
    le = budget.LithoErrors(ab, [feat], cd=5.0, placement=10.0, focus=50.0, zernike_weight=0.05)
    k, j = 2, 9
    i = list(ab.terms).index(j)
    Q = feat.sensitivities[k].quadratic["cd"][list(feat.sensitivities[k].terms).index(j)]
    assert le.weights[k, i] == pytest.approx(np.sqrt(0.05**2 + abs(Q)/10.0))
    assert np.allclose(le.weights[:, -2:], 0.05)
    y = np.random.default_rng(1).normal(size=len(X_SLIT)*ab.block_size)
    assert np.allclose(le.unpack(le.from_aberrations(y))[0], y)


def test_curved_slit_of_a_ring_field():
    # a rotationally symmetric lens: on an arc around the axis the Zernikes
    # only rotate, and the middle of the arc is the straight slit at y = R
    system = caldwell_krf_lens()
    R = 10.0
    arc = budget.SlitAberrations(system, [0.0, 6.0], arc_radius=R, terms=range(4, 10),
                                 scanner_correction=False)
    straight = budget.SlitAberrations(system, [0.0], y=(R,), terms=range(4, 10),
                                      scanner_correction=False)
    assert np.allclose(arc.nominal_zernikes[0], straight.nominal_zernikes[0], atol=1e-9)
    Z = arc.nominal_zernikes
    t = list(arc.terms)
    for j in (4, 9):   # rotationally symmetric terms stay
        assert np.isclose(Z[0, t.index(j)], Z[1, t.index(j)], atol=2e-3)
    for a, b in ((5, 6), (7, 8)):   # astigmatism and coma keep their magnitude
        assert np.isclose(np.hypot(Z[0, t.index(a)], Z[0, t.index(b)]),
                          np.hypot(Z[1, t.index(a)], Z[1, t.index(b)]), rtol=2e-2, atol=2e-3)
    # the straight slit at x = 6 is farther from the axis: more astigmatism
    off = budget.SlitAberrations(system, [6.0], y=(R,), terms=range(4, 10), scanner_correction=False)
    assert not np.allclose(off.nominal_zernikes[0], Z[1], atol=1e-3)
