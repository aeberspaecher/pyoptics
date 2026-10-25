#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check Manhattan polygon layouts (pyoptics.litho.mask) and the OPC
helpers of pyoptics.litho.opc: fragments, edge placement errors, model-based
OPC, bias tables and SRAFs.
"""

import numpy as np
import pytest

from pyoptics.litho import Illumination, LithoSimulation, MaskLayout, Projector, ThinMask
from pyoptics.litho import mask, opc


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


L_SHAPE = [(-60, -50), (40, -50), (40, -20), (-30, -20), (-30, 60), (-60, 60)]


# polygons:

def test_polygon_union():
    r = mask.polygon_rectangles([L_SHAPE])
    assert np.isclose(sum(a*b for _, _, a, b in r), mask.polygon_area(L_SHAPE)) and len(r) == 2
    # overlapping squares: area of the union
    r = mask.polygon_rectangles([rect(0, 0, 40, 40), rect(20, 20, 60, 60)])
    assert np.isclose(sum(a*b for _, _, a, b in r), 2*1600 - 400)
    with pytest.raises(ValueError):
        mask.polygon_rectangles([[(0, 0), (10, 5), (0, 10), (-5, 5)]])


def test_polygon_helpers():
    P = mask.counterclockwise([(0, 0), (0, 10), (0, 10), (5, 10), (10, 10), (10, 0)])
    assert len(P) == 4 and mask.polygon_area(P) == 100
    Q = mask.offset_polygon(L_SHAPE, 5)
    assert np.isclose(mask.polygon_area(Q), mask.polygon_area(L_SHAPE) + 5*(100 + 110 + 30 + 70 + 30 + 80)
                      + 5*5*(5 - 1))  # perimeter*d + (convex - concave corners)*d**2


def test_polygon_coefficients():
    """Fourier coefficients of a polygon layout against the FFT of the
    sampled transmission; a rectangle polygon equals the rectangle layout."""

    lay = mask.MaskLayout.from_polygons(200, 160, [L_SHAPE])
    m, n = np.arange(-5, 6), np.arange(-4, 5)
    c = lay.region_coefficients(m, n)["feature"]
    N = 1600
    x, y = -100 + 200*(np.arange(N) + 0.5)/N, -80 + 160*(np.arange(N) + 0.5)/N
    t = lay.transmission(x, y, t_clear=0.0, t_absorber=1.0)
    F = np.fft.fft2(t)/N**2
    shift = np.outer(np.exp(-2j*np.pi*n*y[0]/160), np.exp(-2j*np.pi*m*x[0]/200))
    assert np.abs(F[np.ix_(n % N, m % N)]*shift - c).max() < 1e-3
    contact = MaskLayout.contact_holes(150, 60, cd_y=40)
    poly = MaskLayout.from_polygons(150, 150, [rect(-30, -20, 30, 20)], tone="dark")
    c0, c1 = contact.region_coefficients(m, n), poly.region_coefficients(m, n)
    assert all(np.allclose(c0[k], c1[k]) for k in c0)
    assert np.allclose(poly.scaled(bias=4).features[0][2:], (64, 44))
    with pytest.raises(ValueError):
        poly.region_coefficients(m, n, boundary_width=2.0)
    with pytest.raises(ValueError):
        MaskLayout.from_polygons(150, None, [rect(-30, -20, 30, 20)])


# fragments and EPE:

def test_fragments():
    T = MaskLayout.from_polygons(300, 300, [L_SHAPE, rect(60, 60, 110, 70)])
    F = opc.FragmentedLayout(T, max_length=30, corner_length=10)
    assert F.lengths.max() <= 30 + 1e-9 and np.isclose(F.lengths.sum(), 2*(100 + 110) + 2*60)
    assert np.allclose(F.layout(0.0).features, T.features)
    for d in (-3.0, 4.0):  # moving all fragments equally is an offset of the polygons
        for P, Q in zip(F.polygons(d*np.ones(len(F))), T.polygons):
            assert np.allclose(P, mask.offset_polygon(Q, d))
    with pytest.raises(ValueError):  # touching polygons
        opc.FragmentedLayout(MaskLayout.from_polygons(300, 300, [rect(0, 0, 50, 50), rect(50, 0, 80, 50)]))


@pytest.mark.parametrize("bright", [False, True])
def test_edge_placement_errors(bright):
    """A threshold contour that is a circle of radius 40 around a square of
    half width 30."""

    T = MaskLayout.from_polygons(200, 200, [rect(-30, -30, 30, 30)])
    F = opc.FragmentedLayout(T, corner_length=10)
    x, y = opc.image_grid(T, 1.0)
    X, Y = np.meshgrid(x, y)
    I = 0.3*(X**2 + Y**2)/40**2
    if bright:
        I = 0.6 - I
    epe = opc.edge_placement_errors(I, x, y, F.sites, F.normal, 0.3, bright)
    # along a normal from a site (s, 30): edge at sqrt(40**2 - s**2) - 30
    s = np.abs(np.where(F.normal[:, 0] != 0, F.sites[:, 1], F.sites[:, 0]))
    assert np.allclose(epe, np.sqrt(40**2 - s**2) - 30, atol=0.05)
    I_low = np.full_like(I, 0.1 if not bright else 0.5)  # feature everywhere: no edge found
    assert np.all(opc.edge_placement_errors(I_low, x, y, F.sites, F.normal, 0.3, bright, 20) == 20)


# model-based OPC:

@pytest.fixture(scope="module")
def line_end_sim():
    T = MaskLayout.from_polygons(200, 500, [rect(-25, -215, 25, -35), rect(-25, 35, 25, 215)])
    sim = LithoSimulation(T, ThinMask.att_psm(), Illumination.annular(0.9, 0.6, n_samples=9,
                                                                          polarization="unpolarized"),
                          Projector(1.35, 193.0, n=1.437))
    dose = sim.with_(layout=MaskLayout.lines_spaces(200, 50)).dose_to_size(50, 0.3)

    return sim, dose


def test_imagers_agree(line_end_sim):
    sim, dose = line_end_sim
    epe = [opc.model_opc(sim, dose=dose, max_iter=0, imager=im).epe for im in ("abbe", "hopkins", "socs")]
    assert np.allclose(epe[0], epe[1], atol=1e-6) and np.allclose(epe[0], epe[2], atol=0.1)
    ends = opc.FragmentedLayout(sim.layout).sites[:, 0] == 0  # middle of the line ends
    assert np.all(epe[0][ends] < -10)  # line end pull back


def test_model_opc(line_end_sim):
    sim, dose = line_end_sim
    r = opc.model_opc(sim, dose=dose)
    edge = ~r.fragments.corner
    assert np.abs(r.epe[edge]).max() < 3 and r.history[-1, 1] < 0.3*r.history[0, 1]
    ends = r.fragments.sites[:, 0] == 0
    # the line ends are extended, all by the same amount (symmetry)
    assert ends.sum() == 4 and np.all(r.offsets[ends] > 5)
    assert np.allclose(r.offsets[ends], r.offsets[ends][0], atol=0.1)
    local = opc.model_opc(sim, dose=dose, method="local", max_iter=5)
    assert local.history[-1, 1] < local.history[0, 1]
    with pytest.raises(ValueError):
        opc.model_opc(sim, method="newton")


# rule-based OPC and SRAFs:

@pytest.fixture(scope="module")
def lines_sim():
    return LithoSimulation(MaskLayout.lines_spaces(110, 45), ThinMask.att_psm(),
                           Illumination.annular(0.95, 0.7, n_samples=9, polarization="TE"),
                           Projector(1.35, 193.0, n=1.437))


def test_bias_table(lines_sim):
    pitches = [110, 200, 400]
    bias, dose = opc.bias_table(lines_sim, pitches, 45.0)
    assert abs(bias[0]) < 1e-2 and np.all(np.diff(bias) > 0)
    for p, b in zip(pitches, bias):
        assert np.isclose(lines_sim.with_(layout=MaskLayout.lines_spaces(p, 45 + b)).cd(0.3, dose), 45,
                          atol=0.01)


def test_srafs(lines_sim):
    L = opc.insert_srafs(MaskLayout.lines_spaces(400, 45), 20, 100, count=2)
    assert sorted(f[0] for f in L.features[1:]) == [-100, 100, 200]  # the two at 200 merged
    L = opc.insert_srafs(MaskLayout.lines_spaces(400, 45, orientation="x"), 20, 100, count=1)
    assert sorted(f[1] for f in L.features[1:]) == [-100, 100]
    assert len(opc.insert_srafs(MaskLayout.lines_spaces(150, 45), 20, 100).features) == 1  # no room
    sim = lines_sim.with_(layout=opc.insert_srafs(MaskLayout.lines_spaces(400, 45), 20, 100))
    dose = sim.dose_to_size(45, 0.3)
    assert opc.sraf_margin(sim, 0.3, dose, foci=(-50, 0, 50)) > 0
    wide = lines_sim.with_(layout=opc.insert_srafs(MaskLayout.lines_spaces(400, 45), 45, 130))
    assert opc.sraf_margin(wide, 0.3, dose) < 0  # as wide as the line: prints
    with pytest.raises(ValueError):
        opc.insert_srafs(MaskLayout.contact_holes(200, 50), 20, 100)
