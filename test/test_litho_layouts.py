#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check general (non-Manhattan) polygon layouts of pyoptics.litho.mask
and the import of polygons from layout libraries (gds_polygons(),
MaskLayout.from_gds())."""

import numpy as np
import pytest

from pyoptics import polygons as pg
from pyoptics.litho import FDTDMask, Illumination, LithoSimulation, MaskLayout, Projector, ThinMask
from pyoptics.litho import mask, opc


PROJ = Projector(1.35, 193.0, n=1.437)


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_triangles_equal_rectangle():
    # a rectangle cut along its diagonal into two triangles: the polygon
    # transform against the exact rectangle decomposition, also for images
    R = MaskLayout.from_polygons(300, 260, [rect(-60, -20, 40, 50), rect(70, 60, 110, 120)])
    T = MaskLayout.from_polygons(300, 260, [[(-60, -20), (40, -20), (40, 50)], [(-60, -20), (40, 50), (-60, 50)],
                                            rect(70, 60, 110, 120)])
    assert R.manhattan and not T.manhattan
    m, n = np.arange(-4, 5), np.arange(-3, 4)
    cR, cT = R.region_coefficients(m, n), T.region_coefficients(m, n)
    for key in ("feature", "field"):
        assert np.allclose(cR[key], cT[key], atol=1e-14)
    ill = Illumination.annular(0.9, 0.6, n_samples=7)
    x = np.linspace(-150, 150, 31)
    I_R = LithoSimulation(R, ThinMask.att_psm(), ill, PROJ).image(x, x)
    I_T = LithoSimulation(T, ThinMask.att_psm(), ill, PROJ).image(x, x)
    assert np.allclose(I_R, I_T, atol=1e-12)


def test_transmission_and_coefficients():
    V = pg.regular_polygon(8, 60, center=(150, -40), angle=np.pi/8)  # partly outside the period
    L = MaskLayout.from_polygons(400, 300, [V], tone="dark")
    x = -200 + (np.arange(400) + 0.5)
    y = -150 + (np.arange(300) + 0.5)
    t = L.transmission(x, y)
    assert t.mean() == pytest.approx(pg.polygon_area(V)/(400*300), rel=1e-2)  # pixel sampling
    c = L.region_coefficients([0, 1], [0, 2])
    assert c["feature"][0, 0] == pytest.approx(pg.polygon_area(V)/(400*300))
    assert c["field"][0, 0] == pytest.approx(1 - pg.polygon_area(V)/(400*300))
    # first-order coefficient against the FFT of the (finely) sampled transmission
    T = np.fft.fft2(t)/t.size
    assert abs(c["feature"][1, 1] - T[2, 1]*np.exp(-2j*np.pi*(1*(-200 + 0.5)/400 + 2*(-150 + 0.5)/300))) < 2e-4
    # reference feature: bounding box center, chords through it
    x0, y0, a, b = L.features[0]
    assert (x0, y0) == pytest.approx((150, -40))
    assert a == pytest.approx(120*np.cos(np.pi/8)) and b == pytest.approx(a)


def test_rotated_contact_conventional():
    # with a rotationally symmetric source, a contact rotated by 45 degrees
    # images like the axis-parallel one, rotated (up to the small
    # interaction with the neighbours of the square lattice)
    sq = pg.regular_polygon(4, 60/np.sqrt(2), angle=np.pi/4)
    diamond = pg.regular_polygon(4, 60/np.sqrt(2))
    ill = Illumination.conventional(0.4, n_samples=9)
    sim = LithoSimulation(MaskLayout.from_polygons(800, 800, [sq], tone="dark"), ThinMask.binary(), ill, PROJ)
    simd = sim.with_(layout=MaskLayout.from_polygons(800, 800, [diamond], tone="dark"))
    u = np.linspace(-80, 80, 33)
    I = sim.image(u, [0.0])[0]
    Id = simd.image(u/np.sqrt(2), u/np.sqrt(2)).diagonal()
    assert np.max(np.abs(I - Id)) < 0.01*I.max()


def test_bias_and_overlaps():
    V = pg.regular_polygon(8, 50)
    L = MaskLayout.from_polygons(400, 400, [V])
    Lb = L.scaled(bias=4.0)
    n, d = 8, 2.0
    perimeter = 8*2*50*np.sin(np.pi/8)
    assert pg.polygon_area(Lb.polygons[0]) == pytest.approx(pg.polygon_area(V) + perimeter*d
                                                            + n*np.tan(np.pi/n)*d**2)
    with pytest.raises(ValueError):
        MaskLayout.from_polygons(400, 400, [V, V + [30, 0]])
    with pytest.raises(ValueError):  # overlap with its own periodic copy
        MaskLayout.from_polygons(90, 400, [V])
    MaskLayout.from_polygons(400, 400, [V, V + [30, 0]], check_overlaps=False)


def test_unsupported_models():
    L = MaskLayout.from_polygons(400, 400, [pg.regular_polygon(6, 50)])
    with pytest.raises(ValueError):
        L.region_coefficients([0], [0], boundary_width=5.0)
    with pytest.raises(ValueError):
        opc.FragmentedLayout(L)
    with pytest.raises(NotImplementedError):
        FDTDMask([(1.0, 0.1, 50.0)]).order_jones(L, [0], [0], [0.0], [0.0], 1.35, 193.0, 4.0)


# layout libraries (duck typed):

class FakeGdstkPolygon(object):
    def __init__(self, points, layer=0, datatype=0):
        self.points, self.layer, self.datatype = np.asarray(points, float), layer, datatype


class FakeGdstkCell(object):
    def __init__(self, polygons):
        self._p = polygons

    def get_polygons(self, apply_repetitions=True, include_paths=True, depth=None, layer=None,
                     datatype=None):
        return list(self._p)


class FakeGdspyPolygonSet(object):
    def __init__(self, polygons, layers, datatypes):
        self.polygons, self.layers, self.datatypes = polygons, layers, datatypes


class FakeGdspyCell(object):
    def __init__(self, by_spec):
        self._d = by_spec

    def get_polygons(self, by_spec=False, depth=None):
        if by_spec:
            return self._d
        return [P for v in self._d.values() for P in v]


def test_gds_polygons():
    a, b = np.array(rect(0, 0, 1, 2)), np.array([(2, 0), (3, 1), (2, 2)])
    cell = FakeGdstkCell([FakeGdstkPolygon(a, 1, 0), FakeGdstkPolygon(b, 2, 0), FakeGdstkPolygon(a + 5, 1, 7)])
    assert len(mask.gds_polygons(cell)) == 3
    assert len(mask.gds_polygons(cell, layer=1)) == 2
    assert len(mask.gds_polygons(cell, layer=1, datatype=0)) == 1
    assert len(mask.gds_polygons(cell, layer=[1, 2], datatype=0)) == 2
    gcell = FakeGdspyCell({(1, 0): [a, a + 5], (3, 0): [b]})
    assert len(mask.gds_polygons(gcell, layer=1)) == 2
    assert len(mask.gds_polygons({(1, 0): [a], (2, 0): [b]}, layer=2)) == 1
    assert len(mask.gds_polygons(FakeGdspyPolygonSet([a, b], [1, 4], [0, 0]), layer=4)) == 1
    assert len(mask.gds_polygons([a, b])) == 2 and len(mask.gds_polygons(a)) == 1
    assert len(mask.gds_polygons(np.array([a, a + 3]))) == 2
    with pytest.raises(ValueError):
        mask.gds_polygons([1.0, 2.0])


def test_from_gds_window():
    # a contact array cell in um; the window is one cell, the layout in nm
    cell = FakeGdspyCell({(1, 0): [np.array(rect(-0.04, -0.04, 0.04, 0.04)) + [0.1, 0.1]],
                          (2, 0): [np.array([(0.0, 0.0), (0.3, 0.0), (0.0, 0.3)])]})
    L = MaskLayout.from_gds(cell, (0.0, 0.0, 0.2, 0.2), layer=1, tone="dark", scale=1000)
    assert L.manhattan and (L.pitch_x, L.pitch_y) == pytest.approx((200, 200))
    assert L.features[0] == pytest.approx((0.0, 0.0, 80.0, 80.0))
    ref = MaskLayout.contact_holes(200, 80)
    m = np.arange(-2, 3)
    assert np.allclose(L.region_coefficients(m, m)["feature"], ref.region_coefficients(m, m)["feature"])
    # the triangle is clipped to the window
    T = MaskLayout.from_gds(cell, (0.0, 0.0, 0.2, 0.2), layer=2, scale=1000)
    assert not T.manhattan
    assert pg.polygon_area(T.polygons[0]) == pytest.approx(0.2*0.2*1e6 - 0.5*0.1*0.1*1e6)
    with pytest.raises(ValueError):
        MaskLayout.from_gds(cell, (1.0, 1.0, 2.0, 2.0), layer=1)


def test_gdstk_if_available():
    gdstk = pytest.importorskip("gdstk")
    lib = gdstk.Library()
    c = lib.new_cell("C")
    c.add(gdstk.rectangle((0, 0), (0.05, 0.2), layer=1))
    c.add(gdstk.regular_polygon((0.3, 0.3), 0.04, 8, layer=2))
    c.add(gdstk.FlexPath([(0, 0.5), (0.3, 0.8)], 0.04, layer=1))
    top = lib.new_cell("T")
    top.add(gdstk.Reference(c, (0, 0), columns=2, rows=1, spacing=(1, 1)))
    assert len(mask.gds_polygons(top)) == 6
    L = MaskLayout.from_gds(top, (-0.1, -0.1, 0.5, 0.5), layer=2, tone="dark", scale=1000)
    assert pg.polygon_area(L.polygons[0]) == pytest.approx(gdstk.regular_polygon((0, 0), 0.04, 8).area()*1e6)
