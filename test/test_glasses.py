#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the glass catalogues: dispersion formulas against the catalogue
values, partial dispersion and the normal line, search, use as a material
and in the optimizer."""

import numpy as np
import pytest

from pyoptics import glasses as gl
from pyoptics import lens_optimization as lo
from pyoptics import raytrace as rt


@pytest.mark.parametrize("vendor", gl.VENDORS)
def test_formulas_reproduce_catalogue_nd_and_Vd(vendor):
    c = gl.load(vendor)
    assert len(c) > 100
    for g in c.values():
        assert abs(g.n("d") - g.n_d) < 2e-4, g
        assert abs(g.abbe_number() - g.V_d) < 0.3, g


def test_nbk7_and_normal_line():
    bk7 = gl.get("N-BK7")
    # SCHOTT data sheet: n_d 1.51680, n_F 1.52238, n_C 1.51432, n_g 1.52668
    assert np.isclose(bk7.n("d"), 1.51680, atol=1e-5)
    assert np.isclose(bk7.n("F"), 1.52238, atol=2e-5)
    assert np.isclose(bk7.n("C"), 1.51432, atol=2e-5)
    assert np.isclose(bk7.n("g"), 1.52668, atol=2e-5)
    # N-BK7 is close to the normal line, N-FK51A far above it
    assert abs(bk7.dPgF) < 0.002
    fk51 = gl.get("N-FK51A")
    assert fk51.dPgF > 0.03
    assert np.isclose(fk51.dPgF, fk51.dPgF_catalog, atol=5e-4)


def test_catalog_search_and_filter():
    c = gl.catalog(("schott", "ohara"))
    assert len(c) == len(gl.load("schott")) + len(gl.load("ohara"))
    assert c.nearest(1.5168, 64.17).name == "N-BK7"
    assert gl.catalog("schott").nearest(1.5168, 64.17).name == "N-BK7"
    std = gl.catalog("schott", status=("standard", "preferred"))
    assert 0 < len(std) < len(gl.load("schott"))
    flints = c.filter(V_range=(0, 30))
    assert all(g.V_d <= 30 for g in flints.values())
    assert len(c.nearest(1.6, 40.0, k=5)) == 5


def test_glass_is_a_raytrace_material():
    bk7 = gl.get("N-BK7")
    s = rt.Surface(50.0, 5.0, material=bk7)
    assert np.isclose(s.index(587.56, 1.0), 1.5168, atol=1e-5)
    assert np.isclose(s.index(486.13, 1.0), bk7.n("F"))


def test_snap_to_real_glasses():
    S = [rt.Surface(60.0, 5.0, material=(1.531, 66.2)), rt.Surface(-40.0, 2.5, material=(1.625, 36.1)),
         rt.Surface(-160.0, 95.0, n_after=1.0)]
    system = rt.System(S, 0, 587.56e-6, entrance_pupil_radius=10.0)
    opt = lo.DampedLeastSquares(system, lo.glass(0) + lo.glass(1), [lo.seidel("W040")])
    chosen = opt.snap_glasses(gl.catalog("schott"))
    m = opt.system.surfaces[0].material
    assert isinstance(m, gl.Glass) and m.name == chosen[0]
    # a catalogue glass can become a model glass variable again
    v = lo.GlassIndex(0)
    assert np.isclose(v.get(opt.system), m.n_d)


def test_secondary_spectrum():
    bk7, f2 = gl.get("N-BK7"), gl.get("F2")
    # thin achromat of normal glasses: the g focus is about f/600 from F and C
    d = gl.secondary_spectrum(bk7, f2, f=100.0)
    assert 0.12 < abs(d) < 0.25
    # a fluor crown with a short flint: the partial dispersions match better
    assert abs(gl.secondary_spectrum(gl.get("N-FK51A"), gl.get("N-KZFS11"), 100.0)) < abs(d)/2
