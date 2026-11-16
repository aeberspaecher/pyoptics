#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.scatterometry."""

import numpy as np
import pytest

from pyoptics import scatterometry as sc
from pyoptics import ellipsometry as el


def test_slices_of_a_trapezoid():
    z, c, w, t = sc.trapezoid_slices(0.05, 0.1, 80.0, n_slices=10)
    assert np.sum(t) == pytest.approx(0.1)
    assert np.all(np.diff(w) > 0)  # top slice first, narrowest
    assert w[-1] == pytest.approx(0.05 - 2*0.005/np.tan(np.radians(80.0)))
    assert np.allclose(c, 0.0)
    _, c2, w2, _ = sc.trapezoid_slices(0.05, 0.1, 80.0, n_slices=10, asymmetry=4.0)
    assert np.all(c2 < 0)  # the steeper left wall: the top moves towards -x
    _, _, w3, _ = sc.trapezoid_slices(0.05, 0.1, 90.0, n_slices=10, top_rounding=0.02)
    assert w3[0] < w3[-1] == pytest.approx(0.05)
    x, zz = sc.profile_outline(0.05, 0.1, 80.0)
    assert zz.max() == pytest.approx(0.1) and x.min() == pytest.approx(-0.025)


def test_unpatterned_film_matches_thin_films():
    wl = np.array([0.45, 0.6, 0.75])
    for az in (0.0, 30.0):
        J = sc.zero_order_jones(0.3, lambda w: [sc.Layer(0.1, el.sio2_index(w))], wl, 65.0,
                                el.silicon_index, azimuth=az, n_orders=3)
        psi, delta = sc.psi_delta(J)
        p_ref, d_ref = el.psi_delta(wl, 65.0, [(el.sio2_index, 0.1)], el.silicon_index)
        assert psi == pytest.approx(p_ref, abs=1e-8)
        assert delta == pytest.approx(d_ref, abs=1e-8)
        assert np.max(np.abs(J[:, 0, 1])) < 1e-12
        M = sc.mueller_from_jones(J)
        N, C, S = el.ncs_from_psi_delta(p_ref, d_ref)
        assert M[:, 0, 1] == pytest.approx(-N) and M[:, 2, 3] == pytest.approx(S)


def _grating(asymmetry=0.0):
    n_res = lambda w: el.cauchy(w, 1.6, 0.01)
    return lambda w: sc.grating_layers(w, 0.1, 0.05, 0.1, 86.0, n_res,
                                       films=[(el.sio2_index, 0.05)], asymmetry=asymmetry)


def test_grating_symmetries():
    wl = np.array([0.5])
    # planar mount: no conversion between p and s
    J = sc.zero_order_jones(0.1, _grating(), wl, 65.0, el.silicon_index, n_orders=8)
    assert np.max(np.abs(J[:, 0, 1])) < 1e-12 and np.max(np.abs(J[:, 1, 0])) < 1e-12
    # conical mount, symmetric lines: reciprocity, J_ps = -J_sp (Born & Wolf sign of p)
    J = sc.zero_order_jones(0.1, _grating(), wl, 65.0, el.silicon_index, azimuth=45.0,
                            n_orders=8)
    assert abs(J[0, 0, 1]) > 0.05
    assert J[0, 0, 1] == pytest.approx(-J[0, 1, 0], rel=1e-8)
    # along the lines (azimuth 90): block diagonal Mueller matrix unless asymmetric
    M0 = sc.mueller_from_jones(sc.zero_order_jones(0.1, _grating(), wl, 65.0,
                                                   el.silicon_index, azimuth=90.0, n_orders=8))
    M4 = sc.mueller_from_jones(sc.zero_order_jones(0.1, _grating(4.0), wl, 65.0,
                                                   el.silicon_index, azimuth=90.0, n_orders=8))
    assert np.max(np.abs(M0[0, :2, 2:])) < 1e-10
    assert np.max(np.abs(M4[0, :2, 2:])) > 3e-3


def test_signature_depends_on_cd():
    wl = np.linspace(0.4, 0.8, 9)

    def sig(cd):
        L = lambda w: sc.grating_layers(w, 0.1, cd, 0.1, 88.0, el.silicon_index)
        return sc.psi_delta(sc.zero_order_jones(0.1, L, wl, 65.0, el.silicon_index,
                                                n_orders=8))

    a, b = sig(0.040), sig(0.041)
    assert 0.01 < np.max(np.abs(a[0] - b[0])) < 5.0
