#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.segmented: segmented apertures, their far fields
against the polygon Fourier transform and the array factor, and phasing."""

import numpy as np
import pytest

from pyoptics import segmented as sg
from pyoptics.coronagraph import mft, pupil_grid

LAM = 2e-6


@pytest.fixture(scope="module")
def jwst():
    x = pupil_grid(128, 6.8)
    return x, sg.jwst_like_pupil(x, x)


def test_hex_centres():
    C = sg.hex_centres(2, 1.0)
    assert len(C) == 18
    r = np.sort(np.hypot(*C.T))
    assert np.allclose(r, [1]*6 + [np.sqrt(3)]*6 + [2]*6)
    d = np.hypot(*(C[:, None] - C[None]).T)
    assert np.min(d[d > 0]) == pytest.approx(1.0)      # no overlaps
    assert len(sg.hex_centres(1, 1.0, skip_centre=False)) == 7


def test_polygon_mask_and_ft():
    flat = 1.32
    V = sg.hexagon_vertices((0.1, -0.05), flat, 0.2)
    x = pupil_grid(200, 1.8)
    M = sg.polygon_mask(x, x, V, 8)
    dx = x[1] - x[0]
    area = np.sqrt(3)/2*flat**2
    assert M.sum()*dx**2 == pytest.approx(area, rel=1e-3)
    assert complex(sg.polygon_ft(V, 0.0, 0.0)) == pytest.approx(area)
    u = np.linspace(-5, 5, 41)
    U, W = np.meshgrid(u, u)
    F = mft(M, x, x, u, u)
    Fa = sg.polygon_ft(V, 2*np.pi*U, 2*np.pi*W)
    assert np.max(np.abs(F - Fa)) < 1e-3*area
    X, Y = np.meshgrid(x - 0.1, x + 0.05)
    rms = np.sqrt(np.sum(M*(X**2 + Y**2))/M.sum())
    assert rms == pytest.approx(sg.hexagon_rms_radius(flat), rel=2e-3)


def test_array_factor(jwst):
    x, P = jwst
    th = np.linspace(-1.5e-6, 1.5e-6, 31)
    E = P.masks.sum(0)                                  # no struts
    F = sg.far_field(E, x, x, th, th, LAM)
    T, S = np.meshgrid(th, th)
    k = 2*np.pi/LAM
    Fa = sg.polygon_ft(sg.hexagon_vertices((0, 0), P.flat), k*T, k*S)*sg.array_factor(P.centres, k*T, k*S)/LAM
    assert np.max(np.abs(F - Fa)) < 1e-2*np.abs(Fa).max()       # pixelized gaps


def test_strut_area():
    x = pupil_grid(400, 2.0)
    T = sg.strut_mask(x, x, [0.3, 2.0], 0.05, 0.8, 8)
    assert np.sum(1 - T)*(x[1] - x[0])**2 == pytest.approx(2*0.05*0.8, rel=0.02)


def test_piston_and_tilt_on_axis(jwst):
    x, P = jwst
    n = len(P.centres)
    rng = np.random.default_rng(0)
    p = rng.normal(0, 0.03*LAM, n)
    E0 = sg.segment_field(P, x, x, LAM)
    E = sg.segment_field(P, x, x, LAM, piston=p)
    S = np.abs(sg.far_field(E, x, x, [0.0], [0.0], LAM))**2/np.abs(sg.far_field(E0, x, x, [0.0], [0.0], LAM))**2
    A = (P.masks*P.struts).sum((1, 2))
    exact = np.abs(np.sum(A*np.exp(2j*np.pi*p/LAM)))**2/A.sum()**2
    assert S[0, 0] == pytest.approx(exact, rel=1e-6)
    sigma = 2*np.pi*np.sqrt(np.average((p - np.average(p, weights=A))**2, weights=A))/LAM
    assert exact == pytest.approx(np.exp(-sigma**2), rel=0.01)
    # a tilted segment contributes its hexagon's transform at k = -2 pi t/lambda
    t = 0.3e-6
    tip = np.zeros(n)
    tip[4] = t
    single = sg.SegmentedPupil(P.masks[4:5], P.centres[4:5], P.flat, P.angle, np.ones_like(P.struts),
                               P.masks[4])
    F = sg.far_field(sg.segment_field(single, x, x, LAM, tip=[t]), x, x, [0.0], [0.0], LAM)[0, 0]
    Fa = sg.polygon_ft(sg.hexagon_vertices((0, 0), P.flat), -2*np.pi*t/LAM, 0.0)/LAM
    assert abs(F) == pytest.approx(abs(complex(Fa)), rel=0.01)


def test_dispersed_fringes(jwst):
    x, P = jwst
    wl = np.linspace(1.0e-6, 2.0e-6, 60)
    th = np.linspace(-2e-6, 2e-6, 81)
    for p in (4e-6, -11e-6):
        I, B = sg.dispersed_fringes(P, x, x, p, wl, th)
        assert B == pytest.approx(P.flat + 0.007)
        assert sg.dfs_piston(I, wl, th, B, 30e-6) == pytest.approx(p, rel=0.02)


def test_phase_retrieval(jwst):
    x, P = jwst
    rng = np.random.default_rng(3)
    p = rng.normal(0, 0.08*LAM, len(P.centres))
    E = sg.segment_field(P, x, x, LAM, piston=p)
    psfs, phases = sg.diversity_psfs(E, [-4, 4], 6.6, x, x)
    phase, err = sg.misell(psfs, phases, P.pupil, 200)
    assert err[-1] < err[0]/5
    q, _, _ = sg.fit_segments(phase, P, x, x, LAM)
    d = (q - p) - np.mean(q - p)
    assert np.std(d) < 0.01*LAM
