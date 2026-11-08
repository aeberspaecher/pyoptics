#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the schlieren and knife-edge tools against analytical results: the
Hilbert transform image of a knife at the focus, the knife-edge rule for
defocus, the zonal test of a paraboloid, Toepler's contrast, Ronchi fringe
spacing, the index of air and the shadowgraph of a paraboloidal phase.
"""

import numpy as np
import pytest

from pyoptics import schlieren as sch
from pyoptics.utils import grid2d, pad


WL = 0.55E-3     # mm
D, R = 150.0, 2400.0
K0 = 2*np.pi/WL


@pytest.fixture(scope="module")
def pupil():
    x, y, X, Y = grid2d(-0.65*D, 0.65*D, -0.65*D, 0.65*D, 256)
    return x, y, X, Y, (np.hypot(X, Y) <= D/2).astype(float)


def test_focal_coordinates_tilt(pupil):
    # a tilt W = alpha*x focuses at xi = f*alpha: the knife just below passes it
    x, y, X, Y, A = pupil
    alpha = 1E-4
    xi, eta = sch.focal_coordinates(x, y, WL, R)
    assert xi[len(x)//2] == 0.0
    field = A*np.exp(1j*K0*alpha*X)
    below = np.abs(sch.filtered_image(field, x, y, WL, R,
                                      lambda XI, ETA: sch.knife_edge(XI, ETA, 0.5*R*alpha)))**2
    above = np.abs(sch.filtered_image(field, x, y, WL, R,
                                      lambda XI, ETA: sch.knife_edge(XI, ETA, 1.5*R*alpha)))**2
    inside = np.hypot(X, Y) < 0.4*D
    assert below[inside].mean() > 0.97
    assert above[inside].mean() < 0.03


def test_knife_at_focus_hilbert():
    # rect pupil (separable): I = (1 + ln^2|(x - a1)/(x - a2)|/pi^2)/4 inside, the
    # log term alone outside; padded to make the periodic window negligible
    x, y, X, Y = grid2d(-100, 100, -100, 100, 256)
    A = ((np.abs(X) < 40) & (np.abs(Y) < 40)).astype(float)
    Ap, xp, yp = pad(A, x, y, factor=8)
    img = np.abs(sch.filtered_image(Ap, xp, yp, WL, R, sch.knife_edge))**2
    row = img[len(yp)//2]
    dx = x[1] - x[0]
    cols = np.where(Ap[len(yp)//2] > 0)[0]
    lo, hi = xp[cols[0]] - dx/2, xp[cols[-1]] + dx/2
    ana = (np.where((xp > lo) & (xp < hi), 0.25, 0.0)
           + np.log(np.abs((xp - lo)/(xp - hi)))**2/(4*np.pi**2))
    sel = (np.minimum(np.abs(xp - lo), np.abs(xp - hi)) > 8*dx) & (np.abs(xp) < 80)
    assert np.max(np.abs(row[sel] - ana[sel])) < 0.01


@pytest.mark.parametrize("delta, dark", [(-3.0, "left"), (3.0, "right")])
def test_knife_rule(pupil, delta, dark):
    # inside focus the shadow comes from the side of the knife (left), outside
    # focus from the other side
    x, y, X, Y, A = pupil
    W = sch.defocus_wavefront(X, Y, R, delta)
    img = np.abs(sch.filtered_image(A*np.exp(1j*K0*W), x, y, WL, R, sch.knife_edge))**2
    left = img[(X < -20) & (X > -60) & (np.abs(Y) < 20)].mean()
    right = img[(X > 20) & (X < 60) & (np.abs(Y) < 20)].mean()
    assert (left < 0.05 and right > 0.9) if dark == "left" else (right < 0.05 and left > 0.9)
    geo = sch.geometric_image(W, x, y, R, sch.knife_edge)
    assert geo[len(y)//2, len(x)//2 - 40] == (0.0 if dark == "left" else 1.0)


@pytest.mark.parametrize("rz", [30.0, 60.0])
def test_couder_zone_paraboloid(pupil, rz):
    # two windows at +-rz: equal brightness when the knife is at the zone's
    # centre of curvature, rz**2/R behind the paraxial one
    x, y, X, Y, A = pupil
    W = sch.conic_wavefront_at_center(X, Y, R, -1.0)
    win = (np.abs(np.abs(X) - rz) < 5) & (np.abs(Y) < 12)
    d0 = sch.zonal_offset(rz, R, -1.0)
    assert d0 == pytest.approx(rz**2/R)
    assert sch.zonal_offset(rz, R, -1.0, moving_source=True) == pytest.approx(rz**2/(2*R))

    def imbalance(d):
        img = np.abs(sch.filtered_image(win*np.exp(1j*K0*(W + sch.defocus_wavefront(X, Y, R, d))),
                                        x, y, WL, R, sch.knife_edge))**2
        return img[win & (X < 0)].mean() - img[win & (X > 0)].mean()

    assert imbalance(d0 - 0.1) < 0 < imbalance(d0 + 0.1)


def test_toepler_contrast_extended_source(pupil):
    # a slow ripple seen with a slit source of half width a: dI/I = f eps/a
    x, y, X, Y, A = pupil
    period, w, a = 50.0, 2E-5, 0.05
    W = w*np.cos(2*np.pi*X/period)
    offsets = np.linspace(-a, a, 41)
    img = sch.extended_source_image(A*np.exp(1j*K0*W), x, y, WL, R, sch.knife_edge, offsets)
    geo = sch.geometric_image(W, x, y, R, sch.knife_edge, offsets)
    eps_x, _ = sch.transverse_aberration(W, x, y, 1.0)
    inside = np.hypot(X, Y) < 0.35*D
    pred = 0.5*(1 + sch.knife_edge_contrast(eps_x, R, a))
    assert np.max(np.abs(img[inside] - pred[inside])) < 0.04
    assert np.max(np.abs(geo[inside] - pred[inside])) < 0.02
    assert sch.knife_edge_contrast(1.0, R, a) == 1.0


@pytest.mark.parametrize("a, period", [(0.0, 10.0), (0.1, 6.0), (0.1, 40.0)])
def test_knife_edge_transfer(pupil, a, period):
    # linear theory against the incoherent sum: min(xi_1/a, 1) k w, 2 k w for a point
    x, y, X, Y, A = pupil
    w = WL/300
    W = w*np.cos(2*np.pi*X/period)
    field = A*np.exp(1j*K0*W)
    if a == 0:
        img = np.abs(sch.filtered_image(field, x, y, WL, R, sch.knife_edge))**2
    else:
        img = sch.extended_source_image(field, x, y, WL, R, sch.knife_edge, np.linspace(-a, a, 61))
    inside = np.hypot(X, Y) < 0.3*D
    rel = img[inside]/img[inside].mean() - 1
    lin = sch.knife_edge_transfer(W, x, y, WL, R, a)[inside]
    expected = (2.0 if a == 0 else min(WL*R/period/a, 1.0))*K0*w
    assert np.max(np.abs(lin)) == pytest.approx(expected, rel=0.02)
    # fitted amplitude (the rim diffraction of the pupil adds a slow background)
    fit = np.sum(rel*lin)/np.sum(lin*lin)
    assert fit == pytest.approx(1.0, abs=0.05)


def test_ronchi_fringes_sphere(pupil):
    # defocused sphere: straight fringes with the pupil period p R/delta
    x, y, X, Y, A = pupil
    p, delta = 0.254, 20.0
    W = sch.defocus_wavefront(X, Y, R, delta)
    geo = sch.geometric_image(W, x, y, R, lambda XI, ETA: sch.ronchi_ruling(XI, ETA, p))
    row = geo[len(y)//2]*A[len(y)//2]
    edges = np.where(np.diff(row[np.abs(x) < 0.45*D]) != 0)[0]
    spacing = np.diff(x[np.abs(x) < 0.45*D][edges]).mean()*2
    assert spacing == pytest.approx(p*R/delta, rel=0.03)
    t = sch.ronchi_ruling(np.array([0.0, p/4 - 1E-6, p/4 + 1E-6, p]), 0.0, p)
    assert list(t) == [1.0, 1.0, 0.0, 1.0]


def test_filters():
    xi = np.array([-1.0, 0.0, 0.2, 1.0])
    assert list(sch.knife_edge(xi, 0*xi)) == [0.0, 0.5, 1.0, 1.0]
    assert list(sch.knife_edge(0*xi, xi, angle=np.pi/2)) == [0.0, 0.5, 1.0, 1.0]
    assert list(sch.slit(xi, 0*xi, 0.5)) == [0.0, 1.0, 1.0, 0.0]
    assert list(sch.wire(xi, 0*xi, 0.5)) == [1.0, 0.0, 0.0, 1.0]


def test_air_index():
    assert sch.air_index(288.15) - 1 == pytest.approx(2.77E-4, rel=0.01)
    # hot air: n - 1 scales with 1/T
    assert (sch.air_index(576.3) - 1)/(sch.air_index(288.15) - 1) == pytest.approx(0.5)


def test_shadowgraph_paraboloid(pupil):
    # W = c r^2: laplace(W) = 4c everywhere
    x, y, X, Y, A = pupil
    c = 1E-6
    C = sch.shadowgraph_contrast(c*(X**2 + Y**2), x, y, 500.0)
    assert np.allclose(C[5:-5, 5:-5], -500.0*4*c)
