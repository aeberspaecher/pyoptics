#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check pyoptics.polygons: moments, the exact Fourier transform of a
polygon, point containment and clipping."""

import numpy as np
import pytest

from pyoptics import polygons as pg


PENTAGON = np.array([[0, 0], [3, 0.5], [2, 2], [0.5, 1.5], [1, 0.8]])  # concave


def numeric_ft(V, kx, ky, n=1601):
    lo, hi = V.min(axis=0) - 0.1, V.max(axis=0) + 0.1
    x = np.linspace(lo[0], hi[0], n)
    y = np.linspace(lo[1], hi[1], n)
    X, Y = np.meshgrid(x, y)
    inside = pg.contains(V, X, Y)
    return np.sum(np.exp(-1j*(kx*X + ky*Y))*inside)*(x[1] - x[0])*(y[1] - y[0])


def test_area_moments():
    sq = [(1, 2), (3, 2), (3, 5), (1, 5)]
    A, S, J = pg.polygon_moments(sq)
    assert A == pytest.approx(6) and np.allclose(S, [12, 21])
    assert J[0, 0] == pytest.approx(3*(27 - 1)/3) and J[0, 1] == pytest.approx(2*21/2*3/3*2)
    assert pg.polygon_area(sq[::-1]) == pytest.approx(-6)
    assert np.allclose(pg.polygon_centroid(sq), [2, 3.5])
    # closing vertex is dropped
    assert pg.polygon_area(sq + [sq[0]]) == pytest.approx(6)


@pytest.mark.parametrize("k", [(1.3, -0.7), (0.0, 2.1), (4.0, 3.0)])
def test_ft_against_quadrature(k):
    F = pg.polygon_ft(PENTAGON, *k)
    assert abs(F - numeric_ft(PENTAGON, *k)) < 2e-3*pg.polygon_area(PENTAGON)


def test_ft_orientation_offset_and_small_k():
    k = np.array([0.0, 1e-9, 1e-6, 1e-4, 1e-2, 1.0])
    F = pg.polygon_ft(PENTAGON, k, 0.6*k)
    assert np.allclose(F, pg.polygon_ft(PENTAGON[::-1], k, 0.6*k), rtol=1e-12, atol=0)
    assert F[0] == pytest.approx(pg.polygon_area(PENTAGON))
    # smooth across the switch to the Taylor series near k = 0
    size = np.max(np.hypot(*(PENTAGON - PENTAGON.mean(axis=0)).T))
    kk = np.array([0.999, 1.001])*1e-4/size/np.hypot(1, 0.6)
    F2 = pg.polygon_ft(PENTAGON, kk, 0.6*kk)
    c = pg.polygon_centroid(PENTAGON)
    ref = pg.polygon_area(PENTAGON)*np.exp(-1j*(kk*c[0] + 0.6*kk*c[1]))
    assert np.allclose(F2, ref, rtol=1e-8)
    # shift theorem, also far from the origin
    d = np.array([1e5, -3e4])
    k = np.linspace(0.1, 3, 7)
    assert np.allclose(pg.polygon_ft(PENTAGON + d, k, -k),
                       pg.polygon_ft(PENTAGON, k, -k)*np.exp(-1j*(k*d[0] - k*d[1])), atol=1e-9)


def test_ft_rectangle_and_keyhole():
    a, b = 3.0, 2.0
    rect = [(-a/2, -b/2), (a/2, -b/2), (a/2, b/2), (-a/2, b/2)]
    kx, ky = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(-4, 4, 9))
    exact = a*b*np.sinc(kx*a/(2*np.pi))*np.sinc(ky*b/(2*np.pi))
    assert np.allclose(pg.polygon_ft(rect, kx, ky), exact, atol=1e-12)
    # square with a square hole, as a keyhole polygon (the cut cancels)
    outer = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    keyhole = [(-2, -2), (2, -2), (2, 2), (-2, 2), (-2, 0), (-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1),
               (-1, 0), (-2, 0)]
    hole = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    assert pg.polygon_area(keyhole) == pytest.approx(12)
    assert np.allclose(pg.polygon_ft(keyhole, kx, ky),
                       pg.polygon_ft(outer, kx, ky) - pg.polygon_ft(hole, kx, ky), atol=1e-12)
    assert np.allclose(pg.polygons_ft([outer, hole], kx, ky),
                       pg.polygon_ft(outer, kx, ky) + pg.polygon_ft(hole, kx, ky))


def test_circle_polygon():
    V = pg.circle_polygon(1.0, n=256)
    assert pg.polygon_area(V) == pytest.approx(np.pi)
    # Airy amplitude 2 pi J1(k)/k
    from scipy.special import j1
    k = np.array([0.5, 2.0, 5.0])
    assert np.allclose(pg.polygon_ft(V, k, 0*k), 2*np.pi*j1(k)/k, atol=1e-3)


def test_contains_half_open():
    left = [(0, 0), (1, 0), (1, 1), (0, 1)]
    right = [(1, 0), (2, 0), (2, 1), (1, 1)]
    tri_a = [(0, 0), (2, 0), (2, 1)]
    tri_b = [(0, 0), (2, 1), (0, 1)]
    x, y = np.meshgrid(np.linspace(0.05, 1.95, 39), np.linspace(0.05, 0.95, 19))
    x = np.concatenate([x.ravel(), [1.0, 1.0, 1.5]])
    y = np.concatenate([y.ravel(), [0.5, 0.25, 0.75]])
    n = pg.contains(left, x, y).astype(int) + pg.contains(right, x, y)
    assert np.all(n == 1)
    n = pg.contains(tri_a, x, y).astype(int) + pg.contains(tri_b, x, y)
    assert np.all(n == 1)
    assert pg.contains(PENTAGON, 1.0, 1.0) and not pg.contains(PENTAGON, 0.3, 0.6)


def test_clip():
    sq = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    C = pg.clip_to_rectangle(sq, -1, 0, 5, 5)
    assert pg.polygon_area(C) == pytest.approx(6)
    assert pg.clip_to_rectangle(sq, 3, 3, 4, 4) is None
    # a U shape cut by the window into two legs joined along the border
    U = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
    C = pg.clip_to_rectangle(U, -1, 2, 4, 4)
    assert abs(pg.polygon_area(C)) == pytest.approx(2)
    k = np.linspace(0.3, 3, 5)
    legs = [[(0, 2), (1, 2), (1, 3), (0, 3)], [(2, 2), (3, 2), (3, 3), (2, 3)]]
    assert np.allclose(pg.polygon_ft(C, k, 0.4*k), pg.polygons_ft(legs, k, 0.4*k))
    # a triangle cut by a corner of the window
    T = pg.clip_to_rectangle([(0, 0), (4, 0), (0, 4)], 1, 1, 5, 5)
    assert pg.polygon_area(T) == pytest.approx(2)
