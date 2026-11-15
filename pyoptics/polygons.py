#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Polygons: area and moments, the exact Fourier transform, point
containment and clipping.

A polygon is an (N, 2) array of vertices (x, y). A closing vertex equal to
the first one is dropped. Unless noted otherwise the orientation does not
matter. Simple polygons and weakly simple ones (holes joined to the outline
by a cut, the "keyhole" polygons of GDSII layouts) are allowed: the two
edges of the cut run in opposite directions and cancel in every edge sum
below.

Fourier transform (`polygon_ft()`): with exp(-i k.r) = div(i k exp(-i k.r)
/k^2), the area integral of exp(-i k.r) over the polygon becomes a sum over
its edges (Gauss' theorem). The edge j from a_j to b_j with the vector
e_j = b_j - a_j, the outward normal times length N_j = (e_y, -e_x)
(counterclockwise) and the midpoint m_j contributes ::

    F(k) = i/k^2 sum_j (k.N_j) exp(-i k.m_j) sinc(k.e_j/2),

sinc(x) = sin(x)/x. The transform is exact for any k, without sampling of
the polygon; along the normal of an edge the edge gives a streak with the
amplitude L_j/k (the diffraction spikes of polygonal apertures). The terms
are of the order k L_j each and add up to about k^2 A near k = 0, so for
``k*size`` below 1e-4 the second order Taylor series from the area moments is
used instead. Vertices are taken relative to the vertex mean, so that
polygons far from the origin (a clip from a large layout) do not lose
digits in the phases.

The Fourier coefficients of a periodic pattern of polygons (period P_x x
P_y) are c_mn = sum_p F_p(2 pi m/P_x, 2 pi n/P_y)/(P_x P_y), whether or not
the polygons lie inside one period: the transform of one copy, sampled at
the reciprocal lattice, is the series of the periodized pattern. Polygons
must not overlap (also not with periodic copies of themselves), or the
overlap counts twice; GDSII tools merge them, e.g. gdstk.boolean(polygons,
[], "or").

Functions:

- polygon_area(), polygon_centroid(), polygon_moments(): signed area and
  moments (positive for counterclockwise vertices).
- polygon_ft(), polygons_ft(): exact Fourier transform of one polygon and
  of a set of disjoint polygons.
- contains(): even-odd point in polygon test on arrays of points.
- clip_to_rectangle(): Sutherland-Hodgman clipping to an axis-parallel
  window.
- path_polygon(): outline of a wire of constant width along a polyline
  (miter joins, like a GDSII path with flush ends).
- regular_polygon(), circle_polygon(): helpers.

References
----------
S.-W. Lee and R. Mittra, "Fourier transform of a polygonal shape function
and its application in electromagnetics", IEEE Trans. Antennas Propag. 31,
99-103 (1983).

I. E. Sutherland and G. W. Hodgman, "Reentrant polygon clipping", Commun.
ACM 17, 32-42 (1974).
"""

import numpy as np


__all__ = ["as_polygon", "polygon_area", "polygon_centroid", "polygon_moments", "polygon_ft",
           "polygons_ft", "contains", "clip_to_rectangle", "path_polygon", "regular_polygon",
           "circle_polygon"]


def as_polygon(vertices):
    """Vertices as an (N, 2) float array, without a closing vertex.

    Parameters
    ----------
    vertices : sequence of (x, y)
        At least three vertices.

    Returns
    -------
    V : (N, 2) array
    """

    V = np.array(vertices, dtype=float)
    if V.ndim != 2 or V.shape[1] != 2:
        raise ValueError("a polygon is a sequence of (x, y) vertices")
    if len(V) > 1 and np.all(V[0] == V[-1]):
        V = V[:-1]
    if len(V) < 3:
        raise ValueError("a polygon needs at least three vertices")

    return V


def polygon_area(vertices):
    """Signed area (shoelace formula), positive for counterclockwise vertices.

    Parameters
    ----------
    vertices : (N, 2) array

    Returns
    -------
    A : float
    """

    V = as_polygon(vertices)
    c = V.mean(axis=0)
    x, y = (V - c).T

    return 0.5*float(np.sum(x*np.roll(y, -1) - np.roll(x, -1)*y))


def polygon_moments(vertices, origin=(0.0, 0.0)):
    """Area moments up to second order, relative to `origin`.

    Parameters
    ----------
    vertices : (N, 2) array
    origin : (float, float), optional

    Returns
    -------
    A : float
        Signed area.
    S : (2,) array
        First moments (integrals of x and y).
    J : (2, 2) array
        Second moments (integrals of x^2, xy, y^2).
    """

    V = as_polygon(vertices) - np.asarray(origin, float)
    x0, y0 = V.T
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    w = x0*y1 - x1*y0
    A = 0.5*np.sum(w)
    S = np.array([np.sum((x0 + x1)*w), np.sum((y0 + y1)*w)])/6
    Jxx = np.sum((x0**2 + x0*x1 + x1**2)*w)/12
    Jyy = np.sum((y0**2 + y0*y1 + y1**2)*w)/12
    Jxy = np.sum((x0*y1 + 2*x0*y0 + 2*x1*y1 + x1*y0)*w)/24

    return float(A), S, np.array([[Jxx, Jxy], [Jxy, Jyy]])


def polygon_centroid(vertices):
    """Centroid (x, y) of the polygon area.

    Parameters
    ----------
    vertices : (N, 2) array

    Returns
    -------
    c : (2,) array
    """

    V = as_polygon(vertices)
    o = V.mean(axis=0)
    A, S, _ = polygon_moments(V, o)

    return o + S/A


def polygon_ft(vertices, kx, ky):
    """Exact Fourier transform of a polygon, the integral of exp(-i (kx x +
    ky y)) over its area.

    The orientation of the vertices does not matter.

    Parameters
    ----------
    vertices : (N, 2) array
        Vertices (x, y).
    kx, ky : arrays
        Angular spatial frequencies (rad per length unit), broadcast
        against each other.

    Returns
    -------
    F : complex array
        The transform, of the broadcast shape of kx and ky; at k = 0 the
        area.
    """

    V = as_polygon(vertices)
    o = V.mean(axis=0)
    W = V - o
    A, S, J = polygon_moments(W)
    if A < 0:  # clockwise: reverse, so that the normals point outward
        W = W[::-1]
        A, S, J = -A, -S, -J
    kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
    k2 = kx**2 + ky**2
    size = np.max(np.hypot(W[:, 0], W[:, 1]))
    small = k2*size**2 < 1e-8
    k2s = np.where(small, 1.0, k2)
    E = np.roll(W, -1, axis=0) - W
    M = W + 0.5*E
    F = np.zeros(kx.shape, complex)
    for (ex, ey), (mx, my) in zip(E, M):
        F += (kx*ey - ky*ex)*np.exp(-1j*(kx*mx + ky*my))*np.sinc((kx*ex + ky*ey)/(2*np.pi))
    F = 1j*F/k2s
    if np.any(small):  # A - i k.S - k.J.k/2
        T = A - 1j*(kx*S[0] + ky*S[1]) - 0.5*(kx**2*J[0, 0] + 2*kx*ky*J[0, 1] + ky**2*J[1, 1])
        F = np.where(small, T, F)

    return F*np.exp(-1j*(kx*o[0] + ky*o[1]))


def polygons_ft(polygons, kx, ky):
    """Fourier transform of the union of disjoint polygons (the sum of
    `polygon_ft()`).

    Parameters
    ----------
    polygons : sequence of (N, 2) arrays
    kx, ky : arrays
        Angular spatial frequencies, broadcast against each other.

    Returns
    -------
    F : complex array
    """

    kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
    F = np.zeros(kx.shape, complex)
    for P in polygons:
        F += polygon_ft(P, kx, ky)

    return F


def contains(vertices, x, y):
    """Points inside a polygon (even-odd rule).

    The test is half open: of two polygons that share an edge, a point on
    the edge is inside exactly one (the edge is evaluated in the same way
    for both orientations).

    Parameters
    ----------
    vertices : (N, 2) array
    x, y : arrays
        Points, broadcast against each other.

    Returns
    -------
    inside : bool array
    """

    V = as_polygon(vertices)
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    inside = np.zeros(x.shape, dtype=bool)
    for a, b in zip(V, np.roll(V, -1, axis=0)):
        if a[1] == b[1]:
            continue
        if a[1] > b[1]:  # the same arithmetic for both directions of the edge
            a, b = b, a
        crosses = (y >= a[1]) & (y < b[1])
        x_int = a[0] + (y - a[1])*((b[0] - a[0])/(b[1] - a[1]))
        inside ^= crosses & (x < x_int)

    return inside


def clip_to_rectangle(vertices, x0, y0, x1, y1):
    """Clip a polygon to the window [x0, x1] x [y0, y1] (Sutherland-Hodgman).

    For a concave polygon that leaves and re-enters the window, the parts
    are joined by edges on the window border that run there and back; such
    a polygon is weakly simple, which `polygon_ft()` and `contains()`
    handle.

    Parameters
    ----------
    vertices : (N, 2) array
    x0, y0, x1, y1 : numbers
        The window.

    Returns
    -------
    V : (M, 2) array or None
        The clipped polygon, None if nothing (no area) is left.
    """

    P = [tuple(v) for v in as_polygon(vertices)]
    # (inside test, intersection with the boundary) for the four sides
    sides = [(lambda p: p[0] >= x0, lambda p, q: (x0, p[1] + (q[1] - p[1])*(x0 - p[0])/(q[0] - p[0]))),
             (lambda p: p[0] <= x1, lambda p, q: (x1, p[1] + (q[1] - p[1])*(x1 - p[0])/(q[0] - p[0]))),
             (lambda p: p[1] >= y0, lambda p, q: (p[0] + (q[0] - p[0])*(y0 - p[1])/(q[1] - p[1]), y0)),
             (lambda p: p[1] <= y1, lambda p, q: (p[0] + (q[0] - p[0])*(y1 - p[1])/(q[1] - p[1]), y1))]
    for inside, cut in sides:
        if not P:
            break
        out = []
        for i, q in enumerate(P):
            p = P[i - 1]
            if inside(q):
                if not inside(p):
                    out.append(cut(p, q))
                out.append(q)
            elif inside(p):
                out.append(cut(p, q))
        P = out
    if len(P) < 3:
        return None
    V = np.array(P, dtype=float)
    V = V[np.any(V != np.roll(V, 1, axis=0), axis=1)]  # repeated vertices
    if len(V) < 3 or abs(polygon_area(V)) == 0:
        return None

    return V


def regular_polygon(n, radius, center=(0.0, 0.0), angle=0.0):
    """Regular polygon (counterclockwise).

    Parameters
    ----------
    n : int
        Number of vertices.
    radius : number
        Circumradius.
    center : (float, float), optional
    angle : number, optional
        Direction of the first vertex in radians.

    Returns
    -------
    V : (n, 2) array
    """

    a = angle + 2*np.pi*np.arange(n)/n

    return np.column_stack([center[0] + radius*np.cos(a), center[1] + radius*np.sin(a)])


def circle_polygon(radius, center=(0.0, 0.0), n=64, equal_area=True):
    """Circle as a regular polygon.

    Parameters
    ----------
    radius : number
    center : (float, float), optional
    n : int, optional
        Number of vertices.
    equal_area : bool, optional
        Scale the circumradius so that the polygon has the area of the
        circle (pi r^2) instead of being inscribed.

    Returns
    -------
    V : (n, 2) array
    """

    R = radius*np.sqrt(2*np.pi/(n*np.sin(2*np.pi/n))) if equal_area else radius

    return regular_polygon(n, R, center)


def path_polygon(points, width, extension=0.0):
    """Outline of a wire of constant width along a polyline.

    The edges run at width/2 on both sides of the polyline, joined at the
    bends by miters (the intersections of the offset edges), as a GDSII
    path. A bend should not be sharper than about 150 degrees (the miter
    grows as 1/cos(angle/2)).

    Parameters
    ----------
    points : (N, 2) array
        The center line, N >= 2.
    width : number
        Width of the wire (perpendicular to each segment).
    extension : number, optional
        Extension of the ends beyond the first and the last point (0: flush
        ends, width/2: square ends).

    Returns
    -------
    V : (2N, 2) array
        Counterclockwise outline.
    """

    P = np.array(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2 or len(P) < 2:
        raise ValueError("a path needs at least two (x, y) points")
    d = np.diff(P, axis=0)
    t = d/np.linalg.norm(d, axis=1)[:, None]
    P[0] -= extension*t[0]
    P[-1] += extension*t[-1]
    n = np.column_stack([-t[:, 1], t[:, 0]])  # left normals of the segments
    # normal at the points: of the segment at the ends, miter in between
    m = np.empty_like(P)
    m[0], m[-1] = n[0], n[-1]
    s = n[:-1] + n[1:]
    m[1:-1] = s/np.sum(s*n[1:], axis=1)[:, None]
    left, right = P + width/2*m, P - width/2*m
    V = np.concatenate([right, left[::-1]])

    return V if polygon_area(V) > 0 else V[::-1]
