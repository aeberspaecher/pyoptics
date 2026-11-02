#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Shapes that make up a structure.

Every shape has a method contains(x[, y]) that takes broadcastable
coordinate arrays and returns a boolean array. Shapes are combined with a
medium in a Simulation's structure list, where later entries are drawn on
top of earlier ones. Shapes can be added (union) and subtracted
(difference): ``Circle((0, 0), 2) - Circle((0, 0), 1.5)`` is a ring.
"""

import numpy as np


class Shape(object):
    """Base class: a region of space, defined by contains()."""

    def contains(self, *coords):
        raise NotImplementedError

    def __add__(self, other):
        return Union(self, other)

    def __sub__(self, other):
        return Difference(self, other)


class Union(Shape):
    """Points inside any of the shapes."""

    def __init__(self, *shapes):
        self.shapes = shapes

    def contains(self, *coords):
        inside = self.shapes[0].contains(*coords)
        for s in self.shapes[1:]:
            inside = inside | s.contains(*coords)
        return inside


class Difference(Shape):
    """Points inside a but not inside b."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def contains(self, *coords):
        return self.a.contains(*coords) & ~self.b.contains(*coords)


class Everywhere(Shape):
    """The whole domain."""

    def contains(self, *coords):
        return np.ones(np.broadcast(*coords).shape, dtype=bool)


class Slab(Shape):
    """Layer lo <= coordinate < hi along an axis (0: x, 1: y), unbounded in
    the other directions. Works in 1D and 2D."""

    def __init__(self, lo, hi, axis=0):
        self.lo, self.hi, self.axis = lo, hi, axis

    def contains(self, *coords):
        c = coords[self.axis]
        inside = (c >= self.lo) & (c < self.hi)
        return np.broadcast_to(inside, np.broadcast(*coords).shape)


class Rect(Shape):
    """Axis-parallel rectangle (2D) given by its centre and size."""

    def __init__(self, center, size):
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)

    def contains(self, x, y):
        return (np.abs(x - self.center[0]) <= self.size[0]/2) & \
               (np.abs(y - self.center[1]) <= self.size[1]/2)


class Circle(Shape):
    """Disc (2D), the cross section of a cylinder along z."""

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = radius

    def contains(self, x, y):
        return (x - self.center[0])**2 + (y - self.center[1])**2 < self.radius**2


class Annulus(Shape):
    """Ring between two radii (2D), e.g. a ring resonator."""

    def __init__(self, center, r_inner, r_outer):
        self.center = np.asarray(center, dtype=float)
        self.r_inner, self.r_outer = r_inner, r_outer

    def contains(self, x, y):
        r2 = (x - self.center[0])**2 + (y - self.center[1])**2
        return (r2 >= self.r_inner**2) & (r2 < self.r_outer**2)


class Polygon(Shape):
    """Simple polygon (2D, convex or concave) given by its vertices."""

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=float)

    def contains(self, x, y):
        # crossing number: a ray towards -x from each point crosses the
        # edges whose y range contains the point's y; with the points sorted
        # by y, each edge only touches the points in its y range
        x, y = np.broadcast_arrays(x, y)
        shape = x.shape
        xf, yf = x.ravel(), y.ravel()
        inside = np.zeros(xf.shape, dtype=bool)
        order = np.argsort(yf, kind="stable")
        ys = yf[order]
        xv, yv = self.vertices[:, 0], self.vertices[:, 1]
        for x0, y0, x1, y1 in zip(xv, yv, np.roll(xv, -1), np.roll(yv, -1)):
            if y0 == y1:
                continue
            a, b = np.searchsorted(ys, [min(y0, y1), max(y0, y1)], side="left")
            if a == b:
                continue
            pts = order[a:b]
            x_cross = x0 + (yf[pts] - y0)*(x1 - x0)/(y1 - y0)
            inside[pts] ^= xf[pts] < x_cross
        return inside.reshape(shape)


class Path(Polygon):
    """Waveguide of constant width along a polyline (2D).

    Parameters
    ----------
    points : array (n, 2)
        Centre line; sample curves finely enough (a few points per cell
        along strongly curved parts).
    width : float

    The outline is the centre line offset by +-width/2 along the normals
    (averaged at the vertices).
    """

    def __init__(self, points, width):
        p = np.asarray(points, dtype=float)
        if len(p) < 2:
            raise ValueError("a path needs at least two points")
        d = np.gradient(p, axis=0)
        t = d/np.linalg.norm(d, axis=1)[:, None]
        nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
        self.points = p
        self.width = width
        super().__init__(np.concatenate([p + nrm*width/2, (p - nrm*width/2)[::-1]]))


def sbend_points(start, length, offset, n=200, axis=0):
    """Centre line of a raised-cosine S-bend from `start`, `length` along
    `axis` and `offset` across it: y = offset*(1 - cos(pi s/length))/2."""

    s = np.linspace(0, length, n)
    q = offset*(1 - np.cos(np.pi*s/length))/2
    pts = np.stack([s, q], axis=1) if axis == 0 else np.stack([q, s], axis=1)
    return pts + np.asarray(start, dtype=float)


def SBend(start, length, offset, width, n=200, axis=0):
    """Raised-cosine S-bend waveguide (a Path), see sbend_points()."""

    return Path(sbend_points(start, length, offset, n, axis), width)


class Bend(Shape):
    """Circular waveguide bend (2D): the part of an annulus between two
    angles (radians, counter-clockwise from +x).

    Parameters
    ----------
    center : (x, y)
    radius : float
        Radius of the centre line.
    width : float
    angles : (start, stop)
        e.g. (0, pi/2) for a quarter circle in the first quadrant.
    """

    def __init__(self, center, radius, width, angles):
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.width = width
        self.angles = angles

    def contains(self, x, y):
        dx, dy = x - self.center[0], y - self.center[1]
        r2 = dx**2 + dy**2
        a0, a1 = self.angles
        phi = np.mod(np.arctan2(dy, dx) - a0, 2*np.pi)
        return (r2 >= (self.radius - self.width/2)**2) & (r2 < (self.radius + self.width/2)**2) & \
               (phi <= np.mod(a1 - a0, 2*np.pi) + 1e-12)


def Ring(center, radius, width):
    """Ring waveguide (an Annulus) with centre-line radius `radius`."""

    return Annulus(center, radius - width/2, radius + width/2)
