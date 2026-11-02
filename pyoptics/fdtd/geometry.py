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
        x, y = np.broadcast_arrays(x, y)
        inside = np.zeros(x.shape, dtype=bool)
        xv, yv = self.vertices[:, 0], self.vertices[:, 1]
        for (x0, y0), (x1, y1) in zip(zip(xv, yv), zip(np.roll(xv, -1), np.roll(yv, -1))):
            if y0 == y1:
                continue
            crosses = (y0 > y) != (y1 > y)
            x_cross = x0 + (y - y0)*(x1 - x0)/(y1 - y0)
            inside ^= crosses & (x < x_cross)
        return inside
