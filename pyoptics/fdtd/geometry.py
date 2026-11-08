#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Shapes that make up a structure.

Every shape has a method contains(x[, y[, z]]) that takes broadcastable
coordinate arrays and returns a boolean array. 2D shapes (Rect, Circle,
Polygon, the waveguides, Lattice) become 3D ones by extrusion with Prism;
Sphere, Box and Cylinder are 3D shapes; Slab and Everywhere work in any
dimension. Shapes are combined with a
medium in a Simulation's structure list, where later entries are drawn on
top of earlier ones. Shapes can be added (union) and subtracted
(difference): ``Circle((0, 0), 2) - Circle((0, 0), 1.5)`` is a ring.

References
----------
A. Farjadpour, D. Roundy, A. Rodriguez, M. Ibanescu, P. Bermel, J. D.
Joannopoulos, S. G. Johnson and G. W. Burr, "Improving accuracy by subpixel
smoothing in the finite-difference time domain", Opt. Lett. 31, 2972-2974
(2006).

C. Kottke, A. Farjadpour and S. G. Johnson, "Perturbation theory for
anisotropic dielectric interfaces, and application to subpixel smoothing of
discretized numerical methods", Phys. Rev. E 77, 036611 (2008).

A. F. Oskooi, C. Kottke and S. G. Johnson, "Accurate finite-difference
time-domain simulation of anisotropic media by subpixel smoothing", Opt.
Lett. 34, 2778-2780 (2009).
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
    """Layer lo <= coordinate < hi along an axis (0: x, 1: y, 2: z),
    unbounded in the other directions. Works in 1D, 2D and 3D."""

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


class PolarShape(Shape):
    """Star-shaped region (2D) bounded by a curve r(phi) in polar
    coordinates about a centre: the points with distance < r(phi) at their
    polar angle phi.

    Parameters
    ----------
    center : (x, y)
    radius : callable
        r(phi), positive for all phi (vectorized).
    """

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = radius

    def contains(self, x, y):
        dx, dy = x - self.center[0], y - self.center[1]
        return np.hypot(dx, dy) < self.radius(np.arctan2(dy, dx))

    def boundary(self, n=721):
        """Points (n, 2) on the boundary curve."""
        phi = np.linspace(0, 2*np.pi, n)
        r = self.radius(phi)
        return np.column_stack([self.center[0] + r*np.cos(phi), self.center[1] + r*np.sin(phi)])


class Limacon(PolarShape):
    """Limacon of Pascal (2D): r(phi) = R (1 + eps cos(phi)) about the
    centre, the boundary of a deformed microdisk (Wiersig and Hentschel,
    Phys. Rev. Lett. 100, 033901 (2008)). eps = 0 is the circle; the curve is
    convex up to eps = 1/2 (beyond, it has a dent at phi = pi) and
    star-shaped up to eps < 1. The largest radius R (1 + eps) is at phi = 0
    (+x); the side at phi = pi is flattened (curvature (1 - 2 eps)/(1 - eps)**2/R).

    Parameters
    ----------
    center : (x, y)
    radius : number
        R.
    eps : number
        Deformation, 0 <= eps < 1.
    """

    def __init__(self, center, radius, eps):
        if not 0 <= eps < 1:
            raise ValueError("the limacon needs 0 <= eps < 1")
        self.R, self.eps = float(radius), float(eps)
        super(Limacon, self).__init__(center, lambda phi: self.R*(1 + self.eps*np.cos(phi)))

    def area(self):
        """Enclosed area pi R**2 (1 + eps**2/2)."""
        return np.pi*self.R**2*(1 + self.eps**2/2)


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


class Lattice(Shape):
    """Copies of a shape on a lattice (2D): photonic crystals, gratings of
    pillars, arrays of holes.

    The sites are center + i a1 + j a2 for n1 x n2 indices i, j centred on
    0 (i = -(n1 - 1)/2 ... (n1 - 1)/2, half-integers for even n1), minus
    the sites listed in `remove`; `shift` moves single sites (e.g. the
    rods next to a cavity). The shape is given relative to its site and
    must fit into the cell around it (every point is tested against its
    nearest site only), which makes the test independent of the number of
    sites.

    Parameters
    ----------
    shape : Shape
        Centred on (0, 0), e.g. Circle((0, 0), 0.2).
    a1, a2 : (x, y)
        Lattice vectors, e.g. (1, 0) and (0, 1) for a square lattice, (1, 0)
        and (0.5, sqrt(3)/2) for a triangular one.
    n : (n1, n2)
        Number of sites along a1 and a2; None for an unbounded axis.
    center : (x, y)
    remove : iterable of (i, j)
        Sites left empty: (0, 0) for a point defect, [(i, 0) for all i] for
        a line defect.
    shift : dict {(i, j): (dx, dy)}
        Displaced sites.

    Examples
    --------
    A square lattice of rods with a line defect along x:

    >>> rods = Lattice(Circle((0, 0), 0.2), (1, 0), (0, 1), (21, 11),
    ...                remove=[(i, 0) for i in range(-10, 11)])
    """

    def __init__(self, shape, a1, a2, n, center=(0, 0), remove=(), shift=None):
        self.shape = shape
        self.A = np.array([a1, a2], dtype=float).T          # columns: lattice vectors
        self.Ainv = np.linalg.inv(self.A)
        self.n = tuple(n)
        self.center = np.asarray(center, dtype=float)
        self.remove = {tuple(float(v) for v in s) for s in remove}
        self.shift = {tuple(float(v) for v in k): np.asarray(v, dtype=float)
                      for k, v in (shift or {}).items()}

    def sites(self):
        """Positions of all sites (after removal and shifts), (N, 2)."""

        idx = []
        rng = [np.arange(n) - (n - 1)/2 for n in self.n]
        for i in rng[0]:
            for j in rng[1]:
                if (float(i), float(j)) not in self.remove:
                    idx.append((i, j))
        pts = np.array([self.center + self.A @ np.array(ij) for ij in idx])
        for k, (i, j) in enumerate(idx):
            if (float(i), float(j)) in self.shift:
                pts[k] += self.shift[(float(i), float(j))]
        return pts

    def contains(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        dx, dy = x - self.center[0], y - self.center[1]
        u = self.Ainv[0, 0]*dx + self.Ainv[0, 1]*dy
        v = self.Ainv[1, 0]*dx + self.Ainv[1, 1]*dy
        inside = np.zeros(x.shape, dtype=bool)
        # the site of a point: the nearest one (rounding in the shifted index
        # space, half-integer sites for even counts), and for a shifted site
        # also its neighbours
        offs = [(n - 1)/2 % 1 if n is not None else 0.0 for n in self.n]
        i0 = np.round(u - offs[0]) + offs[0]
        j0 = np.round(v - offs[1]) + offs[1]
        cand = [(0, 0)] + ([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
                           if self.shift else [])
        for di, dj in cand:
            i, j = i0 + di, j0 + dj
            ok = np.ones(x.shape, dtype=bool)
            for ind, n in ((i, self.n[0]), (j, self.n[1])):
                if n is not None:
                    ok &= np.abs(ind) <= (n - 1)/2 + 1e-9
            sx = self.center[0] + self.A[0, 0]*i + self.A[0, 1]*j
            sy = self.center[1] + self.A[1, 0]*i + self.A[1, 1]*j
            for (ri, rj) in self.remove:
                ok &= ~((np.abs(i - ri) < 1e-9) & (np.abs(j - rj) < 1e-9))
            for (si, sj), d in self.shift.items():
                m = (np.abs(i - si) < 1e-9) & (np.abs(j - sj) < 1e-9)
                sx = np.where(m, sx + d[0], sx)
                sy = np.where(m, sy + d[1], sy)
            inside |= ok & self.shape.contains(x - sx, y - sy)
        return inside


class Sphere(Shape):
    """Ball (3D) given by its centre and radius."""

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = radius

    def contains(self, x, y, z):
        c = self.center
        return (x - c[0])**2 + (y - c[1])**2 + (z - c[2])**2 < self.radius**2


class Box(Shape):
    """Axis-parallel cuboid (3D) given by its centre and size."""

    def __init__(self, center, size):
        self.center = np.asarray(center, dtype=float)
        self.size = np.asarray(size, dtype=float)

    def contains(self, x, y, z):
        c, h = self.center, self.size/2
        return (np.abs(x - c[0]) <= h[0]) & (np.abs(y - c[1]) <= h[1]) & (np.abs(z - c[2]) <= h[2])


class Prism(Shape):
    """A 2D shape extruded along an axis (3D): the points with lo <= the
    coordinate along `axis` < hi whose other two coordinates (in the order
    x, y, z with `axis` left out) lie in the 2D shape. Prism(Circle(...),
    ...) is a finite cylinder, Prism(Path(...), ...) a strip waveguide.

    Parameters
    ----------
    shape : Shape
        2D shape.
    lo, hi : number
        Extent along the axis.
    axis : int
        0, 1 or 2 (default: z).
    """

    def __init__(self, shape, lo, hi, axis=2):
        self.shape, self.lo, self.hi, self.axis = shape, lo, hi, axis

    def contains(self, x, y, z):
        coords = (x, y, z)
        c = coords[self.axis]
        others = [v for a, v in enumerate(coords) if a != self.axis]
        return (c >= self.lo) & (c < self.hi) & self.shape.contains(*others)


def Cylinder(center, radius, length, axis=2):
    """Circular cylinder (3D) with its axis along x, y or z: a Prism of a
    Circle. center is the 3D centre of the cylinder."""

    center = np.asarray(center, dtype=float)
    c2 = [v for a, v in enumerate(center) if a != axis]
    return Prism(Circle(c2, radius), center[axis] - length/2, center[axis] + length/2, axis)
