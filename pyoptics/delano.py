#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Delano's y-ybar diagram: first order layout of optical systems.

Delano (1963) plotted the height ybar of the paraxial chief ray against the
height y of the paraxial marginal ray. A system of thin lenses (or of
surfaces) becomes a polygon: each corner is a lens (a surface), each edge
the space between two of them. The diagram holds all first order data of
the system: from it the powers, the spacings, the pupils, the cardinal
points and the conjugates can be read off, and a system can be built from a
polygon drawn by hand. This is how lens designers lay out complex systems
such as lithographic projection lenses before any surface is bent.

Conventions
-----------
Points are P = (ybar, y), plotted with ybar on the abscissa and y on the
ordinate (Delano's original orientation). The reduced slopes are
omega = n u (marginal ray) and omegabar = n ubar (chief ray), with the
direction slopes u = dy/dz as in `pyoptics.raytrace`. The Lagrange invariant
is ::

    H = omegabar*y - omega*ybar = n (ubar y - u ybar),

the same as `raytrace.Paraxial.lagrange`. For two consecutive points the
reduced distance tau = t/n of the space between them follows from the area
of the triangle they span with the origin, ::

    tau_k = (y_k ybar_(k+1) - ybar_k y_(k+1))/H,

and the power of a corner from the change of direction of the polygon, ::

    phi_k = -(omegabar_(k-1) omega_k - omega_(k-1) omegabar_k)/H,

with the reduced slopes omega_k = (y_(k+1) - y_k)/tau_k of the edges. So
far from the origin, a corner with a given bend has a small power, while
near the origin a small bend already is a strong lens. Further properties
of the diagram:

- The object point (if finite) and the image point lie on the ybar axis
  (y = 0), the aperture stop and the pupils on the y axis (ybar = 0). An
  edge that is parallel to the y axis is a telecentric space (chief ray
  parallel to the axis).
- A line through the origin cuts the polygon in conjugate points (planes
  that are images of each other); the ratio of the distances from the
  origin is their magnification.
- The object and image space lines (first and last edges, extended)
  intersect in the principal points; the line through the origin parallel
  to the image space line meets the object space line in the front focal
  point and vice versa.
- Scaling all marginal ray heights y by a factor scales all lengths by the
  same factor at the same Lagrange invariant, i.e. the same NA and field
  (`DelanoDiagram.with_length()`). Scaling ybar does the same at a
  different field.
- The Petzval sum of thin lenses in air is sum(phi_k/n_k), independent of
  the heights: positive lenses far from the origin (large beams) and
  negative ones close to it (waists) flatten the field.

`DelanoDiagram.from_system()` makes the diagram of a `raytrace.System` (one
point per surface vertex), `DelanoDiagram.to_system()` the thin lens system
(`raytrace.IdealLens`) of a diagram and `thicken()` a system of real thick
lenses with the same first order properties. `refine()` polishes a hand drawn
diagram (Petzval sum, room for the lenses, signs of the powers) and
`DelanoDiagram.plot()` draws it.

The polygon has to wind around the origin in one sense (counterclockwise for
H < 0, i.e. object above the axis and marginal ray rising): a step back in
angle is a negative distance. A positive lens is a convex corner (the polygon
turns towards the origin), a negative lens a concave one.

Reference: E. Delano, "First-order design and the y, ybar diagram", Appl.
Opt. 2, 1251-1256 (1963); R. V. Shack, "Analytic system design with pencil
and ruler", Proc. SPIE 39, 127 (1973).
"""

from collections import namedtuple

import numpy as np

from . import raytrace


CardinalPoints = namedtuple("CardinalPoints", ["efl", "front_focal", "rear_focal", "front_principal",
                                               "rear_principal"])
CardinalPoints.__doc__ = """Cardinal points from the diagram: effective focal
length (in the image space medium, efl = rear_focal - rear_principal) and z
positions of the focal and principal points (physical lengths along the
unfolded axis, z = 0 at the first point of the diagram); infinite for an
afocal system."""

Pupils = namedtuple("Pupils", ["stop", "entrance", "exit"])
Pupils.__doc__ = """z positions of the aperture stop (the crossings of the
polygon with the y axis, an array) and of the entrance and exit pupils (the
first and last edges extended to ybar = 0; infinite if telecentric)."""


def _cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def _intersect(p, dp, q, dq):
    """Parameters (a, b) of the intersection p + a dp = q + b dq of two lines
    in the diagram (NaN if parallel)."""

    det = _cross(dq, dp)
    if abs(det) < 1e-15*np.hypot(*dp)*np.hypot(*dq):
        return np.nan, np.nan
    r = q - p

    return _cross(dq, r)/det, _cross(dp, r)/det


class DelanoDiagram:
    """Polygon of paraxial marginal and chief ray heights (y-ybar diagram).

    Parameters
    ----------
    ybar, y : sequences of length K
        Chief and marginal ray heights at the points (object, lenses or
        surfaces, image) in the order the light meets them.
    H : number
        Lagrange invariant n (ubar y - u ybar).
    n : number or sequence of length K - 1, optional
        Refractive index of each space between consecutive points (signed:
        negative where the light travels in -z behind a mirror). Default 1.
    labels : sequence of str, optional
        Names of the points for plot().

    Notes
    -----
    The reduced distances, slopes and powers follow from the points and H
    alone (see the module docstring); n only converts reduced to physical
    distances.
    """

    def __init__(self, ybar, y, H, n=1.0, labels=None):
        self.ybar = np.asarray(ybar, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.ybar.shape != self.y.shape or self.y.ndim != 1 or len(self.y) < 2:
            raise ValueError("ybar and y need the same length (at least 2).")
        self.H = float(H)
        if self.H == 0:
            raise ValueError("The Lagrange invariant must not vanish.")
        self.n = np.broadcast_to(np.asarray(n, dtype=float), (len(self.y) - 1,)).copy()
        self.labels = list(labels) if labels is not None else [str(k) for k in range(len(self.y))]

    def __repr__(self):
        return "DelanoDiagram({} points, H={:.6g}, length={:.6g})".format(len(self.y), self.H,
                                                                         self.length)

    def __len__(self):
        return len(self.y)

    @property
    def points(self):
        """Points (ybar, y), shape (K, 2)."""

        return np.column_stack([self.ybar, self.y])

    @property
    def tau(self):
        """Reduced distances t/n of the K - 1 spaces."""

        return (self.y[:-1]*self.ybar[1:] - self.ybar[:-1]*self.y[1:])/self.H

    @property
    def thickness(self):
        """Physical (signed) distances n*tau of the spaces."""

        return self.n*self.tau

    @property
    def z(self):
        """z positions of the points (unfolded, z = 0 at the first point)."""

        return np.concatenate([[0.0], np.cumsum(np.abs(self.thickness))])

    @property
    def length(self):
        """Total length from the first to the last point."""

        return self.z[-1]

    @property
    def slopes(self):
        """Reduced slopes (omegabar, omega) = n (ubar, u) of the spaces,
        shape (K - 1, 2)."""

        return np.diff(self.points, axis=0)/self.tau[:, None]

    @property
    def power(self):
        """Powers of the K - 2 inner points (thin lenses or surfaces)."""

        w = self.slopes

        return -_cross(w[:-1], w[1:])/self.H

    @property
    def focal_length(self):
        """Focal lengths n/phi of the inner points in the medium in front of
        them (as `raytrace.IdealLens.f`)."""

        with np.errstate(divide="ignore"):
            return np.abs(self.n[:-1])/self.power

    def petzval_sum(self, n_lens):
        """Petzval sum sum(phi_k/n_k) of thin lenses in air.

        Parameters
        ----------
        n_lens : number or sequence
            Refractive index of the glass of each inner point.

        Returns
        -------
        P : number
            Petzval sum (1/radius of the Petzval surface is -n' P).
        """

        return np.sum(self.power/np.asarray(n_lens, dtype=float))

    @property
    def magnification(self):
        """Lateral magnification between the first and the last point,
        ybar[-1]/ybar[0] (meaningful if both lie on the ybar axis)."""

        return self.ybar[-1]/self.ybar[0]

    @property
    def NA(self):
        """Numerical apertures |omega| of the first and the last space."""

        w = self.slopes[:, 1]

        return abs(w[0]), abs(w[-1])

    def heights(self):
        """Paraxial beam radius |y| + |ybar| at every point (the clear
        semi diameter that the full field needs)."""

        return np.abs(self.y) + np.abs(self.ybar)

    def with_length(self, length):
        """Diagram with all y scaled such that the total length becomes
        `length` (same H, NA, field and magnification).

        Parameters
        ----------
        length : number
            New total length (first to last point).

        Returns
        -------
        DelanoDiagram
        """

        return DelanoDiagram(self.ybar, self.y*length/self.length, self.H, self.n, self.labels)

    def plot(self, ax=None, annotate=True, color="C0", label=None, areas=False, stop=True,
             signs=True, **kwargs):
        """Draw the diagram: ybar on the abscissa, y on the ordinate.

        Parameters
        ----------
        ax : matplotlib axis, optional
            Defaults to the current axis.
        annotate : bool, optional
            Write the labels of the points next to them.
        color : color, optional
        label : str, optional
            Legend label of the polygon.
        areas : bool, optional
            Shade the triangles spanned by the spaces with the origin (their
            areas are H*tau/2).
        stop : bool, optional
            Mark the crossings with the y axis (aperture stop).
        signs : bool, optional
            Mark positive inner points with filled, negative ones with open
            circles.
        **kwargs
            Passed to the plot of the polygon.

        Returns
        -------
        ax : matplotlib axis
        """

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        yb, y = self.ybar, self.y
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvline(0, color="0.6", lw=0.6)
        if areas:
            for k in range(len(y) - 1):
                ax.fill([0, yb[k], yb[k + 1]], [0, y[k], y[k + 1]], color=color, alpha=0.08, lw=0)
        ax.plot(yb, y, "-", color=color, label=label, **kwargs)
        inner = slice(1, len(y) - 1)
        if signs:
            pos = self.power > 0
            ax.plot(yb[inner][pos], y[inner][pos], "o", color=color, ms=5)
            ax.plot(yb[inner][~pos], y[inner][~pos], "o", mfc="white", mec=color, ms=5)
        else:
            ax.plot(yb[inner], y[inner], "o", color=color, ms=4)
        ax.plot(yb[[0, -1]], y[[0, -1]], "s", color=color, ms=6)
        if stop:
            for k in range(len(y) - 1):
                if yb[k]*yb[k + 1] < 0:
                    s = yb[k]/(yb[k] - yb[k + 1])
                    ax.plot(0, y[k] + s*(y[k + 1] - y[k]), "*", color=color, ms=11)
        if annotate:
            for k in range(len(y)):
                ax.annotate(self.labels[k], (yb[k], y[k]), textcoords="offset points",
                            xytext=(4, 4), fontsize=8, color=color)
        ax.set_xlabel(r"chief ray height $\bar y$")
        ax.set_ylabel("marginal ray height $y$")

        return ax

    # first order properties:
    def pupils(self):
        """Stop and pupil positions, see `Pupils`.

        Returns
        -------
        Pupils
        """

        z, t, yb = self.z, self.thickness, self.ybar
        stops = []
        for k in range(len(yb) - 1):
            if yb[k] == 0 or (yb[k]*yb[k + 1] < 0):
                stops.append(z[k] + abs(t[k])*yb[k]/(yb[k] - yb[k + 1]))
        w = self.slopes
        ep = (z[0] - self.n[0]*yb[0]/w[0, 0]) if w[0, 0] != 0 else np.inf
        xp = (z[-1] - self.n[-1]*yb[-1]/w[-1, 0]) if w[-1, 0] != 0 else np.inf

        return Pupils(np.array(stops), ep, xp)

    def cardinal_points(self):
        """Focal and principal points from the object and image space lines.

        Returns
        -------
        CardinalPoints
        """

        P, w, z, n = self.points, self.slopes, self.z, self.n
        p0, d0, p1, d1 = P[0], w[0], P[-1], w[-1]
        a, b = _intersect(p0, d0, p1, d1)
        if not np.isfinite(a):
            return CardinalPoints(np.inf, np.inf, np.inf, np.inf, np.inf)
        z_h, z_h1 = z[0] + n[0]*a, z[-1] + n[-1]*b
        # focal points: lines through the origin parallel to the other space
        a_f, _ = _intersect(p0, d0, np.zeros(2), d1)
        b_f, _ = _intersect(p1, d1, np.zeros(2), d0)
        z_f, z_f1 = z[0] + n[0]*a_f, z[-1] + n[-1]*b_f

        return CardinalPoints(z_f1 - z_h1, z_f, z_f1, z_h, z_h1)

    def conjugate(self, k, m):
        """Point on space k of the polygon that is conjugate to point m.

        Parameters
        ----------
        k : int
            Index of the space (edge from point k to point k + 1, extended).
        m : int
            Index of a point of the diagram.

        Returns
        -------
        z : number
            z position of the conjugate (on the extended edge k).
        magnification : number
            Lateral magnification from point m to its conjugate.
        """

        P, w = self.points, self.slopes
        a, c = _intersect(P[k], w[k], np.zeros(2), P[m])

        return self.z[k] + abs(self.n[k])*a, c

    # systems:
    @classmethod
    def from_system(cls, system, field=None):
        """Diagram of a `raytrace.System`: one point per vertex, plus the
        object point and the paraxial image point.

        For an object at infinity the first point is a point of object
        space instead (label "object space"), as far in front of the first
        vertex as the entrance pupil is from it (or the entrance pupil radius
        if that is zero), so that the first edge is the object space line.

        Parameters
        ----------
        system : raytrace.System
            Centred system (tilts and decentres are ignored, as in its
            paraxial data).
        field : number, optional
            Field of the chief ray (default: the maximum field of the system).

        Returns
        -------
        DelanoDiagram
        """

        p = system.paraxial(field)
        n_s = system.n*system.sigma
        y, yb = list(p.marginal[:, 0]), list(p.chief[:, 0])
        n = list(n_s[1:-1])
        labels = [s.label if s.label is not None else str(i + 1)
                  for i, s in enumerate(system.surfaces)]
        if not system.infinite_object:
            y = [0.0] + y
            yb = [raytrace._field_xy(system.max_field if field is None else field)[1]] + yb
            n = [n_s[0]] + n
            labels = ["object"] + labels
        else:
            (_, u), (_, ub) = system._paraxial_rays(field)
            D = abs(p.ep_z) if p.ep_z != 0 else p.ep_radius
            y = [y[0] - D*u] + y
            yb = [yb[0] - D*ub] + yb
            n = [n_s[0]] + n
            labels = ["object space"] + labels
        if np.isfinite(p.image_distance):
            y.append(0.0)
            yb.append(p.image_height)
            n.append(n_s[-1])
            labels.append("image")

        return cls(yb, y, p.lagrange, n, labels)

    def to_system(self, wavelength, units="mm", margin=None, stop=None):
        """Thin lens system of the diagram.

        The first point is the object, the last the image (both should lie
        on the ybar axis), every inner point an `raytrace.IdealLens`. A plane
        dummy surface is inserted as the aperture stop where the polygon
        crosses the y axis.

        Parameters
        ----------
        wavelength : number
            Wavelength in the length unit.
        units : str, optional
            Length unit of the system.
        margin : number, optional
            Semi diameters of the lenses are margin*(|y| + |ybar|); None
            (default): unlimited, only the stop limits the aperture.
        stop : int, optional
            Index of the y axis crossing to use as the stop if there are
            several (default: the first).

        Returns
        -------
        raytrace.System
        """

        surfaces, stop_index = self._surfaces(lambda k, t: [raytrace.IdealLens(
            self.focal_length[k - 1], t, label=self.labels[k])], stop)
        system = raytrace.System(surfaces, stop_index, wavelength,
                                 object_distance=self.thickness[0], object_height=self.ybar[0],
                                 n_object=abs(self.n[0]), units=units)
        _set_apertures(system, self.slopes[0, 1]/self.n[0], margin)

        return system

    def _surfaces(self, make_lens, stop=None, offsets=None, check=True):
        """Surfaces of the inner points with the stop inserted; make_lens(k, t)
        returns the surfaces of point k with t the distance from its last
        surface to the next point; offsets[k] = (front, back) shortens the
        spaces in front of and behind point k."""

        K = len(self)
        if K < 3:
            raise ValueError("A system needs at least one inner point.")
        if self.y[0] != 0 or self.y[-1] != 0:
            raise ValueError("The first and the last point must be object and image (y = 0).")
        t = self.thickness.copy()
        if offsets is not None:
            for k in range(1, K - 1):
                t[k - 1] -= offsets[k][0]
                t[k] -= offsets[k][1]
        yb = self.ybar
        crossings = [k for k in range(1, K - 2) if yb[k] == 0 or yb[k]*yb[k + 1] < 0]
        crossings += [K - 2] if yb[K - 2] == 0 else []
        if not crossings:
            raise ValueError("The polygon does not cross the y axis between the lenses: no stop.")
        k_stop = crossings[0 if stop is None else stop]
        surfaces, stop_index = [], None
        for k in range(1, K - 1):
            if k == k_stop and yb[k] == 0:
                group = make_lens(k, t[k])
                stop_index = len(surfaces)
                group[0].semi_diameter = 1.0  # set by _set_apertures()
                surfaces += group
                continue
            if k == k_stop:
                # stop in the space behind point k, at the crossing of the polygon:
                s = yb[k]/(yb[k] - yb[k + 1])
                d = s*self.thickness[k] - (offsets[k][1] if offsets is not None else 0.0)
                if check and (d < 0 or d > t[k]):
                    raise ValueError("The stop lies inside lens {}.".format(self.labels[k]))
                group = make_lens(k, d)
                surfaces += group
                stop_index = len(surfaces)
                surfaces.append(raytrace.Surface(np.inf, t[k] - d, semi_diameter=1.0, label="stop"))
                continue
            surfaces += make_lens(k, t[k])
        if check and np.any(t < 0):
            bad = [self.labels[k] for k in range(K - 1) if t[k] < 0]
            raise ValueError("Negative spaces behind points {} (lenses overlap).".format(bad))

        return surfaces, stop_index


def refine(diagram, n_lens, min_space=0.0, spaces=None, max_height=None, signs=None, anchor=1.0,
           deviation=0.1, petzval=1e3, length=None):
    """Polish a hand drawn diagram: zero Petzval sum, room for the lenses,
    powers of the intended signs, close to the sketch.

    The inner points move (both coordinates; points on a telecentric first
    or last edge keep their ybar, so the telecentricity stays). Since the
    diagram is rescaled to its length (`DelanoDiagram.with_length()`) after
    every step, the NA, field and magnification stay those of the sketch.
    The residuals (least squares) are

    - petzval*(Petzval sum)*length (thin lenses of index n_lens),
    - spaces shorter than min_space and the deviation from target `spaces`,
    - beam heights |y| + |ybar| above max_height,
    - powers whose sign differs from `signs` (default: those of the sketch),
    - deviation*(marginal ray deviation y*phi of each lens)/(mean |y*phi|),
      which spreads the bending of the beam over the lenses,
    - anchor*(distance of the points from the sketch) in units of the
      largest |ybar| and |y| of the sketch.

    Parameters
    ----------
    diagram : DelanoDiagram
        Sketch: object first, image last, thin lenses in between.
    n_lens : number or sequence
        Glass index of the lenses, for the Petzval sum.
    min_space : number or sequence, optional
        Smallest physical length of each space.
    spaces : dict, optional
        {space index: target length} (e.g. the working distance, -1).
    max_height : number, optional
        Largest beam height |y| + |ybar| (clear semi diameter).
    signs : sequence of +-1, optional
        Signs of the lens powers.
    anchor, deviation, petzval : numbers, optional
        Weights of the residuals.
    length : number, optional
        Total length (default: that of the sketch).

    Returns
    -------
    DelanoDiagram
    """

    from scipy.optimize import least_squares

    length = diagram.length if length is None else length
    sketch = diagram.with_length(length)
    K = len(sketch)
    signs = np.sign(sketch.power) if signs is None else np.asarray(signs, dtype=float)
    yb0, y0 = sketch.ybar.copy(), sketch.y.copy()
    s_yb, s_y = np.max(np.abs(yb0)), np.max(np.abs(y0))
    free_yb = [k for k in range(1, K - 1)
               if not (k == 1 and yb0[1] == yb0[0]) and not (k == K - 2 and yb0[-2] == yb0[-1])]
    free_y = list(range(1, K - 1))
    min_space = np.broadcast_to(np.asarray(min_space, dtype=float), (K - 1,))
    spaces = {} if spaces is None else spaces
    dev0 = np.mean(np.abs(sketch.y[1:-1]*sketch.power))

    def unpack(x):
        yb, y = yb0.copy(), y0.copy()
        yb[free_yb] = x[:len(free_yb)]*s_yb
        y[free_y] = x[len(free_yb):]*s_y
        return DelanoDiagram(yb, y, sketch.H, sketch.n, sketch.labels).with_length(length)

    def residuals(x):
        d = unpack(x)
        t = d.thickness
        out = [[petzval*d.petzval_sum(n_lens)*length],
               10*np.maximum(0, min_space - t)/np.maximum(min_space, 1e-3*length),
               [10*(t[k] - v)/v for k, v in spaces.items()],
               100*np.maximum(0, 0.05*dev0 - signs*d.y[1:-1]*d.power)/dev0,
               deviation*d.y[1:-1]*d.power/dev0,
               anchor*(d.ybar - yb0)[1:-1]/s_yb, anchor*(d.y/d.y.max() - y0/s_y)[1:-1]]
        if max_height is not None:
            out.append(10*np.maximum(0, d.heights() - max_height)/max_height)
        r = np.concatenate(out)
        return np.where(np.all(d.tau > 0), r, 1e3)

    x0 = np.concatenate([yb0[free_yb]/s_yb, y0[free_y]/s_y])
    r = least_squares(residuals, x0)

    return unpack(r.x)


def thick_lens_curvatures(phi, n, t, shape):
    """Curvatures of a thick lens in air of power phi, thickness t and
    Coddington shape factor X = (c1 + c2)/(c1 - c2).

    X = 0: equiconvex (equiconcave), X = +-1: plano-convex (concave) with the
    plane side second (X = 1) or first (X = -1), |X| > 1: menisci (X > 1:
    both centres of curvature in +z for a positive lens).

    Parameters
    ----------
    phi : number
        Power (1/f) of the thick lens, nonzero.
    n : number
        Refractive index of the glass.
    t : number
        Centre thickness.
    shape : number
        Shape factor X.

    Returns
    -------
    c1, c2 : numbers
        Vertex curvatures 1/R1 and 1/R2.
    """

    if phi == 0:
        raise ValueError("A lens of zero power has no shape factor.")
    a = (n - 1)**2*t*(shape**2 - 1)/(4*n)
    b = n - 1
    disc = b**2 + 4*a*phi
    if disc < 0:
        raise ValueError("No lens of power {:.4g} with thickness {:.4g} and shape {:.4g}."
                         .format(phi, t, shape))
    d = 2*phi/(b + np.sqrt(disc))  # c1 - c2, the root that goes to phi/(n - 1) for t -> 0

    return d*(shape + 1)/2, d*(shape - 1)/2


def principal_planes(c1, c2, n, t):
    """Principal planes of a thick lens in air.

    Parameters
    ----------
    c1, c2 : numbers
        Vertex curvatures.
    n : number
        Refractive index of the glass.
    t : number
        Centre thickness.

    Returns
    -------
    phi : number
        Power.
    delta, delta_prime : numbers
        Distances of the front principal plane from the first vertex and of
        the rear principal plane from the second vertex (positive in +z).
    """

    phi1, phi2 = (n - 1)*c1, (1 - n)*c2
    phi = phi1 + phi2 - phi1*phi2*t/n

    return phi, t*phi2/(n*phi), -t*phi1/(n*phi)


def thicken(diagram, wavelength, material, thickness, shape, units="mm", margin=None, stop=None,
            n_medium=1.0, length=None, check=True):
    """System of thick singlets with the first order properties of a thin
    lens diagram.

    Every inner point of the diagram becomes a singlet of the given power,
    centre thickness and shape factor whose front principal plane lies at
    the end of the space in front of the thin lens and whose rear principal
    plane at the start of the space behind it. The paraxial rays outside the
    glass, and with them the stop, the pupils, the NA, the magnification and
    the telecentricity, stay those of the diagram; only the distance between
    the principal planes of each lens (the hiatus) adds to the length. With
    `length`, the diagram is scaled (`DelanoDiagram.with_length()`) such that
    the thick system has this length from object to image.

    Parameters
    ----------
    diagram : DelanoDiagram
        Thin lens diagram with the object first and the image last, spaces of
        index n_medium.
    wavelength : number
        Wavelength in the length unit.
    material : tuple, callable, number or list of these
        Glass of the lenses: (n_d, V_d), a function n(wavelength_nm) or an
        index (as for `raytrace.Surface`), one for all lenses or a list with
        one per lens.
    thickness, shape : number or sequence
        Centre thicknesses and shape factors (see thick_lens_curvatures()), one
        for all lenses or one per lens.
    units : str, optional
        Length unit.
    margin : number, optional
        Semi diameters of the lenses are margin*(|y| + |ybar|) of the paraxial
        rays at each surface; None (default): unlimited, only the stop limits
        the aperture.
    stop : int, optional
        Which y axis crossing is the stop (see DelanoDiagram.to_system()).
    n_medium : number, optional
        Index of the spaces between the lenses (as the diagram's n).
    length : number, optional
        Object to image distance of the thick system (default: that of the
        diagram plus the hiatus of the lenses).
    check : bool, optional
        Raise a ValueError if lenses overlap (negative air spaces) or the
        stop falls into a lens (default). False returns such systems, e.g.
        for an optimization that penalizes the overlaps itself.

    Returns
    -------
    raytrace.System
    """

    K = len(diagram)
    n_lens = K - 2
    thickness = np.broadcast_to(np.asarray(thickness, dtype=float), (n_lens,))
    shape = np.broadcast_to(np.asarray(shape, dtype=float), (n_lens,))
    materials = material if isinstance(material, list) else [material]*n_lens
    if len(materials) != n_lens:
        raise ValueError("One material per lens or one for all.")
    wavelength_nm = wavelength*raytrace.NM_PER_UNIT[units]
    glass = [{"n_after": float(m)} if np.isscalar(m) else {"material": m} for m in materials]
    n_rel = [raytrace.Surface(**g).index(wavelength_nm, n_medium)/n_medium for g in glass]

    def lenses(d):
        phi = d.power/n_medium
        curvatures, offsets, hiatus = {}, {}, 0.0
        for j in range(n_lens):
            c1, c2 = thick_lens_curvatures(phi[j], n_rel[j], thickness[j], shape[j])
            _, delta, delta_prime = principal_planes(c1, c2, n_rel[j], thickness[j])
            # the space in front ends at the front principal plane (delta
            # behind the first vertex), the space behind starts at the rear
            # principal plane (delta_prime behind the second vertex):
            offsets[j + 1] = (delta, -delta_prime)
            curvatures[j + 1] = (c1, c2)
            hiatus += thickness[j] + delta_prime - delta
        return curvatures, offsets, hiatus

    if length is not None:
        for _ in range(50):
            hiatus = lenses(diagram)[2]
            new = diagram.with_length(length - hiatus)
            if abs(new.length - diagram.length) < 1e-12*length:
                break
            diagram = new
        diagram = new
    curvatures, offsets, _ = lenses(diagram)

    def make_lens(k, t_after):
        R1, R2 = (1/c if c != 0 else np.inf for c in curvatures[k])
        return [raytrace.Surface(R1, thickness[k - 1], label=diagram.labels[k] + "a",
                                 **glass[k - 1]),
                raytrace.Surface(R2, t_after, n_after=n_medium, label=diagram.labels[k] + "b")]

    surfaces, stop_index = diagram._surfaces(make_lens, stop, offsets, check)
    system = raytrace.System(surfaces, stop_index, wavelength,
                             object_distance=diagram.thickness[0] - offsets[1][0],
                             object_height=diagram.ybar[0], n_object=n_medium, units=units)
    _set_apertures(system, diagram.slopes[0, 1]/diagram.n[0], margin)

    return system


def _set_apertures(system, u, margin):
    """Stop semi diameter for the object space marginal ray slope u, the
    others margin*(|y| + |ybar|) of the paraxial rays (unlimited if margin is
    None)."""

    ys, _ = system._paraxial_trace(u*system.object_distance, u)
    system.surfaces[system.stop_index].semi_diameter = abs(ys[system.stop_index])
    if margin is None:
        return
    p = system.paraxial()
    for i, s in enumerate(system.surfaces):
        if i != system.stop_index:
            s.semi_diameter = margin*(abs(p.marginal[i, 0]) + abs(p.chief[i, 0]))
