#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Damped least squares optimization of `raytrace.System` lenses.

A `LensOptimizer` varies curvatures and thicknesses of a system and
minimizes the sum of squares of the residuals of a list of operands, with
scipy's trust region least squares (finite difference Jacobian). Each
operand is a function of the system that returns residuals, already divided
by their tolerance, so that 1 means "at the tolerance". Factories for the
usual operands:

- `ray_errors()`: transverse ray aberrations relative to the chief ray in
  the image plane on rings of the pupil (the spot sizes; zero defocus,
  spherical aberration, coma, astigmatism, field curvature),
- `distortion()`: chief ray image height against the nominal magnification,
- `image_telecentricity()`, `object_telecentricity()`: chief ray slopes in
  image and object space,
- `paraxial_target()`: any attribute of `raytrace.Paraxial` (magnification,
  efl, NA...),
- `clearances()`: penalties for overlapping lenses, too thin centres and
  edges and negative air spaces at the rim of the beam.

Constraints that should hold exactly are kept by the optimizer rather than
by operands: a total `length` (object to image) by solving the image
distance (or another thickness), and an `object_na` by setting the semi
diameter of the stop from the paraxial marginal ray in every step.

Lens design is a non-convex problem and a local optimizer starting from a
first order layout easily runs into total internal reflection or rays that
miss surfaces at full aperture and field. `LensOptimizer.run()` therefore
takes `stages` of increasing aperture and field (a curriculum, as in
end-to-end lens design by gradient descent, e.g. X. Yang, Q. Fu and W.
Heidrich, "Curriculum learning for ab initio deep learned refractive
optics", Nature Communications (2024)): the system is first corrected at a
small NA and field, which then grow to the specification.

Rays are traced without vignetting (clip=False); a ray that fails (it misses
a surface or is totally reflected) gives a large residual.
"""

import copy
from collections import namedtuple

import numpy as np

from .raytrace import CoordinateBreak, IdealLens, Rays


FAILED = 1e3  # residual of a failed ray (in units of the tolerance)

OptimizationResult = namedtuple("OptimizationResult", ["system", "cost", "stages", "success"])
OptimizationResult.__doc__ = """Result of LensOptimizer.run(): the optimized
system (at the full aperture and field), the final cost (half the sum of
squares of the residuals), per stage (aperture fraction, field fraction,
cost, function evaluations) and whether the last stage converged."""


def _pupil_rings(rings, spokes):
    """Normalized pupil points on the half pupil x >= 0 (meridional
    symmetry): `spokes` points per unit radius on each ring."""

    px, py = [], []
    for r in rings:
        n = max(3, int(round(spokes*r)))
        a = np.linspace(0, np.pi, n)
        px += list(r*np.sin(a))
        py += list(r*np.cos(a))

    return np.array(px), np.array(py)


def _trace(system, fields, px, py):
    """Image plane points and directions of the rays (px, py) at each field
    (the chief ray first), in one trace, cached on the system for the
    operands of one evaluation. Returns a list with (I, D) per field."""

    cache = system.__dict__.setdefault("_operand_cache", {})
    keys = [(float(f), px.tobytes(), py.tobytes()) for f in fields]
    missing = [(f, k) for f, k in zip(fields, keys) if k not in cache]
    if missing:
        px1, py1 = np.concatenate([[0.0], px]), np.concatenate([[0.0], py])
        rays = [system.rays(f, px1, py1) for f, _ in missing]
        joined = Rays(np.concatenate([np.reshape(r.points, (-1, 3)) for r in rays]),
                      np.concatenate([np.reshape(r.directions, (-1, 3)) for r in rays]),
                      np.concatenate([np.reshape(r.opl, -1) for r in rays]))
        I, D = system._image_hits(system.trace(joined, clip=False))
        for j, (_, k) in enumerate(missing):
            sl = slice(j*len(px1), (j + 1)*len(px1))
            cache[k] = (I[sl], D[sl])

    return [cache[k] for k in keys]


def _chief(system, field):
    """Image plane point and direction of the chief ray at field (reusing a
    cached trace of the field if there is one)."""

    for key, value in system.__dict__.get("_operand_cache", {}).items():
        if key[0] == float(field):
            return value[0][0], value[1][0]
    I, D = _trace(system, [field], np.zeros(0), np.zeros(0))[0]

    return I[0], D[0]


def _aimed_chief(system, fields):
    """Object space directions, image plane points and directions of the
    real chief rays at the (y) fields, aimed at the centre of the stop
    (whatever the system's ray_aiming), cached for the operands of one
    evaluation. Finite objects: Newton's method on the object space slope,
    all fields in one trace per step."""

    cache = system.__dict__.setdefault("_operand_cache", {})
    key = ("aimed",) + tuple(float(f) for f in fields)
    if key in cache:
        return cache[key]
    if system.infinite_object:
        rays = [system.rays(f, 0.0, 0.0, aim=True) for f in fields]
        d0 = np.concatenate([np.reshape(r.directions, (-1, 3)) for r in rays])
        rays = Rays(np.concatenate([np.reshape(r.points, (-1, 3)) for r in rays]), d0,
                    np.concatenate([np.reshape(r.opl, -1) for r in rays]))
    else:
        f = np.asarray(fields, float)
        n = len(f)
        O = np.stack([np.zeros(n), f, np.full(n, -system.object_distance)], axis=-1)

        def launch(b):
            d = np.stack([np.zeros(len(b)), b, np.ones(len(b))], axis=-1)

            return Rays(np.tile(O, (len(b)//n, 1)), d/np.linalg.norm(d, axis=1)[:, None],
                        np.zeros(len(b)))

        o, R = system.frames()[0][system.stop_index]
        b = system._object_slopes()[1]*f
        eps = 1e-7
        for _ in range(4):
            bundle = system.trace(launch(np.concatenate([b, b + eps])), clip=False,
                                  last=system.stop_index)
            y = ((bundle.points[-1] - o) @ R)[:, 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                step = y[:n]*eps/(y[n:] - y[:n])
            b = b - np.where(np.isfinite(step), step, 0.0)
            if np.all(np.abs(step) < 1e-12):
                break
        rays = launch(b)
        d0 = rays.directions
    I, D = system._image_hits(system.trace(rays, clip=False))
    cache[key] = (d0, I, D)

    return cache[key]


def _fields(system, fields):
    return [f*system.max_field for f in fields]


def ray_errors(fields=(0.0, 0.5, 0.75, 1.0), rings=(0.4, 0.7, 0.9, 1.0), spokes=6,
               tolerance=1e-4):
    """Transverse ray aberrations relative to the chief ray.

    Parameters
    ----------
    fields : sequence, optional
        Fields as fractions of the maximum field of the system (y fields).
    rings, spokes : optional
        Normalized pupil radii of the rings and the number of rays per unit
        radius on each (half) ring.
    tolerance : number, optional
        Tolerance in the length unit of the system (default 0.1 um for mm).

    Returns
    -------
    operand : callable
        operand(system) returns the x and y errors of all rays / tolerance.
    """

    px, py = _pupil_rings(rings, spokes)

    def operand(system):
        out = []
        for I, _ in _trace(system, _fields(system, fields), px, py):
            e = (I[1:, :2] - I[0, :2])/tolerance
            out.append(np.nan_to_num(e, nan=FAILED, posinf=FAILED, neginf=FAILED).ravel())

        return np.concatenate(out)

    return operand


def distortion(magnification, fields=(0.5, 0.75, 1.0), tolerance=1e-5, aim=False):
    """Chief ray image height minus magnification*object height (finite
    objects).

    Parameters
    ----------
    magnification : number
        Nominal lateral magnification.
    fields : sequence, optional
        Fields as fractions of the maximum object height.
    tolerance : number, optional
        Tolerance in the length unit (default 10 nm for mm).
    aim : bool, optional
        Use the real chief ray through the centre of the stop (ray aiming
        for this ray only) instead of the system's chief ray (which, without
        ray aiming, is aimed at the paraxial entrance pupil).

    Returns
    -------
    operand : callable
    """

    def operand(system):
        f = _fields(system, fields)
        if aim:
            out = (_aimed_chief(system, f)[1][:, 1] - magnification*np.array(f))/tolerance
        else:
            out = [(_chief(system, fk)[0][1] - magnification*fk)/tolerance for fk in f]

        return np.nan_to_num(np.array(out), nan=FAILED)

    return operand


def image_telecentricity(fields=(0.5, 0.75, 1.0), tolerance=1e-3, aim=False):
    """Image space chief ray slopes dy/dz.

    Parameters
    ----------
    fields : sequence, optional
        Fields as fractions of the maximum field.
    tolerance : number, optional
        Tolerance in radians (default 1 mrad).
    aim : bool, optional
        Use the real chief ray through the centre of the stop (see
        `distortion()`). Without ray aiming the system's chief ray is aimed
        at the paraxial entrance pupil, and the pupil aberrations of a high
        NA lens make its slopes differ from those of the real chief ray by
        tens of mrad.

    Returns
    -------
    operand : callable
    """

    def operand(system):
        f = _fields(system, fields)
        D = _aimed_chief(system, f)[2] if aim else np.array([_chief(system, fk)[1] for fk in f])
        out = D[:, 1]/D[:, 2]/tolerance

        return np.nan_to_num(np.array(out), nan=FAILED)

    return operand


def object_telecentricity(tolerance=1e-4, aim=False, fields=(0.5, 0.75, 1.0)):
    """Object space chief ray slope (finite objects): zero for an entrance
    pupil at infinity.

    Parameters
    ----------
    tolerance : number, optional
        Tolerance in radians.
    aim : bool, optional
        False (default): the paraxial chief ray slope at the maximum field.
        True: the slopes of the real chief rays through the centre of the
        stop at `fields` (see `image_telecentricity()`).
    fields : sequence, optional
        Fields as fractions of the maximum field (with aim).

    Returns
    -------
    operand : callable
    """

    def operand(system):
        if aim:
            d = _aimed_chief(system, _fields(system, fields))[0]

            return np.nan_to_num(d[:, 1]/d[:, 2]/tolerance, nan=FAILED)
        u_c = system._object_slopes()[1]

        return np.array([u_c*system.max_field/tolerance])

    return operand


def paraxial_target(name, value, tolerance):
    """Attribute `name` of `System.paraxial()` (e.g. "magnification", "efl",
    "na") against a target value.

    Parameters
    ----------
    name : str
        Attribute of `raytrace.Paraxial`.
    value : number
        Target.
    tolerance : number
        Tolerance.

    Returns
    -------
    operand : callable
    """

    def operand(system):
        return np.array([(getattr(system.paraxial(), name) - value)/tolerance])

    return operand


def clearances(min_air=1.0, min_center=None, min_edge=3.0, margin=1.05, tolerance=0.01,
               min_object_distance=None, min_image_distance=None):
    """Penalties for lenses that are too thin and air spaces that close.

    Thicknesses are checked on the axis and at the rim of the beam, the
    paraxial |y| + |ybar| (at the maximum field) times margin, the larger of
    the two surfaces of each space. During the stages of
    `LensOptimizer.run()` the rim is that of the full aperture and field.

    Parameters
    ----------
    min_air : number, optional
        Smallest air space, on axis and at the rim (also object and image
        distance).
    min_center : number, optional
        Smallest centre thickness of a lens (default: min_edge).
    min_edge : number, optional
        Smallest edge thickness of a lens.
    margin : number, optional
        Rim height relative to the paraxial beam.
    tolerance : number, optional
        A violation by tolerance gives a residual of 1 (default 10 um for
        mm); satisfied constraints give 0.
    min_object_distance, min_image_distance : number, optional
        Smallest distances from the object to the first vertex and from the
        last vertex to the image (working distance); default min_air.

    Returns
    -------
    operand : callable
    """

    min_center = min_edge if min_center is None else min_center

    min_object = min_air if min_object_distance is None else min_object_distance
    min_image = min_air if min_image_distance is None else min_image_distance

    def operand(system):
        p = system.paraxial()
        na_scale, field_scale = system.__dict__.get("_stage", (1.0, 1.0))
        rim = margin*(np.abs(p.marginal[:, 0])/na_scale + np.abs(p.chief[:, 0])/field_scale)
        n = system.n
        out = [min_object - system.object_distance]
        S = system.surfaces
        for i in range(len(S)):
            t = S[i].thickness*system.sigma[i + 1]
            glass = not system._is_gap(n[i + 1])
            if i == len(S) - 1:
                out.append(min_image - t)
                continue
            if isinstance(S[i], CoordinateBreak) or isinstance(S[i + 1], CoordinateBreak):
                continue
            r = max(rim[i], rim[i + 1])
            # a rim beyond the semi aperture of a sphere (conic) counts as
            # missing edge thickness, so that the penalty has a gradient:
            r_max = min(_max_radius(S[i]), _max_radius(S[i + 1]))
            r2 = np.array([min(r, 0.999*r_max)**2])
            edge = t + system.sigma[i + 1]*(S[i + 1].sag(r2)[0] - S[i].sag(r2)[0])
            edge -= max(0.0, r - 0.999*r_max)
            if glass:
                out += [min_center - t, min_edge - edge]
            else:
                out += [min_air - t, min_air - edge]
        out = np.nan_to_num(np.array(out), nan=10*min_edge)

        return np.maximum(0.0, out)/tolerance

    return operand


def _max_radius(surface):
    """Largest radius at which the conic of a surface has a sag."""

    c, K = surface.c, surface.conic
    if c == 0 or 1 + K <= 0:
        return np.inf

    return 1/(abs(c)*np.sqrt(1 + K))


class LensOptimizer:
    """Least squares optimization of curvatures and thicknesses.

    Parameters
    ----------
    system : raytrace.System
        Starting system (copied).
    operands : sequence of callables
        Functions of the system that return residuals (see the factories
        of this module).
    curvatures : sequence of int, optional
        Indices of the surfaces whose curvature varies. Default: all
        refracting and reflecting surfaces (stops and dummy surfaces that do
        not change the medium, ideal lenses and coordinate breaks stay).
    thicknesses : sequence of int, optional
        Indices of the surfaces whose thickness (to the next surface) varies.
        Default: all air spaces, except the solved one.
    object_distance : bool, optional
        Vary the object distance (finite objects).
    length : number, optional
        Keep the distance from object (finite objects; else the first vertex)
        to image at this value by solving the thickness of surface `solve`.
    solve : int, optional
        Surface whose thickness is solved for the length (default: the last
        one, the image distance).
    object_na : number, optional
        Keep the object space NA n*|u| of the paraxial marginal ray (finite
        objects) by setting the semi diameter of the stop.
    aspheres : sequence of int, optional
        Surfaces whose aspheric coefficients vary (even aspheres, see
        `raytrace.Surface`).
    asphere_orders : sequence of int, optional
        Powers of r that vary on these surfaces (default r**4 and r**6).
        Internally the variables are the sags A_k r**k at the rim of the
        paraxial full field beam.
    """

    def __init__(self, system, operands, curvatures=None, thicknesses=None, object_distance=False,
                 length=None, solve=-1, object_na=None, aspheres=(), asphere_orders=(4, 6)):
        self.system = system.copy()
        self.operands = list(operands)
        S = self.system.surfaces
        self.solve = solve % len(S)
        if curvatures is None:
            n = self.system.n
            curvatures = [i for i, s in enumerate(S)
                          if not isinstance(s, (IdealLens, CoordinateBreak))
                          and (s.mirror or n[i] != n[i + 1])]
        if thicknesses is None:
            n = self.system.n
            thicknesses = [i for i in range(len(S))
                           if self.system._is_gap(n[i + 1]) and i != self.solve]
        self.curvatures = list(curvatures)
        self.thicknesses = list(thicknesses)
        self.vary_object_distance = object_distance
        self.length = length
        self.object_na = object_na
        self.field_scale = 1.0
        self.na_scale = 1.0
        self._max_field = self.system.max_field
        self._apply(self.system)
        # aspheric coefficients as sags at the rim of the full beam:
        p = self.system.paraxial()
        rim = np.abs(p.marginal[:, 0]) + np.abs(p.chief[:, 0])
        self.aspheres = [(i, k, rim[i]) for i in aspheres for k in asphere_orders]
        for i in aspheres:
            a = list(S[i].aspheric) + [0.0]*(max(asphere_orders)//2 - 1 - len(S[i].aspheric))
            S[i].aspheric = tuple(a)

    @property
    def x(self):
        """Current values of the variables."""

        S = self.system.surfaces
        x = [S[i].c for i in self.curvatures] + [S[i].thickness for i in self.thicknesses]
        if self.vary_object_distance:
            x.append(self.system.object_distance)
        x += [S[i].aspheric[k//2 - 2]*r**k for i, k, r in self.aspheres]

        return np.array(x)

    def _scales(self):
        S = self.system.surfaces
        c = np.array([abs(S[i].c) for i in self.curvatures])
        c_scale = np.full(len(c), max(np.median(c), 1e-12) if len(c) else 1.0)
        t = np.array([abs(S[i].thickness) for i in self.thicknesses]
                     + ([self.system.object_distance] if self.vary_object_distance else []))
        t_scale = np.full(len(t), max(np.median(t), 1e-12) if len(t) else 1.0)
        a_scale = np.full(len(self.aspheres), 100*self.system.wavelength)

        return np.concatenate([c_scale, t_scale, a_scale])

    def _apply(self, system):
        """Solves and stage scaling."""

        system.max_field = self._max_field*self.field_scale
        system._stage = (self.na_scale, self.field_scale)
        if self.length is not None:
            S = system.surfaces
            others = sum(abs(s.thickness) for i, s in enumerate(S) if i != self.solve)
            start = 0.0 if system.infinite_object else system.object_distance
            S[self.solve].thickness = (self.length - start - others)*system.sigma[self.solve + 1]
        if self.object_na is not None:
            u = self.na_scale*self.object_na/system.n_object
            ys, _ = system._paraxial_trace(u*system.object_distance, u)
            system.surfaces[system.stop_index].semi_diameter = abs(ys[system.stop_index])

        return system

    def system_at(self, x):
        """Copy of the system with the variables x.

        Parameters
        ----------
        x : array
            Variables in the order of `x`.

        Returns
        -------
        raytrace.System
        """

        system = copy.deepcopy(self.system)
        system.__dict__.pop("_operand_cache", None)
        S = system.surfaces
        k = 0
        for i in self.curvatures:
            S[i].R = 1/x[k] if x[k] != 0 else np.inf
            k += 1
        for i in self.thicknesses:
            S[i].thickness = x[k]
            k += 1
        if self.vary_object_distance:
            system.object_distance = x[k]
            k += 1
        for i, order, r in self.aspheres:
            a = list(S[i].aspheric)
            a[order//2 - 2] = x[k]/r**order
            S[i].aspheric = tuple(a)
            k += 1

        return self._apply(system)

    def residuals(self, x=None):
        """All operand residuals at x (default: the current variables).

        Parameters
        ----------
        x : array, optional
            Variables.

        Returns
        -------
        array
        """

        system = self.system_at(self.x if x is None else x)
        with np.errstate(all="ignore"):
            return np.concatenate([np.atleast_1d(op(system)) for op in self.operands])

    def cost(self):
        """Half the sum of squares of the residuals of the current system."""

        return 0.5*np.sum(self.residuals()**2)

    def run(self, stages=((1.0, 1.0),), max_nfev=100, **kwargs):
        """Optimize, in stages of growing aperture and field.

        Parameters
        ----------
        stages : sequence of (aperture_fraction, field_fraction), optional
            Fractions of object_na and of the maximum field per stage
            (default: one stage at the full values). The aperture fraction
            needs object_na.
        max_nfev : int, optional
            Function evaluations per stage (without those of the Jacobian).
        **kwargs
            Passed to scipy.optimize.least_squares (e.g. x_scale="jac"
            instead of the default scales: the median curvature and
            thickness, 100 wavelengths for the aspheric sags).

        Returns
        -------
        OptimizationResult
        """

        from scipy.optimize import least_squares

        log = []
        r = None
        for na_scale, field_scale in stages:
            self.na_scale, self.field_scale = na_scale, field_scale
            options = dict(x_scale=self._scales(), max_nfev=max_nfev)
            options.update(kwargs)
            r = least_squares(self.residuals, self.x, **options)
            self.system = self.system_at(r.x)
            log.append((na_scale, field_scale, r.cost, r.nfev))
        self.na_scale, self.field_scale = 1.0, 1.0
        self.system = self._apply(self.system)
        self.system.__dict__.pop("_stage", None)

        return OptimizationResult(self.system.copy(), self.cost(), log,
                                  bool(r.success) if r is not None else True)
