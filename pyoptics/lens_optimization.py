#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Damped least squares optimization of `raytrace.System` lenses.

Two optimizers:

- `DampedLeastSquares`, the classical lens design optimizer (see the section
  further down): `Variable` objects (`Curvature`, `Thickness`, `AirSpace`,
  `Conic`, `Aspheric`, model glasses `GlassIndex`/`GlassAbbe` with
  `glass()`), solves (`solve_paraxial_focus()`, `solve_entrance_pupil()`,
  `solve_curvature_for_efl()`), operands (the factories below plus
  `polychromatic_ray_errors()`, `spot_rms()`, `wavefront()`, `seidel()`,
  `chromatic()`, `focal_shift()`) and `Constraint` objects held with
  Lagrange multipliers; a small glass catalogue `GLASSES` with
  `nearest_glass()` and the glass map outline `GLASS_MAP`
  (`glass_map_constraint()`). Notebook 58.
- `LensOptimizer`, described here, which uses scipy.

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

References
----------
K. Levenberg, "A method for the solution of certain non-linear problems in
least squares", Q. Appl. Math. 2, 164-168 (1944).

D. W. Marquardt, "An algorithm for least-squares estimation of nonlinear
parameters", J. Soc. Ind. Appl. Math. 11, 431-441 (1963).

M. A. Branch, T. F. Coleman and Y. Li, "A subspace, interior, and conjugate
gradient method for large-scale bound-constrained minimization problems", SIAM
J. Sci. Comput. 21, 1-23 (1999).

X. Yang, Q. Fu and W. Heidrich, "Curriculum learning for ab initio deep learned
refractive optics", Nat. Commun. 15 (2024).

R. Kingslake and R. B. Johnson, "Lens Design Fundamentals", 2nd ed., Academic
Press (2010).

G. H. Spencer, "A flexible automatic lens correction procedure", Appl. Opt. 2,
1257-1264 (1963).

C. G. Wynne and P. M. J. H. Wormell, "Lens design by computer", Appl. Opt. 2,
1233-1238 (1963).

H. B. Nielsen, "Damping parameter in Marquardt's method", Technical Report
IMM-REP-1999-05, Technical University of Denmark (1999).

M. J. Kidger, "Fundamental Optical Design", SPIE Press (2001).
"""

import copy
from collections import namedtuple

import numpy as np

from .glasses import Glass
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
    paraxial ``|y| + |ybar|`` (at the maximum field) times margin, the larger of
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
        Keep the object space NA ``n*|u|`` of the paraxial marginal ray (finite
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


# ---------------------------------------------------------------------------
# Damped least squares with variables, solves and constraints
# ---------------------------------------------------------------------------
#
# The optimizer below works in the way of the classical lens design programs
# (Spencer 1963, Wynne and Wormell 1963, Kidger 2001): a list of variables
# (curvatures, thicknesses, conics, aspheric terms, model glasses), a merit
# function that is the sum of squares of weighted operands (targets), solves
# that hold first order properties exactly after every change, and
# constraints that are kept by Lagrange multipliers instead of entering the
# merit function. Every iteration builds the Jacobian of operands and
# constraints by finite differences: the change of the system for each
# variable is the same as for a tolerance sensitivity (`System.
# with_radius_change()`, `with_axial_shift()`, `with_index_change()` for
# the curvatures, air spaces and glasses), and the columns of the Jacobian
# are the sensitivities of the operands. The step is the damped least
# squares (Levenberg-Marquardt) step with Marquardt's scaling,
#
#     (J^T J + lambda D) dx = -J^T r,   D = diag(J^T J),
#
# extended to the linearized active constraints A dx = b by Lagrange
# multipliers mu (Spencer 1963), i.e. the KKT system
#
#     [J^T J + lambda D   A^T] [dx]   [-J^T r]
#     [A                  0  ] [mu] = [  b   ],
#
# solved in the null space of A (the KKT matrix itself is badly conditioned
# when the operands have small tolerances).
# A step that lowers the merit function (plus a penalty on the constraint
# violation) is taken and lambda shrinks, else lambda grows and the step is
# solved again (Nielsen's damping update).


def _uncached(system):
    """Copy of a system without the operand cache."""

    other = system.copy()
    other.__dict__.pop("_operand_cache", None)

    return other


def _at_wavelength(system, wavelength):
    """The system at another wavelength (cached on the system for the
    operands of one evaluation)."""

    if wavelength is None or wavelength == system.wavelength:
        return system
    cache = system.__dict__.setdefault("_operand_cache", {})
    key = ("wavelength", float(wavelength))
    if key not in cache:
        other = system.copy()
        other.__dict__.pop("_operand_cache", None)
        other.wavelength = wavelength
        cache[key] = other

    return cache[key]


# --- glasses ---------------------------------------------------------------

# Nominal (n_d, V_d) of common glasses of the Schott catalogue (Schott AG,
# "Optical Glass - Data Sheets"), as model glasses for
# `materials.abbe_dispersion()`.
GLASSES = {
    "N-FK51A": (1.48656, 84.47),
    "N-FK5": (1.48749, 70.41),
    "N-BK7": (1.51680, 64.17),
    "N-K5": (1.52249, 59.48),
    "N-KF9": (1.52346, 51.54),
    "LLF1": (1.54814, 45.75),
    "N-BAK4": (1.56883, 55.98),
    "N-BAK1": (1.57250, 57.55),
    "N-BALF4": (1.57956, 53.87),
    "N-SK4": (1.61272, 58.63),
    "N-SSK8": (1.61773, 49.83),
    "N-PSK53A": (1.61800, 63.39),
    "N-F2": (1.62005, 36.43),
    "N-SK16": (1.62041, 60.32),
    "N-SF2": (1.64769, 33.82),
    "N-LAK22": (1.65113, 55.89),
    "N-BAF10": (1.67003, 47.11),
    "N-SF5": (1.67271, 32.25),
    "N-SF8": (1.68894, 31.31),
    "N-LAK9": (1.69100, 54.71),
    "N-SF10": (1.72828, 28.53),
    "N-LAF2": (1.74397, 44.85),
    "N-SF11": (1.78472, 25.68),
    "N-SF6": (1.80518, 25.36),
    "N-SF57": (1.84666, 23.78),
    "N-LASF9": (1.85025, 32.17),
    "N-SF66": (1.92286, 20.88),
}

# Outline of the region of the (n_d, V_d) glass map that real glasses fill
# (approximately, through the glasses at its rim), counterclockwise in
# (n_d, V_d).
GLASS_MAP = np.array([(1.4865, 84.5), (1.4875, 70.4), (1.5235, 51.5), (1.6200, 36.4),
                      (1.7850, 25.7), (1.9230, 20.9), (1.9500, 30.0), (1.8850, 40.8),
                      (1.8000, 46.5), (1.7300, 54.5), (1.6200, 63.4), (1.5000, 81.6)])


def _glass_coordinates(n_d, V_d):
    """Scaled glass map coordinates: an index step of 0.01 counts like a
    step of 1 in the Abbe number."""

    return np.array([100*n_d, V_d])


def nearest_glass(n_d, V_d, catalog=None):
    """Name and data of the catalogue glass nearest to a model glass.

    Distance on the glass map with an index step of 0.01 counting as much as
    a step of 1 in the Abbe number.

    Parameters
    ----------
    n_d, V_d : number
        Model glass.
    catalog : dict or glasses.Catalog, optional
        name -> (n_d, V_d) (model glasses, default `GLASSES`) or a catalogue
        of real glasses of `pyoptics.glasses` (name -> Glass), e.g.
        ``glasses.catalog(("schott", "ohara"))``.

    Returns
    -------
    name : str
    glass : tuple (n_d, V_d) or glasses.Glass
        The value of the catalogue (a Glass is the material with the
        vendor's dispersion formula).
    """

    catalog = GLASSES if catalog is None else catalog
    p = _glass_coordinates(n_d, V_d)
    name = min(catalog, key=lambda k: np.sum((_glass_coordinates(*_nd_Vd(catalog[k])) - p)**2))
    g = catalog[name]

    return name, (g if isinstance(g, Glass) else tuple(g))


def _nd_Vd(glass):
    return (glass.n_d, glass.V_d) if isinstance(glass, Glass) else glass


def glass_map_distance(n_d, V_d, outline=GLASS_MAP):
    """Signed distance of a model glass from the rim of the glass map, in
    the scaled coordinates of `nearest_glass()`: negative inside (real
    glasses exist around it), positive outside.

    Parameters
    ----------
    n_d, V_d : number
    outline : (N, 2) array, optional
        Polygon (n_d, V_d) of the region, default `GLASS_MAP`.

    Returns
    -------
    float
    """

    P = np.array([_glass_coordinates(*v) for v in outline])
    q = _glass_coordinates(n_d, V_d)
    d_min, inside = np.inf, False
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        ab = b - a
        t = np.clip(np.dot(q - a, ab)/np.dot(ab, ab), 0.0, 1.0)
        d_min = min(d_min, np.linalg.norm(q - a - t*ab))
        if (a[1] > q[1]) != (b[1] > q[1]):
            x = a[0] + (q[1] - a[1])*ab[0]/ab[1]
            inside ^= q[0] < x

    return -d_min if inside else d_min


# --- variables -------------------------------------------------------------

class Variable:
    """A scalar parameter of a system that the optimizer may change.

    Subclasses implement `get()` and `set()` (in place) and give a typical
    size of a change (`typical()`), which scales the variable for the
    damping and sets the finite difference step. `perturbed()` returns the
    system with the variable changed by h; for curvatures, air spaces and
    glasses it uses the tolerance sensitivities of `raytrace.System`.

    Parameters
    ----------
    bounds : (lower, upper), optional
        Limits of the value (None: unlimited). A step is cut back to the
        limits.
    """

    name = "variable"

    def __init__(self, bounds=(None, None)):
        self.bounds = tuple(bounds)

    def get(self, system):
        """Value of the variable in the system."""

        raise NotImplementedError

    def set(self, system, value):
        """Set the value (in place)."""

        raise NotImplementedError

    def typical(self, system):
        """Typical change of the value (scale of the variable)."""

        return 1.0

    def perturbed(self, system, h):
        """Copy of the system with the value changed by h."""

        other = _uncached(system)
        self.set(other, self.get(system) + h)

        return other

    def clip(self, value):
        """The value limited to the bounds."""

        lo, hi = self.bounds
        if lo is not None:
            value = max(value, lo)
        if hi is not None:
            value = min(value, hi)

        return value

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.name)


def _length_scale(system):
    S = system.surfaces
    size = np.sum(np.abs([s.thickness for s in S]))
    rims = [s.semi_diameter for s in S if s.semi_diameter is not None]

    return max(size, 2*max(rims, default=0.0), 1e-9)


class Curvature(Variable):
    """Vertex curvature c = 1/R of a surface (sensitivity:
    `System.with_radius_change()`)."""

    def __init__(self, surface, bounds=(None, None)):
        super().__init__(bounds)
        self.surface = surface
        self.name = "c{}".format(surface)

    def get(self, system):
        return system.surfaces[self.surface].c

    def set(self, system, value):
        system.surfaces[self.surface].R = 1/value if value != 0 else np.inf

    def typical(self, system):
        return 1/_length_scale(system)

    def perturbed(self, system, h):
        c = self.get(system)
        if c != 0 and c + h != 0:
            other = system.with_radius_change(self.surface, 1/(c + h) - 1/c)
            other.__dict__.pop("_operand_cache", None)
            return other

        return super().perturbed(system, h)


class Thickness(Variable):
    """Axial thickness behind a surface (glass or air). All surfaces behind
    it move with it."""

    def __init__(self, surface, bounds=(None, None)):
        super().__init__(bounds)
        self.surface = surface
        self.name = "t{}".format(surface)

    def get(self, system):
        return system.surfaces[self.surface].thickness

    def set(self, system, value):
        system.surfaces[self.surface].thickness = value

    def typical(self, system):
        return 0.1*_length_scale(system)


class AirSpace(Variable):
    """Position of a lens (surfaces first...last) between its neighbours:
    the air space in front of it, with the air space behind it shrinking by
    the same amount, so that the length stays (sensitivity:
    `System.with_axial_shift()`). The value is the thickness behind surface
    first - 1."""

    def __init__(self, first, last, bounds=(None, None)):
        super().__init__(bounds)
        if first < 1:
            raise ValueError("The lens needs a surface in front of it.")
        self.first, self.last = first, last
        self.name = "shift{}-{}".format(first, last)

    def get(self, system):
        return system.surfaces[self.first - 1].thickness

    def set(self, system, value):
        dz = value - self.get(system)
        system.surfaces[self.first - 1].thickness += dz
        system.surfaces[self.last].thickness -= dz

    def typical(self, system):
        return 0.1*_length_scale(system)

    def perturbed(self, system, h):
        other = system.with_axial_shift(self.first, self.last, h)
        other.__dict__.pop("_operand_cache", None)

        return other


class ObjectDistance(Variable):
    """Distance of a finite object in front of the first vertex."""

    name = "object_distance"

    def get(self, system):
        return system.object_distance

    def set(self, system, value):
        system.object_distance = value

    def typical(self, system):
        return 0.1*_length_scale(system)


class Conic(Variable):
    """Conic constant of a surface."""

    def __init__(self, surface, bounds=(None, None)):
        super().__init__(bounds)
        self.surface = surface
        self.name = "K{}".format(surface)

    def get(self, system):
        return system.surfaces[self.surface].conic

    def set(self, system, value):
        system.surfaces[self.surface].conic = value

    def typical(self, system):
        return 1.0


class Aspheric(Variable):
    """Even aspheric coefficient A_order (of r**order) of a surface, as the
    sag A r**order at the radius `rim` (the variable is a length, which is
    better scaled than the coefficient)."""

    def __init__(self, surface, order, rim, bounds=(None, None)):
        super().__init__(bounds)
        if order < 4 or order % 2:
            raise ValueError("Even orders from 4.")
        self.surface, self.order, self.rim = surface, order, rim
        self.name = "A{}_{}".format(order, surface)

    def _coefficients(self, system):
        a = list(system.surfaces[self.surface].aspheric)
        k = self.order//2 - 2

        return a + [0.0]*(k + 1 - len(a)), k

    def get(self, system):
        a, k = self._coefficients(system)

        return a[k]*self.rim**self.order

    def set(self, system, value):
        a, k = self._coefficients(system)
        a[k] = value/self.rim**self.order
        system.surfaces[self.surface].aspheric = tuple(a)

    def typical(self, system):
        return 100*system.wavelength


def _model_glass(surface):
    m = surface.material
    if m is None and surface.n_after is not None:
        return float(surface.n_after), np.inf
    if isinstance(m, tuple):
        return float(m[0]), float(m[1])
    if isinstance(m, Glass):  # a catalogue glass becomes its model glass
        return m.n_d, m.V_d
    raise ValueError("A glass variable needs a surface with a model glass material=(n_d, V_d), "
                     "a catalogue glass or n_after.")


class GlassIndex(Variable):
    """Index n_d of the model glass (n_d, V_d) behind a surface. The Abbe
    number stays, so the dispersion n_F - n_C = (n_d - 1)/V_d changes a
    little with n_d; the sensitivity is close to that of
    `System.with_index_change()`."""

    def __init__(self, surface, bounds=(1.45, 2.0)):
        super().__init__(bounds)
        self.surface = surface
        self.name = "n{}".format(surface)

    def get(self, system):
        return _model_glass(system.surfaces[self.surface])[0]

    def set(self, system, value):
        s = system.surfaces[self.surface]
        s.material = (float(value), _model_glass(s)[1])
        s.n_after = None

    def typical(self, system):
        return 0.05


class GlassAbbe(Variable):
    """Abbe number V_d of the model glass (n_d, V_d) behind a surface."""

    def __init__(self, surface, bounds=(18.0, 90.0)):
        super().__init__(bounds)
        self.surface = surface
        self.name = "V{}".format(surface)

    def get(self, system):
        return _model_glass(system.surfaces[self.surface])[1]

    def set(self, system, value):
        s = system.surfaces[self.surface]
        s.material = (_model_glass(s)[0], float(value))
        s.n_after = None

    def typical(self, system):
        return 5.0


def glass(surface, n_bounds=(1.45, 2.0), V_bounds=(18.0, 90.0)):
    """Both variables of the model glass behind a surface: [GlassIndex,
    GlassAbbe]. Keep it on the glass map with `glass_map_constraint()` and
    replace it by a real glass with `DampedLeastSquares.snap_glasses()`.

    Parameters
    ----------
    surface : int
        Surface behind which the glass starts (its material must be a
        model glass (n_d, V_d)).
    n_bounds, V_bounds : (lower, upper), optional

    Returns
    -------
    list of Variable
    """

    return [GlassIndex(surface, n_bounds), GlassAbbe(surface, V_bounds)]


def curvatures(system, surfaces=None, bounds=(None, None)):
    """Curvature variables of all refracting and reflecting surfaces (or of
    `surfaces`)."""

    if surfaces is None:
        n = system.n
        surfaces = [i for i, s in enumerate(system.surfaces)
                    if not isinstance(s, (IdealLens, CoordinateBreak))
                    and (s.mirror or n[i] != n[i + 1])]

    return [Curvature(i, bounds) for i in surfaces]


def air_spaces(system, bounds=(None, None), image=False):
    """Thickness variables of the air spaces between the lenses (and of the
    image distance with image=True)."""

    n = system.n
    last = len(system.surfaces) - 1

    return [Thickness(i, bounds) for i in range(len(system.surfaces))
            if system._is_gap(n[i + 1]) and (image or i != last)]


# --- solves ----------------------------------------------------------------

def solve_paraxial_focus(offset=0.0):
    """Solve: image plane in the paraxial focus (marginal ray height zero),
    plus offset (a defocus)."""

    def solve(system):
        system.surfaces[-1].thickness = 0.0
        system.surfaces[-1].thickness = system.paraxial().image_distance + offset

    return solve


def solve_entrance_pupil(radius):
    """Solve: semi diameter of the stop for a paraxial entrance pupil of
    the given radius (object at infinity), i.e. a fixed F-number while the
    lenses in front of the stop change."""

    def solve(system):
        A = system._ep_matrix()[0]
        system.surfaces[system.stop_index].semi_diameter = abs(radius*A)

    return solve


def solve_length(length, surface=-1):
    """Solve: thickness behind `surface` (default: the image distance) for
    a given distance from the object (finite objects; else from the first
    vertex) to the image plane, as `LensOptimizer(length=...)`."""

    def solve(system):
        S = system.surfaces
        k = surface % len(S)
        others = sum(abs(s.thickness) for i, s in enumerate(S) if i != k)
        start = 0.0 if system.infinite_object else system.object_distance
        S[k].thickness = (length - start - others)*system.sigma[k + 1]

    return solve


def solve_object_na(na):
    """Solve: semi diameter of the stop for an object side NA ``n*|u|`` of
    the paraxial marginal ray (finite objects), as
    `LensOptimizer(object_na=...)`."""

    def solve(system):
        u = na/system.n_object
        ys, _ = system._paraxial_trace(u*system.object_distance, u)
        system.surfaces[system.stop_index].semi_diameter = abs(ys[system.stop_index])

    return solve


def solve_curvature_for_efl(surface, efl):
    """Solve: curvature of a surface for the given effective focal length
    (the angle solve of the last surface in classical programs). The power
    of the system is linear in the curvature of one surface, so two
    paraxial traces give the exact solution."""

    def solve(system):
        S = system.surfaces[surface]
        p = []
        for c in (0.0, 1e-3):
            S.R = 1/c if c else np.inf
            p.append(system.power())
        target = system.n[-1]/efl
        c = 1e-3*(target - p[0])/(p[1] - p[0])
        S.R = 1/c if c != 0 else np.inf

    return solve


# --- more operands -----------------------------------------------------------

def _rays_at(system, fields, wavelengths, px, py):
    """(wavelength, field, I, D) of rays (chief ray first) at each
    combination."""

    out = []
    for w in ([None] if wavelengths is None else wavelengths):
        s = _at_wavelength(system, w)
        for f, (I, D) in zip(fields, _trace(s, _fields(s, fields), px, py)):
            out.append((w, f, I, D))

    return out


def _clean(e):
    return np.nan_to_num(e, nan=FAILED, posinf=FAILED, neginf=FAILED)


def polychromatic_ray_errors(wavelengths, fields=(0.0, 0.7, 1.0), rings=(0.4, 0.7, 0.9, 1.0),
                             spokes=6, tolerance=1e-3, weights=None):
    """Transverse ray errors at several wavelengths, relative to the chief
    ray of the system's own wavelength (so that lateral colour counts).

    Parameters
    ----------
    wavelengths : sequence
        Wavelengths in the length unit.
    fields : sequence, optional
        Fractions of the maximum field.
    rings, spokes : optional
        Pupil sampling, see `ray_errors()`.
    tolerance : number, optional
        In the length unit.
    weights : sequence, optional
        Weight per wavelength (default 1).

    Returns
    -------
    operand : callable
    """

    px, py = _pupil_rings(rings, spokes)
    weights = np.ones(len(wavelengths)) if weights is None else np.asarray(weights, float)

    def operand(system):
        chief = {f: I[0, :2] for _, f, I, _ in _rays_at(system, fields, None, px, py)}
        out = []
        for k, w in enumerate(wavelengths):
            for _, f, I, _ in _rays_at(system, fields, [w], px, py):
                e = weights[k]*(I[1:, :2] - chief[f])/tolerance
                out.append(_clean(e).ravel())

        return np.concatenate(out)

    return operand


def spot_rms(fields=(0.0, 0.7, 1.0), wavelengths=None, rings=(0.4, 0.7, 0.9, 1.0), spokes=6,
             tolerance=1e-3, about="centroid"):
    """RMS spot radius per field (all wavelengths together), one residual
    per field.

    Fewer residuals than `ray_errors()`, so the Jacobian carries less
    information, but the merit function reads directly as a spot size.
    The rings are weighted by their share of the pupil area (a ring at
    radius r for the annulus around it).

    Parameters
    ----------
    fields : sequence, optional
        Fractions of the maximum field.
    wavelengths : sequence, optional
        Default: the wavelength of the system.
    rings, spokes : optional
        Pupil sampling.
    tolerance : number, optional
        In the length unit.
    about : {"centroid", "chief"}, optional
        Reference of the spot: centroid of all rays or the chief ray of the
        system's wavelength.

    Returns
    -------
    operand : callable
    """

    px, py = _pupil_rings(rings, spokes)
    r = np.hypot(px, py)
    radii = np.sort(np.unique(np.round(rings, 12)))
    edges = np.concatenate([[0.0], (radii[1:] + radii[:-1])/2, [1.0]])
    area = {float(round(rk, 12)): e1**2 - e0**2 for rk, e0, e1 in zip(radii, edges[:-1], edges[1:])}
    w_ray = np.array([area[float(round(rk, 12))] for rk in r])
    for rk in radii:
        on = np.isclose(r, rk)
        w_ray[on] /= on.sum()

    def operand(system):
        chief = {f: I[0, :2] for _, f, I, _ in _rays_at(system, fields, None, px, py)}
        hits = {f: [] for f in fields}
        for _, f, I, _ in _rays_at(system, fields, wavelengths, px, py):
            hits[f].append(I[1:, :2])
        out = []
        for f in fields:
            P = np.concatenate(hits[f])
            w = np.tile(w_ray, len(hits[f]))
            ok = np.all(np.isfinite(P), axis=1)
            if not np.all(ok):
                out.append(FAILED)
                continue
            ref = (w @ P)/w.sum() if about == "centroid" else chief[f]
            out.append(np.sqrt(w @ np.sum((P - ref)**2, axis=1)/w.sum())/tolerance)

        return _clean(np.array(out))

    return operand


def wavefront(fields=(0.0, 0.7, 1.0), wavelengths=None, rings=(0.4, 0.7, 0.9, 1.0), spokes=6,
              tolerance=0.05):
    """Optical path differences of the rays relative to the chief ray, on
    the reference sphere about the chief ray image point, in waves.

    Parameters
    ----------
    fields : sequence, optional
        Fractions of the maximum field.
    wavelengths : sequence, optional
        Default: the wavelength of the system. The OPD at each is in waves
        of that wavelength.
    rings, spokes : optional
        Pupil sampling.
    tolerance : number, optional
        In waves.

    Returns
    -------
    operand : callable
    """

    px, py = _pupil_rings(rings, spokes)
    px1, py1 = np.concatenate([[0.0], px]), np.concatenate([[0.0], py])

    def operand(system):
        out = []
        for w in ([None] if wavelengths is None else wavelengths):
            s = _at_wavelength(system, w)
            for f in _fields(s, fields):
                bundle = s.trace(s.rays(f, px1, py1), clip=False)
                chief = _chief_bundle(bundle)
                W = s._opd(bundle, chief)[1:]/s.wavelength
                out.append(_clean(W/tolerance))

        return np.concatenate(out)

    return operand


def _chief_bundle(bundle):
    from .raytrace import RayBundle

    return RayBundle(bundle.points[:, :1], bundle.directions[:, :1], bundle.opl[:, :1],
                     bundle.valid[:1], bundle.n)


_SEIDEL_TERMS = {"SI": 0, "SII": 1, "SIII": 2, "SIV": 3, "SV": 4,
                 "W040": 0, "W131": 1, "W222": 2, "W220": 3, "W311": 4}


def seidel(term, value=0.0, tolerance=0.01, surfaces=None, field=None):
    """A Seidel sum against a target: one residual.

    Parameters
    ----------
    term : str
        "SI" ... "SV" (Welford's sums S_I...S_V in the length unit) or the
        wavefront coefficients "W040", "W131", "W222", "W220", "W311" in
        waves (see `raytrace.Seidel`).
    value : number, optional
        Target.
    tolerance : number, optional
    surfaces : sequence of int, optional
        Sum only over these surfaces (default: all), e.g. the share of one
        group.
    field : number, optional
        Field of the chief ray (default: the maximum field).

    Returns
    -------
    operand : callable
    """

    k = _SEIDEL_TERMS[term.upper()]
    waves = term.upper().startswith("W")

    def operand(system):
        sd = system.seidel(field)
        M = sd.W if waves else sd.S
        idx = slice(None) if surfaces is None else list(surfaces)
        v = np.nansum(M[idx, k])

        return np.array([(v - value)/tolerance])

    return operand


def chromatic(kind, d_wavelength, value=0.0, tolerance=1e-3, field=None):
    """Primary axial or lateral colour from the paraxial rays (see
    `System.chromatic_seidel()`): one residual.

    Parameters
    ----------
    kind : {"axial", "lateral"}
        Axial colour: shift of the paraxial focus between the two ends of
        the band (length unit); lateral colour: difference of the paraxial
        chief ray heights.
    d_wavelength : number
        Width of the band, e.g. F - C (486.1 - 656.3 nm).
    value, tolerance : number, optional
    field : number, optional

    Returns
    -------
    operand : callable
    """

    def operand(system):
        c = system.chromatic_seidel(d_wavelength, field)

        return np.array([(getattr(c, kind) - value)/tolerance])

    return operand


def focal_shift(wavelength_a, wavelength_b, value=0.0, tolerance=1e-3):
    """Paraxial image distance at wavelength_a minus that at wavelength_b
    (exact paraxial traces at both, not the first order estimate of
    `chromatic()`); e.g. F minus C for an achromat, or the secondary
    spectrum d minus F.

    Parameters
    ----------
    wavelength_a, wavelength_b : number
        Wavelengths in the length unit.
    value, tolerance : number, optional

    Returns
    -------
    operand : callable
    """

    def operand(system):
        za = _at_wavelength(system, wavelength_a).paraxial().image_distance
        zb = _at_wavelength(system, wavelength_b).paraxial().image_distance

        return np.array([(za - zb - value)/tolerance])

    return operand


# --- constraints -----------------------------------------------------------

class Constraint:
    """A function of the system kept between bounds.

    Given to `DampedLeastSquares` as a constraint it is held by Lagrange
    multipliers (exactly, to first order in every step, for equalities and
    for inequalities that would be violated). Used as an operand (here or in
    `LensOptimizer`) it is a penalty: zero inside the bounds, the violation
    divided by the tolerance outside.

    Parameters
    ----------
    function : callable
        function(system) returns a number or an array.
    lower, upper : number, optional
        Bounds (None: open); lower == upper for an equality.
    tolerance : number, optional
        Violation that counts as 1 (penalty) and that is accepted as
        satisfied in the optimizer (x 1e-3).
    name : str, optional
    """

    def __init__(self, function, lower=None, upper=None, tolerance=1e-3, name=None):
        if lower is None and upper is None:
            raise ValueError("Give a lower or upper bound.")
        self.function, self.lower, self.upper = function, lower, upper
        self.tolerance = tolerance
        self.name = name or getattr(function, "__name__", "constraint")

    @property
    def equality(self):
        return self.lower is not None and self.lower == self.upper

    def values(self, system):
        """Raw values of the function."""

        with np.errstate(all="ignore"):
            return np.nan_to_num(np.atleast_1d(np.asarray(self.function(system), float)),
                                 nan=1/FAILED)

    def violation(self, v):
        """Signed violation of the values v: positive above upper, negative
        below lower, zero inside."""

        out = np.zeros_like(v)
        if self.upper is not None:
            out = np.where(v > self.upper, v - self.upper, out)
        if self.lower is not None:
            out = np.where(v < self.lower, v - self.lower, out)

        return out

    def __call__(self, system):
        return self.violation(self.values(system))/self.tolerance

    def __repr__(self):
        return "Constraint({}, lower={}, upper={})".format(self.name, self.lower, self.upper)


def paraxial_value(name, field=None):
    """Function system -> attribute `name` of System.paraxial() (for a
    Constraint)."""

    def f(system):
        return getattr(system.paraxial(field), name)
    f.__name__ = name

    return f


def total_length(system):
    """First vertex to image plane (absolute thicknesses)."""

    return np.sum(np.abs([s.thickness for s in system.surfaces]))


def center_thickness(surface):
    """Function system -> thickness behind a surface."""

    def f(system):
        return system.surfaces[surface].thickness
    f.__name__ = "t{}".format(surface)

    return f


def edge_thickness(surface, height=None, margin=1.05):
    """Function system -> thickness between a surface and the next at the
    height (default: the paraxial rim ``|y| + |ybar|`` of the beam at full
    aperture and field, times margin, the larger of the two surfaces).
    Beyond the semi aperture of a conic the missing sag counts as negative
    thickness, so that the constraint keeps a gradient."""

    def f(system):
        S = system.surfaces
        a, b = S[surface], S[surface + 1]
        if height is None:
            p = system.paraxial()
            rim = margin*(np.abs(p.marginal[:, 0]) + np.abs(p.chief[:, 0]))
            r = max(rim[surface], rim[surface + 1])
        else:
            r = height
        r_max = min(_max_radius(a), _max_radius(b))
        r2 = np.array([min(r, 0.999*r_max)**2])
        sigma = system.sigma[surface + 1]
        edge = a.thickness*sigma + sigma*(b.sag(r2)[0] - a.sag(r2)[0])

        return edge - max(0.0, r - 0.999*r_max)
    f.__name__ = "edge{}".format(surface)

    return f


def glass_map_constraint(surface, margin=0.0):
    """Constraint that keeps the model glass behind a surface on the glass
    map (`glass_map_distance()` <= -margin)."""

    def f(system):
        return glass_map_distance(*_model_glass(system.surfaces[surface]))
    f.__name__ = "glass_map{}".format(surface)

    return Constraint(f, upper=-margin, tolerance=0.1, name=f.__name__)


# --- the optimizer -----------------------------------------------------------

DLSResult = namedtuple("DLSResult", ["system", "x", "merit", "history", "success", "message",
                                     "multipliers"])
DLSResult.__doc__ = """Result of DampedLeastSquares.run(): the optimized
system, its variables x, the merit function (sum of squares of the
operands), the history (one dict per iteration with merit, damping,
violation of the constraints and x), whether it converged, why it stopped
and the Lagrange multipliers of the active constraints of the last step
(name -> multiplier)."""


class DampedLeastSquares:
    """Damped least squares lens optimization with solves and constraints.

    Parameters
    ----------
    system : raytrace.System
        Starting system (copied).
    variables : sequence of Variable
    operands : sequence of callables
        Functions system -> residuals, already divided by their tolerance
        (the factories of this module, or Constraint objects as penalties).
        The merit function is the sum of the squared residuals.
    constraints : sequence of Constraint, optional
        Held with Lagrange multipliers.
    solves : sequence of callables, optional
        Functions that change the system in place after every change of the
        variables (`solve_paraxial_focus()`, `solve_entrance_pupil()`,
        `solve_curvature_for_efl()`).
    weights : sequence, optional
        Weight of each operand (default 1).
    damping : number, optional
        Initial damping lambda (of every run()).
    """

    def __init__(self, system, variables, operands, constraints=(), solves=(), weights=None,
                 damping=1e-2):
        self.variables = list(variables)
        self.operands = list(operands)
        self.constraints = list(constraints)
        self.solves = list(solves)
        self.weights = np.ones(len(self.operands)) if weights is None else np.asarray(weights, float)
        self.damping = damping
        self.system = self._solve(_uncached(system))
        self._scale = np.array([v.typical(self.system) for v in self.variables])

    def _solve(self, system):
        for s in self.solves:
            s(system)
        system.__dict__.pop("_operand_cache", None)

        return system

    @property
    def x(self):
        """Current values of the variables."""

        return np.array([v.get(self.system) for v in self.variables])

    def system_at(self, x):
        """Copy of the system with the variables x (and the solves applied).

        Parameters
        ----------
        x : array

        Returns
        -------
        raytrace.System
        """

        system = _uncached(self.system)
        for v, value in zip(self.variables, x):
            v.set(system, value)

        return self._solve(system)

    def residuals(self, system=None):
        """Weighted operand residuals of a system (default: the current
        one).

        Parameters
        ----------
        system : raytrace.System, optional

        Returns
        -------
        array
        """

        system = self.system if system is None else system
        with np.errstate(all="ignore"):
            r = [w*np.atleast_1d(op(system)) for w, op in zip(self.weights, self.operands)]
        system.__dict__.pop("_operand_cache", None)

        return np.concatenate(r) if r else np.zeros(0)

    def constraint_values(self, system=None):
        """Values of all constraints (concatenated) and their signed
        violations divided by the tolerances."""

        system = self.system if system is None else system
        if not self.constraints:
            return np.zeros(0), np.zeros(0)
        v = [c.values(system) for c in self.constraints]
        viol = [c.violation(vk)/c.tolerance for c, vk in zip(self.constraints, v)]

        return np.concatenate(v), np.concatenate(viol)

    def merit(self, system=None):
        """Merit function: the sum of the squared residuals."""

        return float(np.sum(self.residuals(system)**2))

    def _constraint_info(self):
        lo, hi, tol, names = [], [], [], []
        for c in self.constraints:
            m = len(c.values(self.system))
            lo += [-np.inf if c.lower is None else c.lower]*m
            hi += [np.inf if c.upper is None else c.upper]*m
            tol += [c.tolerance]*m
            names += [c.name if m == 1 else "{}[{}]".format(c.name, k) for k in range(m)]

        return np.array(lo), np.array(hi), np.array(tol), names

    def jacobian(self, system=None):
        """Jacobians of the residuals and of the constraint values with
        respect to the variables (forward differences; each column is the
        sensitivity of the operands to one variable).

        Parameters
        ----------
        system : raytrace.System, optional
            Default: the current system.

        Returns
        -------
        J : (residuals, variables) array
        A : (constraint values, variables) array
        r : residuals at the system
        g : constraint values at the system
        """

        system = self.system if system is None else system
        r = self.residuals(system)
        g = self.constraint_values(system)[0]
        J = np.zeros((len(r), len(self.variables)))
        A = np.zeros((len(g), len(self.variables)))
        for k, v in enumerate(self.variables):
            h = 1e-6*self._scale[k]
            lo, hi = v.bounds
            x0 = v.get(system)
            if hi is not None and x0 + h > hi:
                h = -h
            s = self._solve(v.perturbed(system, h))
            h = v.get(s) - x0 if v.get(s) != x0 else h  # the step actually taken
            J[:, k] = (self.residuals(s) - r)/h
            if len(g):
                A[:, k] = (self.constraint_values(s)[0] - g)/h

        return J, A, r, g

    def _step(self, J, A, r, target, g, lam, active, bounds):
        """Damped least squares step in scaled variables with the active
        constraints held to first order."""

        D = self._scale
        Js = J*D
        H = Js.T @ Js
        diag = np.diag(H).copy()
        diag = np.maximum(diag, 1e-9*max(diag.max(initial=0.0), 1e-30) + 1e-30)
        H = H + lam*np.diag(diag)
        grad = Js.T @ r
        free = np.ones(len(D), bool)
        free[bounds] = False
        H = H[np.ix_(free, free)]
        grad = grad[free]
        if np.any(active):
            As = (A*D)[np.ix_(active, free)]
            b = (target - g)[active]
            # null space method (better conditioned than the KKT matrix):
            # dz = dz_p + Z y with As dz_p = b and As Z = 0
            U, sv, Vt = np.linalg.svd(As)
            rank = int(np.sum(sv > 1e-10*sv[0])) if len(sv) and sv[0] > 0 else 0
            dz_p = Vt[:rank].T @ ((U[:, :rank].T @ b)/sv[:rank])
            Z = Vt[rank:].T
            if Z.shape[1]:
                y = np.linalg.solve(Z.T @ H @ Z, -Z.T @ (grad + H @ dz_p))
                dz = dz_p + Z @ y
            else:
                dz = dz_p
            mu = np.linalg.lstsq(As.T, -(grad + H @ dz), rcond=None)[0]
        else:
            dz, mu = np.linalg.solve(H, -grad), np.zeros(0)
        dx = np.zeros(len(D))
        dx[free] = dz*D[free]

        return dx, mu

    def step(self, damping):
        """One damped least squares step from the current system (equality
        constraints held to first order, inequalities ignored), without
        taking it.

        Parameters
        ----------
        damping : number
            lambda; 0 is the Gauss-Newton step, large values give short
            steps along the scaled steepest descent.

        Returns
        -------
        dx : array
            Change of the variables.
        """

        J, A, r, g = self.jacobian()
        self._lo, self._hi = self._constraint_info()[:2]
        active = self._lo == self._hi

        return self._step(J, A, r, np.clip(g, self._lo, self._hi), g, damping, active,
                          np.zeros(len(self.variables), bool))[0]

    def _penalty(self, viol):
        return self.constraint_weight*float(np.sum(viol**2))

    def run(self, max_iterations=50, ftol=1e-6, constraint_weight=1e4, verbose=False):
        """Optimize.

        Parameters
        ----------
        max_iterations : int, optional
            Number of Jacobians (outer iterations).
        ftol : number, optional
            Stop when an accepted step lowers the merit function by less
            than this fraction and the constraints hold.
        constraint_weight : number, optional
            Weight of the squared constraint violations (in tolerances)
            against the merit function when steps are accepted or rejected.
        verbose : bool, optional
            Print one line per iteration.

        Returns
        -------
        DLSResult
        """

        self.constraint_weight = constraint_weight
        self._lo, self._hi, tol, names = self._constraint_info()
        lam = self.damping
        nu = 2.0
        lam_ok = lam
        history = []
        success, message, multipliers = False, "maximum number of iterations", {}
        x = self.x
        phi = self.merit() + self._penalty(self.constraint_values()[1])
        for it in range(max_iterations):
            J, A, r, g = self.jacobian()
            viol = self.constraint_values()[1]
            history.append(dict(merit=float(np.sum(r**2)), damping=lam,
                                violation=float(np.max(np.abs(viol), initial=0.0)), x=x.copy()))
            if verbose:
                print("{:3d}  merit {:.6g}  damping {:.3g}  violation {:.3g}".format(
                    it, history[-1]["merit"], lam, history[-1]["violation"]))
            # variables at a bound that the gradient pushes outwards stay:
            grad = J.T @ r
            at_bound = np.array([(v.bounds[0] is not None and xi <= v.bounds[0] and gi > 0)
                                 or (v.bounds[1] is not None and xi >= v.bounds[1] and gi < 0)
                                 for v, xi, gi in zip(self.variables, x, grad)], bool)
            accepted = False
            for _ in range(30):
                # active set: equalities, violated inequalities and those that
                # the step would violate
                active = (self._lo == self._hi) | (g < self._lo) | (g > self._hi)
                target = np.clip(g, self._lo, self._hi)
                for _ in range(len(g) + 1):
                    dx, mu = self._step(J, A, r, target, g, lam, active, at_bound)
                    g_pred = g + A @ dx
                    below = ~active & (g_pred < self._lo - 1e-3*tol)
                    above = ~active & (g_pred > self._hi + 1e-3*tol)
                    if not np.any(below | above):
                        break
                    target = np.where(below, self._lo, np.where(above, self._hi, target))
                    active |= below | above
                x_new = np.array([v.clip(xi + d) for v, xi, d in zip(self.variables, x, dx)])
                s_new = self.system_at(x_new)
                phi_new = self.merit(s_new) + self._penalty(self.constraint_values(s_new)[1])
                if np.isfinite(phi_new) and phi_new < phi:
                    # gain ratio for Nielsen's update
                    Js = J @ dx
                    predicted = float(np.sum(r**2) - np.sum((r + Js)**2))
                    rho = (phi - phi_new)/predicted if predicted > 0 else 0.5
                    lam *= max(1/3, 1 - (2*rho - 1)**3)
                    nu = 2.0
                    lam_ok = lam
                    accepted = True
                    break
                lam *= nu
                nu *= 2
            if not accepted:
                success, message = True, "no step lowers the merit function (local minimum)"
                break
            gain = (phi - phi_new)/max(phi, 1e-300)
            self.system, x, phi = s_new, x_new, phi_new
            multipliers = dict(zip([n for n, a in zip(names, active) if a], mu))
            viol = self.constraint_values()[1]
            if gain < ftol and np.all(np.abs(viol) < 1e-3):
                success, message = True, "converged"
                break
        r = self.residuals()
        viol = self.constraint_values()[1]
        history.append(dict(merit=float(np.sum(r**2)), damping=lam,
                            violation=float(np.max(np.abs(viol), initial=0.0)), x=x.copy()))
        history[-1]["damping"] = lam_ok

        return DLSResult(self.system.copy(), x, float(np.sum(r**2)), history, success, message,
                         multipliers)

    def report(self, names=None):
        """Share of each operand in the merit function and the state of the
        constraints, as text.

        Parameters
        ----------
        names : sequence of str, optional
            Names of the operands (default: operand 1, 2, ...).

        Returns
        -------
        str
        """

        names = names or ["operand {}".format(k + 1) for k in range(len(self.operands))]
        parts = [np.sum((w*np.atleast_1d(op(self.system)))**2)
                 for w, op in zip(self.weights, self.operands)]
        self.system.__dict__.pop("_operand_cache", None)
        total = max(sum(parts), 1e-300)
        lines = ["merit function {:.6g}".format(sum(parts))]
        for n, p in zip(names, parts):
            lines.append("  {:28s} {:12.5g}  {:6.1f} %".format(n, p, 100*p/total))
        for c in self.constraints:
            v = c.values(self.system)
            ok = np.all(np.abs(c.violation(v)) <= 1e-3*c.tolerance)
            lines.append("  {:28s} {}  [{}, {}]  {}".format(
                c.name, np.array2string(v, precision=5), c.lower, c.upper,
                "ok" if ok else "VIOLATED"))

        return "\n".join(lines)

    def snap_glasses(self, catalog=None):
        """Replace every model glass that is a variable by the nearest
        catalogue glass and drop its variables (and glass map constraints).

        Parameters
        ----------
        catalog : dict or glasses.Catalog, optional
            name -> (n_d, V_d), default `GLASSES`, or a catalogue of real
            glasses (`pyoptics.glasses`), whose dispersion formulas then
            replace the Abbe model.

        Returns
        -------
        dict
            surface -> name of the chosen glass.
        """

        surfaces = sorted({v.surface for v in self.variables
                           if isinstance(v, (GlassIndex, GlassAbbe))})
        chosen = {}
        for i in surfaces:
            name, g = nearest_glass(*_model_glass(self.system.surfaces[i]), catalog)
            self.system.surfaces[i].material = g
            self.system.surfaces[i].label = name
            chosen[i] = name
        self.variables = [v for v in self.variables if not isinstance(v, (GlassIndex, GlassAbbe))]
        self.constraints = [c for c in self.constraints
                            if not any(c.name == "glass_map{}".format(i) for i in surfaces)]
        self.system = self._solve(self.system)
        self._scale = np.array([v.typical(self.system) for v in self.variables])

        return chosen
