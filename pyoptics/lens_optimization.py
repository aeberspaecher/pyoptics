#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Damped least squares optimization of `raytrace.System` lenses.

Two optimizers:

- `DampedLeastSquares`, the classical lens design optimizer (see the section
  further down): `Variable` objects (`Curvature`, `Thickness`, `AirSpace`,
  `Conic`, `Aspheric`, the phase coefficients of diffractive surfaces
  `DiffractivePhase` with `diffractive()`, freeform terms `ZernikeTerm`
  with `zernike_terms()`, `Tilt`, `Decenter` and `ImageTilt` of off-axis
  systems (notebook 155, off-axis mirrors and nodal aberrations), model glasses
  `GlassIndex`/`GlassAbbe` with `glass()`), solves (`solve_paraxial_focus()`, `solve_entrance_pupil()`,
  `solve_curvature_for_efl()`), operands (the factories below plus
  `polychromatic_ray_errors()`, `spot_rms()`, `wavefront()`, `seidel()`,
  `chromatic()`, `focal_shift()`) and `Constraint` objects held with
  Lagrange multipliers; a small glass catalogue `GLASSES` with
  `nearest_glass()` and the glass map outline `GLASS_MAP`
  (`glass_map_constraint()`). Notebook 58 (lens optimization).
  Several configurations of one lens (zoom positions, object distances,
  wavelength bands) are optimized together with a `MultiConfiguration`
  in place of the system: `Shared` and `ConfigVariable` variables
  (`per_configuration()`), operands, solves and constraints in some or all
  configurations (`in_configs()`), overrides with `configure()`. Notebook
  153 (multi-configuration optimization).
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
- `ftheta()`: chief ray image height against f*theta (scan lenses),
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
        operand(system) returns the x and y errors of all rays / tolerance;
        operand.linearize(lin) also their analytic Jacobian
        (`ray_derivatives`).
    """

    px, py = _pupil_rings(rings, spokes)

    def operand(system):
        out = []
        for I, _ in _trace(system, _fields(system, fields), px, py):
            e = (I[1:, :2] - I[0, :2])/tolerance
            out.append(np.nan_to_num(e, nan=FAILED, posinf=FAILED, neginf=FAILED).ravel())

        return np.concatenate(out)

    def linearize(lin):
        from .ray_derivatives import clean

        K = len(px) + 1
        rd = lin.trace_fields(_fields(lin.system, fields), np.concatenate([[0.0], px]),
                              np.concatenate([[0.0], py]))
        r, J = [], []
        for j in range(len(fields)):
            sl = slice(j*K + 1, (j + 1)*K)
            r.append(((rd.I[sl, :2] - rd.I[j*K, :2])/tolerance).ravel())
            dI = (rd.dI[:, sl, :2] - rd.dI[:, j*K:j*K + 1, :2])/tolerance
            J.append(dI.reshape(len(dI), -1).T)

        return clean(np.concatenate(r), np.concatenate(J))

    operand.linearize = linearize

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


def ftheta(efl=None, fields=(0.3, 0.5, 0.7, 0.85, 1.0), tolerance=1e-3, aim=False):
    """F-theta law of a scan lens: chief ray image height minus f*theta
    (object at infinity, fields are scan angles).

    Parameters
    ----------
    efl : number, optional
        Focal length f of the law. Default: the paraxial focal length of
        the system in each evaluation (then only the shape of h(theta) is
        corrected, the scale follows the focal length).
    fields : sequence, optional
        Fields as fractions of the maximum field angle.
    tolerance : number, optional
        Tolerance in the length unit (default 1 um for mm).
    aim : bool, optional
        Use the real chief ray through the centre of the stop (see
        `distortion()`). Not needed when the stop is the first surface
        (a scanner in front of the lens).

    Returns
    -------
    operand : callable
    """

    def operand(system):
        th = np.array(_fields(system, fields))
        f = system.paraxial().efl if efl is None else efl
        if aim:
            h = _aimed_chief(system, list(th))[1][:, 1]
        else:
            h = np.array([_chief(system, t)[0][1] for t in th])

        return np.nan_to_num((h - f*th)/tolerance, nan=FAILED)

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
    configurations : sequence of dict or callable, optional
        Overrides of several configurations (see `configure()`: object
        distances, wavelengths, fixed air spaces of a zoom...). The
        residuals of all operands in all configurations make the merit
        function; the variables are shared. Thicknesses that a
        configuration overrides are no default variables.
    configuration_weights : sequence, optional
        Weight of each configuration (default 1).
    """

    def __init__(self, system, operands, curvatures=None, thicknesses=None, object_distance=False,
                 length=None, solve=-1, object_na=None, aspheres=(), asphere_orders=(4, 6),
                 configurations=None, configuration_weights=None):
        self.system = system.copy()
        self.operands = list(operands)
        self.configurations = None if configurations is None else list(configurations)
        n_conf = 1 if configurations is None else len(self.configurations)
        self.configuration_weights = (np.ones(n_conf) if configuration_weights is None
                                      else np.asarray(configuration_weights, float))
        S = self.system.surfaces
        self.solve = solve % len(S)
        overridden = {k % len(S) for o in (self.configurations or ()) if isinstance(o, dict)
                      for k in o if isinstance(k, (int, np.integer))}
        if curvatures is None:
            n = self.system.n
            curvatures = [i for i, s in enumerate(S)
                          if not isinstance(s, (IdealLens, CoordinateBreak))
                          and (s.mirror or n[i] != n[i + 1])]
        if thicknesses is None:
            n = self.system.n
            thicknesses = [i for i in range(len(S))
                           if self.system._is_gap(n[i + 1]) and i != self.solve
                           and i not in overridden]
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
        out = []
        with np.errstate(all="ignore"):
            for w, s in zip(self.configuration_weights, self.configuration_systems(system)):
                out += [w*np.atleast_1d(op(s)) for op in self.operands]

        return np.concatenate(out)

    def configuration_systems(self, system=None):
        """The configurations of a system (default: the current one), with
        the solves of the optimizer applied in each.

        Parameters
        ----------
        system : raytrace.System, optional

        Returns
        -------
        list of raytrace.System
            [system] without configurations.
        """

        system = self.system if system is None else system
        if self.configurations is None:
            return [system]

        out = []
        for o in self.configurations:
            s = self._apply(configure(system, o))
            if isinstance(o, dict) and "max_field" in o:
                s.max_field = o["max_field"]*self.field_scale
            out.append(s)

        return out

    def jacobian(self, x=None):
        """Jacobian of `residuals()`: analytic derivatives of the real ray
        trace (`ray_derivatives`) for the operands that support them,
        central differences for the others.

        Parameters
        ----------
        x : array, optional
            Variables (default: the current ones).

        Returns
        -------
        (residuals, variables) array
        """

        from .ray_derivatives import Linearization

        x = self.x if x is None else np.asarray(x, float)
        steps = 1e-5*self._scales()
        n_conf = 1 if self.configurations is None else len(self.configurations)
        out = []
        for c, w in zip(range(n_conf), self.configuration_weights):
            lin = Linearization(lambda xx, c=c: self.configuration_systems(self.system_at(xx))[c],
                                x, steps)
            out += [w*lin.jacobian(op)[1] for op in self.operands]

        return np.concatenate(out)

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
            jac="analytic" uses `jacobian()` (analytic derivatives of the
            real ray trace).

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
            if options.get("jac") == "analytic":
                options["jac"] = self.jacobian
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
# when the operands have small tolerances). J^T J is never formed: the step
# is the least squares solution of the stacked system
#
#     [sqrt(lambda D)]        [ 0]
#     [J             ] dx  =  [-r],
#
# as in MINPACK (Moré 1978), whose condition number is that of J and not its
# square (the damping rows on top, which keeps the step accurate for a large
# lambda). The columns are first scaled by powers of two to norms near 1
# (LAPACK's dgeequb), which is exact.
# A step that lowers the merit function (plus a penalty on the constraint
# violation) is taken and lambda shrinks, else lambda grows and the step is
# solved again (Nielsen's damping update).


def _pow2_scales(norms):
    """Powers of two 2**-e with norms = m 2**e, 0.5 <= m < 1 (as LAPACK's
    dgeequb): scaling by them is exact. Zero or non-finite norms get 1."""

    norms = np.asarray(norms, float)
    e = np.frexp(np.where(np.isfinite(norms) & (norms > 0), norms, 1.0))[1]

    return np.ldexp(1.0, -np.clip(e, -1000, 1000))


def _lstsq(M, rhs):
    """Least squares solution (SVD, minimum norm if M is rank deficient)."""

    return np.linalg.lstsq(M, rhs, rcond=None)[0]


def _uncached(system):
    """Copy of a system (or of a `MultiConfiguration`) without the operand
    cache."""

    other = system.copy()
    _drop_cache(other)

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


class Tilt(Variable):
    """Tilt angle (radians) of a surface about its local x, y or z axis
    (`raytrace.Surface.tilt`), e.g. of the mirrors of an off-axis system
    (notebook 155, off-axis mirrors and nodal aberrations)."""

    def __init__(self, surface, axis=0, bounds=(None, None)):
        super().__init__(bounds)
        self.surface, self.axis = surface, int(axis)
        self.name = "tilt{}_{}".format("xyz"[self.axis], surface)

    def get(self, system):
        return system.surfaces[self.surface].tilt[self.axis]

    def set(self, system, value):
        t = list(system.surfaces[self.surface].tilt)
        t[self.axis] = float(value)
        system.surfaces[self.surface].tilt = tuple(t)

    def typical(self, system):
        return 1e-2


class Decenter(Variable):
    """Decentre of a surface in x (axis 0) or y (axis 1)
    (`raytrace.Surface.decenter`)."""

    def __init__(self, surface, axis=1, bounds=(None, None)):
        super().__init__(bounds)
        self.surface, self.axis = surface, int(axis)
        self.name = "dec{}_{}".format("xy"[self.axis], surface)

    def get(self, system):
        return system.surfaces[self.surface].decenter[self.axis]

    def set(self, system, value):
        d = list(system.surfaces[self.surface].decenter)
        d[self.axis] = float(value)
        system.surfaces[self.surface].decenter = tuple(d)

    def typical(self, system):
        return 1e-2*_length_scale(system)


class ImageTilt(Variable):
    """Tilt (radians) of the image plane about its x, y or z axis
    (`raytrace.System.image_tilt`)."""

    def __init__(self, axis=0, bounds=(None, None)):
        super().__init__(bounds)
        self.axis = int(axis)
        self.name = "image_tilt{}".format("xyz"[self.axis])

    def get(self, system):
        return system.image_tilt[self.axis]

    def set(self, system, value):
        t = list(system.image_tilt)
        t[self.axis] = float(value)
        system.image_tilt = tuple(t)

    def typical(self, system):
        return 1e-2


class ZernikeTerm(Variable):
    """Coefficient c_j of the Zernike departure of a
    `raytrace.ZernikeSurface` (length units): freeform terms."""

    def __init__(self, surface, j, bounds=(None, None)):
        super().__init__(bounds)
        self.surface, self.j = surface, int(j)
        self.name = "Z{}_{}".format(self.j, surface)

    def get(self, system):
        return dict(system.surfaces[self.surface].zernike).get(self.j, 0.0)

    def set(self, system, value):
        system.surfaces[self.surface].set_zernike({self.j: value})

    def typical(self, system):
        return 100*system.wavelength


def zernike_terms(system, surfaces, terms, bounds=(None, None)):
    """`ZernikeTerm` variables for several terms on several surfaces.

    Parameters
    ----------
    system : raytrace.System
        Only used to check that the surfaces are Zernike surfaces.
    surfaces : sequence of int
    terms : sequence of int
        Zernike indices (in the ordering of each surface).
    bounds : optional

    Returns
    -------
    list of ZernikeTerm
    """

    from .raytrace import ZernikeSurface

    for i in surfaces:
        if not isinstance(system.surfaces[i], ZernikeSurface):
            raise TypeError("Surface {} is not a ZernikeSurface.".format(i))

    return [ZernikeTerm(i, j, bounds) for i in surfaces for j in terms]


class DiffractivePhase(Variable):
    """Phase coefficient A_k of a `raytrace.DiffractiveSurface` (of
    rho**(2k)), as the phase A_k (rim/norm_radius)**(2k) in radians at the
    radius `rim`. k = 1 is the diffractive power, k = 2, 3, ... act like
    aspheric terms (scaled with the wavelength).

    Parameters
    ----------
    surface : int
        Index of the diffractive surface.
    k : int
        Term of the phase polynomial, 1 for rho**2.
    rim : number
        Radius at which the variable is the phase (e.g. the semi diameter).
    bounds : (lower, upper), optional
    """

    def __init__(self, surface, k, rim, bounds=(None, None)):
        super().__init__(bounds)
        if k < 1:
            raise ValueError("k counts from 1 (the rho**2 term).")
        self.surface, self.k, self.rim = surface, int(k), float(rim)
        self.name = "phi{}_{}".format(2*self.k, surface)

    def _coefficients(self, system):
        a = list(system.surfaces[self.surface].phase)

        return a + [0.0]*(self.k - len(a))

    def _scale(self, system):
        return (self.rim/system.surfaces[self.surface].norm_radius)**(2*self.k)

    def get(self, system):
        return self._coefficients(system)[self.k - 1]*self._scale(system)

    def set(self, system, value):
        a = self._coefficients(system)
        a[self.k - 1] = value/self._scale(system)
        system.surfaces[self.surface].phase = tuple(a)

    def typical(self, system):
        return 20*2*np.pi


def diffractive(surface, rim, terms=(1, 2), bounds=(None, None)):
    """Phase variables of a diffractive surface: [DiffractivePhase(surface,
    k, rim) for k in terms] (default: power and the rho**4 term).

    Parameters
    ----------
    surface : int
    rim : number
        Radius at which the variables are phases in radians.
    terms : sequence of int, optional
    bounds : (lower, upper), optional

    Returns
    -------
    list of Variable
    """

    return [DiffractivePhase(surface, k, rim, bounds) for k in terms]


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

def solve_paraxial_focus(offset=0.0, surface=None):
    """Solve: image plane in the paraxial focus (marginal ray height zero),
    plus offset (a defocus).

    Parameters
    ----------
    offset : number, optional
        Defocus.
    surface : int, optional
        Thickness that is solved. Default: the last one (the image
        distance). Another surface (e.g. the air space in front of a cover
        glass or a window, with the image distance behind it fixed) moves
        everything behind it, so that the focus lands on the fixed image
        plane; it has to be followed by plane parallel plates only.

    Returns
    -------
    callable
    """

    def solve(system):
        S = system.surfaces
        if surface is None:
            S[-1].thickness = 0.0
            S[-1].thickness = system.paraxial().image_distance + offset
            return
        for _ in range(2):
            S[surface].thickness += (system.paraxial().image_distance + offset
                                     - S[-1].thickness)*system.sigma[surface + 1]*system.sigma[-1]

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
        With operand.linearize(lin) for the analytic Jacobian
        (`ray_derivatives`).
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

    def linearize(lin):
        from .ray_derivatives import clean

        K = len(px1)
        r, J = [], []
        for w in ([None] if wavelengths is None else wavelengths):
            lw = lin if w is None else lin.mapped(lambda s, w=w: _at_wavelength(s, w))
            s = lw.system
            rd = lw.trace_fields(_fields(s, fields), px1, py1)
            for j in range(len(fields)):
                W, dW = lw.opd(rd, j*K, K)
                r.append(W[1:]/s.wavelength/tolerance)
                J.append(dW[:, 1:].T/s.wavelength/tolerance)

        return clean(np.concatenate(r), np.concatenate(J))

    operand.linearize = linearize

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
    aperture and field, times margin, the larger of the two surfaces; a
    system that is optimized at a fraction of its aperture and field
    carries the fractions as ``system._stage = (na_fraction,
    field_fraction)``, as in `LensOptimizer`, and the rim is scaled back to
    the full beam).
    Beyond the semi aperture of a conic the missing sag counts as negative
    thickness, so that the constraint keeps a gradient."""

    def f(system):
        S = system.surfaces
        a, b = S[surface], S[surface + 1]
        if height is None:
            p = system.paraxial()
            na_scale, field_scale = system.__dict__.get("_stage", (1.0, 1.0))
            rim = margin*(np.abs(p.marginal[:, 0])/na_scale + np.abs(p.chief[:, 0])/field_scale)
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


# ---------------------------------------------------------------------------
# Several configurations of one lens
# ---------------------------------------------------------------------------
#
# A zoom lens is one prescription at several positions of its groups; a
# camera lens focuses from infinity to a close object; a lens may have to
# work in two wavelength bands with a refocus in between, or at several
# temperatures. Classical programs call these configurations (Zemax's
# multi-configuration editor, the "zoom positions" of CODE V): one merit
# function over copies of the system that share most variables and differ
# in a few overrides (the air spaces of the moving groups, the object
# distance, the wavelengths, the aperture). `MultiConfiguration` holds the
# copies; `DampedLeastSquares` accepts it in place of a system. Plain
# variables then act on every configuration at once (`Shared`), plain
# operands, solves and constraints are evaluated in every configuration and
# concatenated (`in_configs()`); `ConfigVariable` and `per_configuration()`
# vary a parameter in one configuration only (a zoom spacing), and
# `in_configs(function, configs)` restricts an operand, a solve or a
# constraint to some of them.


def _drop_cache(system):
    """Forget the cached traces of a system (or of every configuration)."""

    system.__dict__.pop("_operand_cache", None)
    for s in getattr(system, "configs", ()):
        s.__dict__.pop("_operand_cache", None)


def configure(system, overrides):
    """Copy of a system with the overrides of one configuration.

    Parameters
    ----------
    system : raytrace.System
    overrides : dict or callable
        A callable changes the copy in place (anything, e.g. the indices
        at another temperature). A dict may hold:

        - int i: the thickness behind surface i (-1: the image distance),
        - "object_distance", "wavelength", "max_field" (field angle or
          object height), "n_object",
        - "entrance_pupil_radius": the semi diameter of the stop for this
          paraxial entrance pupil radius,
        - "stop_semi_diameter",
        - "surfaces": {i: {attribute: value}} for anything else on a
          surface (material, conic, semi_diameter, tilt...),
        - "apply": a callable as above, called last.

    Returns
    -------
    raytrace.System
    """

    s = _uncached(system)
    if callable(overrides):
        overrides(s)
        return s
    for key, value in overrides.items():
        if isinstance(key, (int, np.integer)):
            s.surfaces[key].thickness = value
        elif key in ("object_distance", "wavelength", "max_field", "n_object"):
            setattr(s, key, value)
        elif key == "surfaces":
            for i, attributes in value.items():
                for name, v in attributes.items():
                    setattr(s.surfaces[i], name, v)
        elif key not in ("entrance_pupil_radius", "stop_semi_diameter", "apply"):
            raise ValueError("Unknown override {!r}.".format(key))
    if "stop_semi_diameter" in overrides:
        s.surfaces[s.stop_index].semi_diameter = overrides["stop_semi_diameter"]
    if "entrance_pupil_radius" in overrides:
        solve_entrance_pupil(overrides["entrance_pupil_radius"])(s)
    if "apply" in overrides:
        overrides["apply"](s)

    return s


class MultiConfiguration:
    """Several configurations of one lens (zoom positions, object
    distances, wavelength bands, temperatures), optimized together by
    `DampedLeastSquares`.

    Parameters
    ----------
    systems : sequence of raytrace.System
        One system per configuration (copied). They should have the same
        surfaces; what differs between them is up to the caller (see
        `from_overrides()` and `from_zoom()`).
    weights : sequence, optional
        Weight of each configuration in the merit function (multiplies the
        residuals of the operands that are evaluated in it; default 1).
    names : sequence of str, optional
        Names of the configurations (for reports).
    """

    def __init__(self, systems, weights=None, names=None):
        self.configs = [_uncached(s) for s in systems]
        if not self.configs:
            raise ValueError("Give at least one configuration.")
        n = len(self.configs)
        self.weights = np.ones(n) if weights is None else np.asarray(weights, float)
        if len(self.weights) != n:
            raise ValueError("One weight per configuration.")
        self.names = (["config {}".format(k + 1) for k in range(n)] if names is None
                      else list(names))

    @classmethod
    def from_overrides(cls, system, overrides, weights=None, names=None):
        """Configurations of one system with overrides (see `configure()`).

        Parameters
        ----------
        system : raytrace.System
        overrides : sequence of dict or callable
            One entry per configuration ({} for the system as it is).
        weights, names : optional
            As for the class.

        Returns
        -------
        MultiConfiguration
        """

        return cls([configure(system, o) for o in overrides], weights, names)

    @classmethod
    def from_zoom(cls, zoom, positions=None, field_angle=None, weights=None):
        """The positions of a `zoom.ZoomLens` as configurations.

        Parameters
        ----------
        zoom : zoom.ZoomLens
        positions : sequence, optional
            Tabulated positions (int) or focal lengths (float, interpolated
            spacings); default: all tabulated positions.
        field_angle : number or sequence, optional
            Maximum field angle per position (default: as in the zoom).
        weights : sequence, optional

        Returns
        -------
        MultiConfiguration
            With the attribute ``zoom_gaps``, the surfaces whose
            thicknesses change with the zoom position.
        """

        positions = list(range(len(zoom))) if positions is None else list(positions)
        angles = np.broadcast_to(np.asarray(field_angle if field_angle is not None else np.nan,
                                            float), (len(positions),))
        systems = [zoom.system(p, field_angle=None if np.isnan(a) else a)
                   for p, a in zip(positions, angles)]
        names = ["f = {:.4g}".format(s.paraxial().efl) for s in systems]
        mc = cls(systems, weights, names)
        mc.zoom_gaps = sorted(zoom.variable)

        return mc

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, k):
        return self.configs[k]

    def __iter__(self):
        return iter(self.configs)

    def copy(self):
        other = copy.copy(self)
        other.__dict__.pop("_operand_cache", None)
        other.configs = [_uncached(s) for s in self.configs]
        other.weights = self.weights.copy()
        other.names = list(self.names)

        return other

    def __repr__(self):
        return "MultiConfiguration({})".format(", ".join(self.names))

    def thickness_table(self, surfaces):
        """Thicknesses behind surfaces in every configuration.

        Parameters
        ----------
        surfaces : sequence of int

        Returns
        -------
        dict
            {surface: array of thicknesses, one per configuration}.
        """

        return {i: np.array([s.surfaces[i].thickness for s in self.configs]) for i in surfaces}

    def to_zoom(self, zoom, groups=None):
        """A `zoom.ZoomLens` with the prescription of the first
        configuration and the spacing table of all of them.

        Parameters
        ----------
        zoom : zoom.ZoomLens
            The zoom the configurations came from (its variable surfaces,
            F-numbers and groups are kept; the focal lengths are those of the
            configurations now).
        groups : sequence, optional
            Default: the groups of zoom.

        Returns
        -------
        zoom.ZoomLens
        """

        from .zoom import ZoomLens

        efl = [s.paraxial().efl for s in self.configs]

        return ZoomLens(self.configs[0], self.thickness_table(sorted(zoom.variable)), efl,
                        zoom.f_numbers, zoom.groups if groups is None else groups)


class Shared(Variable):
    """A variable with the same value in every configuration of a
    `MultiConfiguration` (a curvature, a glass, a fixed air space). Plain
    variables given to `DampedLeastSquares` with a `MultiConfiguration` are
    wrapped in it.

    Parameters
    ----------
    variable : Variable
    """

    def __init__(self, variable):
        super().__init__(variable.bounds)
        self.variable = variable
        self.name = variable.name

    def get(self, system):
        return self.variable.get(system.configs[0])

    def set(self, system, value):
        for s in system.configs:
            self.variable.set(s, value)

    def typical(self, system):
        return self.variable.typical(system.configs[0])

    def perturbed(self, system, h):
        other = copy.copy(system)
        other.__dict__.pop("_operand_cache", None)
        other.configs = [self.variable.perturbed(s, h) for s in system.configs]
        _drop_cache(other)

        return other


class ConfigVariable(Variable):
    """A variable of one configuration only (the air spaces of a zoom at
    one position, the focus of one object distance).

    Parameters
    ----------
    variable : Variable
    config : int
        Index of the configuration.
    """

    def __init__(self, variable, config):
        super().__init__(variable.bounds)
        self.variable = variable
        self.config = config
        self.name = "{}@{}".format(variable.name, config)

    def get(self, system):
        return self.variable.get(system.configs[self.config])

    def set(self, system, value):
        self.variable.set(system.configs[self.config], value)

    def typical(self, system):
        return self.variable.typical(system.configs[self.config])

    def perturbed(self, system, h):
        other = copy.copy(system)
        other.__dict__.pop("_operand_cache", None)
        k = self.config
        other.configs = [self.variable.perturbed(s, h) if j == k else _uncached(s)
                         for j, s in enumerate(system.configs)]

        return other


def per_configuration(variable, configs):
    """One copy of a variable per configuration (`ConfigVariable`).

    Parameters
    ----------
    variable : Variable
    configs : int or sequence of int
        Number of configurations (all of them) or their indices.

    Returns
    -------
    list of ConfigVariable
    """

    configs = range(configs) if isinstance(configs, (int, np.integer)) else configs

    return [ConfigVariable(copy.copy(variable), k) for k in configs]


def _unwrap(variable):
    while isinstance(variable, (Shared, ConfigVariable)):
        variable = variable.variable

    return variable


class in_configs:
    """An operand, solve or constraint function evaluated in some
    configurations of a `MultiConfiguration`, the results concatenated.

    Parameters
    ----------
    function : callable
        function(system) for a single configuration.
    configs : sequence of int, optional
        Indices of the configurations (default: all).
    weighted : bool, optional
        Multiply the results by the weights of the configurations (as for
        operands; `DampedLeastSquares` sets it: True for operands, False for
        constraints).
    """

    def __init__(self, function, configs=None, weighted=True):
        self.function = function
        self.configs = None if configs is None else [int(k) for k in np.atleast_1d(configs)]
        self.weighted = weighted
        self.__name__ = getattr(function, "__name__", "function")

    def which(self, system):
        """Indices of the configurations it is evaluated in."""

        return list(range(len(system.configs))) if self.configs is None else self.configs

    def per_config(self, system):
        """{config: result} (results of solves are None)."""

        out = {}
        for k in self.which(system):
            v = self.function(system.configs[k])
            if v is not None and self.weighted:
                v = system.weights[k]*np.atleast_1d(np.asarray(v, float))
            out[k] = v

        return out

    def __call__(self, system):
        out = [v for v in self.per_config(system).values() if v is not None]
        if not out:
            return None

        return np.concatenate([np.atleast_1d(np.asarray(v, float)) for v in out])

    def with_weighting(self, weighted):
        other = copy.copy(self)
        other.weighted = weighted

        return other


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
    system : raytrace.System or MultiConfiguration
        Starting system (copied). With a `MultiConfiguration`, plain
        variables are `Shared` by all configurations, and plain operands,
        solves and constraints are evaluated in every configuration
        (`in_configs()`); per configuration variables are `ConfigVariable`
        objects (`per_configuration()`).
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
    jac : {"2-point", "analytic"}, optional
        Jacobian of the residuals: forward differences, or analytic
        derivatives of the real ray trace (`ray_derivatives`, differential
        ray tracing) for the operands that support them
        (`off_axis.field_wavefront()`, `off_axis.zernike_wavefront()`,
        `wavefront()`, `ray_errors()` and
        operands with a ``linearize`` method), with central differences for
        the other operands, the constraints and the variables that the
        differential trace does not cover. Solves are differentiated with
        the system (their paraxial effect by central differences). Not
        with a `MultiConfiguration` (forward differences there).
    """

    def __init__(self, system, variables, operands, constraints=(), solves=(), weights=None,
                 damping=1e-2, jac="2-point"):
        self.multi = isinstance(system, MultiConfiguration)
        self.variables = [self._variable(v) for v in variables]
        self.operands = [self._function(op, True) for op in operands]
        self.constraints = [self._constraint(c) for c in constraints]
        self.solves = [self._function(s, False) for s in solves]
        self.weights = np.ones(len(self.operands)) if weights is None else np.asarray(weights, float)
        self.damping = damping
        if jac not in ("2-point", "analytic"):
            raise ValueError("jac is '2-point' or 'analytic'.")
        self.jac = jac
        self.system = self._solve(_uncached(system))
        self._scale = np.array([v.typical(self.system) for v in self.variables])

    def _variable(self, v):
        if self.multi and not isinstance(v, (Shared, ConfigVariable)):
            return Shared(v)
        return v

    def _function(self, f, weighted):
        if not self.multi:
            return f
        if isinstance(f, in_configs):
            return f.with_weighting(weighted)
        return in_configs(f, weighted=weighted)

    def _constraint(self, c):
        if not self.multi:
            return c
        other = copy.copy(c)
        other.function = self._function(c.function, False)

        return other

    def _solve(self, system):
        for s in self.solves:
            s(system)
        _drop_cache(system)

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
        _drop_cache(system)

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

    def merit_by_configuration(self, system=None):
        """Share of each configuration in the merit function (with a
        `MultiConfiguration`).

        Parameters
        ----------
        system : MultiConfiguration, optional
            Default: the current one.

        Returns
        -------
        array
            Sum of the squared weighted residuals per configuration.
        """

        system = self.system if system is None else system
        if not self.multi:
            return np.array([self.merit(system)])
        out = np.zeros(len(system.configs))
        with np.errstate(all="ignore"):
            for w, op in zip(self.weights, self.operands):
                for k, v in op.per_config(system).items():
                    out[k] += np.sum((w*np.atleast_1d(v))**2)
        _drop_cache(system)

        return out

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
        respect to the variables (forward differences, or analytic with
        jac="analytic"; each column is the sensitivity of the operands to
        one variable).

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
        if self.jac == "analytic" and not self.multi and system is self.system:
            return self._analytic_jacobian()
        r = self.residuals(system)
        g = self.constraint_values(system)[0]
        J, A = self._columns(self.variables, self._scale, system, r, g)

        return J, A, r, g

    def linearization(self):
        """`ray_derivatives.Linearization` of the current system with respect
        to the variables (solves included).

        Returns
        -------
        ray_derivatives.Linearization
        """

        from .ray_derivatives import Linearization

        return Linearization(self.system_at, self.x, 1e-5*self._scale, system=self.system)

    def _analytic_jacobian(self):
        lin = self.linearization()
        rs, Js = [], []
        for w, op in zip(self.weights, self.operands):
            r, J = lin.jacobian(op)
            rs.append(w*r)
            Js.append(w*J)
        _drop_cache(self.system)
        r = np.concatenate(rs) if rs else np.zeros(0)
        J = np.concatenate(Js) if Js else np.zeros((0, len(self.variables)))
        g = self.constraint_values()[0]
        A = (lin.finite_difference(lambda s: self.constraint_values(s)[0]) if len(g)
             else np.zeros((0, len(self.variables))))

        return J, A, r, g

    def _columns(self, variables, scales, system, r, g):
        """Forward difference columns of the residuals and constraint
        values for variables."""

        J = np.zeros((len(r), len(variables)))
        A = np.zeros((len(g), len(variables)))
        for k, v in enumerate(variables):
            h = 1e-6*scales[k]
            lo, hi = v.bounds
            x0 = v.get(system)
            if hi is not None and x0 + h > hi:
                h = -h
            s = self._solve(v.perturbed(system, h))
            h = v.get(s) - x0 if v.get(s) != x0 else h  # the step actually taken
            J[:, k] = (self.residuals(s) - r)/h
            if len(g):
                A[:, k] = (self.constraint_values(s)[0] - g)/h

        return J, A

    def predicted_gain(self, candidates, damping=None, jacobian=None):
        """Predicted decrease of the merit function if groups of variables
        were added, from the linearized operands (a ranking of candidates,
        e.g. of the surfaces for an asphere).

        For each group, one damped least squares step with the current
        variables and the group is solved, with the constraints that are
        active (violated or at their bound) held at their current values to
        first order, and its
        predicted merit function is compared with that of the same step with
        the current variables alone. At a minimum of the current variables
        the gain is that part of the residuals that the new columns of the
        Jacobian reach and the old ones do not (it is zero for a variable
        that only repeats what the others do).

        Parameters
        ----------
        candidates : dict or sequence
            name -> sequence of Variable (or a sequence of sequences, named
            by their index).
        damping : number, optional
            lambda of the steps (default: the initial damping).
        jacobian : tuple, optional
            (J, A, r, g) of the current variables from `jacobian()`, to
            reuse.

        Returns
        -------
        gains : dict
            name -> predicted decrease of the merit function.
        base : float
            Predicted merit function of the step with the current
            variables alone.
        """

        if not isinstance(candidates, dict):
            candidates = dict(enumerate(candidates))
        candidates = {k: [self._variable(v) for v in group] for k, group in candidates.items()}
        lam = self.damping if damping is None else damping
        J, A, r, g = self.jacobian() if jacobian is None else jacobian
        self._lo, self._hi, tol, _ = self._constraint_info()
        active = (self._lo == self._hi) | (g <= self._lo + 1e-3*tol) | (g >= self._hi - 1e-3*tol)
        target = g  # held where they are: the gain is not mixed with the cost of restoring them
        scale = self._scale

        def predicted(Jx, Ax, sx):
            self._scale = sx
            try:
                dx = self._step(Jx, Ax, r, target, g, lam, active, np.zeros(len(sx), bool))[0]
            finally:
                self._scale = scale
            return float(np.sum((r + Jx @ dx)**2))

        base = predicted(J, A, scale)
        gains = {}
        for name, group in candidates.items():
            group = list(group)
            sc = np.array([v.typical(self.system) for v in group])
            Jc, Ac = self._columns(group, sc, self.system, r, g)
            gains[name] = base - predicted(np.hstack([J, Jc]), np.hstack([A, Ac]),
                                           np.concatenate([scale, sc]))

        return gains, base

    def _step(self, J, A, r, target, g, lam, active, bounds):
        """Damped least squares step in scaled variables with the active
        constraints held to first order (a step that overflows for a huge
        damping is rejected by the caller)."""

        with np.errstate(over="ignore", invalid="ignore"):
            return self._step_unchecked(J, A, r, target, g, lam, active, bounds)

    def _step_unchecked(self, J, A, r, target, g, lam, active, bounds):

        D = self._scale
        n = len(D)
        Js = J*D
        # Marquardt's damping diag(J^T J), from the column norms (J^T J
        # itself is never formed: it squares the condition number)
        norms = np.sqrt(np.sum(Js**2, axis=0))
        dd = np.maximum(norms**2, 1e-9*max(np.max(norms**2, initial=0.0), 1e-30) + 1e-30)
        free = np.ones(n, bool)
        free[bounds] = False
        n_active = int(np.sum(active))
        if not np.isfinite(lam):
            return np.zeros(n), np.zeros(n_active)  # the limit of infinite damping
        # exact equilibration: powers of two that bring the column norms
        # near 1 (they change no bit of the mantissas)
        E = _pow2_scales(norms[free])
        Jt = Js[:, free]*E
        w = np.sqrt(lam)*(np.sqrt(dd[free])*E)  # damping rows, sqrt(lambda D)
        if not (np.all(np.isfinite(Jt)) and np.all(np.isfinite(r))):
            return np.full(n, np.nan), np.zeros(n_active)
        if n_active:
            At = (A*D)[np.ix_(active, free)]*E
            b = (target - g)[active]
            # null space method (better conditioned than the KKT matrix):
            # dt = dt_p + Z y with At dt_p = b and At Z = 0
            U, sv, Vt = np.linalg.svd(At)
            rank = int(np.sum(sv > 1e-10*sv[0])) if len(sv) and sv[0] > 0 else 0
            dt_p = Vt[:rank].T @ ((U[:, :rank].T @ b)/sv[:rank])
            Z = Vt[rank:].T
        else:
            dt_p, Z = np.zeros(len(E)), np.eye(len(E))
        dt = dt_p
        if Z.shape[1]:
            # min |r + Jt dt|^2 + |w dt|^2 over y, as the least squares
            # problem of the stacked matrix [diag(w); Jt] (Moré 1978); the
            # damping rows come first: they are the heavy rows for a large
            # lambda, and below the others the step would lose
            # sqrt(lambda) eps of its relative accuracy (Powell and Reid 1969)
            M = np.vstack([w[:, None]*Z, Jt @ Z])
            rhs = -np.concatenate([w*dt_p, r + Jt @ dt_p])
            dt = dt_p + Z @ _lstsq(M, rhs)
        if n_active:
            # J^T r + (J^T J + lambda D) dt + A^T mu = 0 (same mu in every scaling)
            grad = Jt.T @ (r + Jt @ dt) + w*(w*dt)
            mu = _lstsq(At.T, -grad)
        else:
            mu = np.zeros(0)
        dx = np.zeros(n)
        dx[free] = dt*E*D[free]

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
            restore = 1.0
            for _ in range(30):
                # active set: equalities, violated inequalities and those that
                # the step would violate; violated constraints are restored
                # by the fraction restore, which halves with every rejected
                # step (far from the constraints, the linearized restoring
                # step can be much too long, and the damping does not
                # shorten it)
                active = (self._lo == self._hi) | (g < self._lo) | (g > self._hi)
                target = np.clip(g, self._lo, self._hi)
                if restore < 1.0:
                    target = g + restore*(target - g)
                # variables that the step would take beyond a bound go to the
                # bound and stay there for this step, and the others are
                # solved again with that change (a clipped step would break
                # the linearized constraints)
                fixed = at_bound.copy()
                shift = np.zeros(len(x))
                for _ in range(len(x) + 1):
                    r_s, g_s = r + J @ shift, g + A @ shift
                    for _ in range(len(g) + 1):
                        dx, mu = self._step(J, A, r_s, target, g_s, lam, active, fixed)
                        dx = dx + shift
                        g_pred = g + A @ dx
                        below = ~active & (g_pred < self._lo - 1e-3*tol)
                        above = ~active & (g_pred > self._hi + 1e-3*tol)
                        if not np.any(below | above):
                            break
                        target = np.where(below, self._lo, np.where(above, self._hi, target))
                        active |= below | above
                    clipped = np.array([v.clip(xi + d) for v, xi, d in zip(self.variables, x, dx)])
                    hit = ~fixed & (clipped != x + dx)
                    if not np.any(hit):
                        break
                    fixed |= hit
                    shift[hit] = clipped[hit] - x[hit]
                x_new = np.array([v.clip(xi + d) for v, xi, d in zip(self.variables, x, dx)])
                try:
                    s_new = self.system_at(x_new)
                    phi_new = self.merit(s_new) + self._penalty(self.constraint_values(s_new)[1])
                except (ValueError, ArithmeticError, NotImplementedError, np.linalg.LinAlgError):
                    phi_new = np.inf  # the solves or the paraxial trace fail: a step too far
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
                restore *= 0.5
            if not accepted:
                success, message = True, "no step lowers the merit function (local minimum)"
                break
            gain = (phi - phi_new)/max(phi, 1e-300)
            self.system, x, phi = s_new, x_new, phi_new
            viol = self.constraint_values()[1]
            if gain < ftol and np.all(np.abs(viol) < 1e-3):
                success, message = True, "converged"
                break
        r = self.residuals()
        g, viol = self.constraint_values()
        history.append(dict(merit=float(np.sum(r**2)), damping=lam,
                            violation=float(np.max(np.abs(viol), initial=0.0)), x=x.copy()))
        history[-1]["damping"] = lam_ok
        if len(g) and history[:-1]:
            # multipliers of the constraints at their bounds at the end (with
            # the last Jacobian): the gradient of the merit function that
            # they balance, J^T r + A^T mu = 0
            active = ((self._lo == self._hi) | (g <= self._lo + 1e-3*tol)
                      | (g >= self._hi - 1e-3*tol))
            if np.any(active):
                D = self._scale
                mu = np.linalg.lstsq((A*D)[active].T, -(J*D).T @ r, rcond=None)[0]
                multipliers = dict(zip([n for n, a in zip(names, active) if a], mu))

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
        _drop_cache(self.system)
        total = max(sum(parts), 1e-300)
        lines = ["merit function {:.6g}".format(sum(parts))]
        for n, p in zip(names, parts):
            lines.append("  {:28s} {:12.5g}  {:6.1f} %".format(n, p, 100*p/total))
        if self.multi:
            for n, p in zip(self.system.names, self.merit_by_configuration()):
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

        surfaces = sorted({_unwrap(v).surface for v in self.variables
                           if isinstance(_unwrap(v), (GlassIndex, GlassAbbe))})
        systems = self.system.configs if self.multi else [self.system]
        chosen = {}
        for i in surfaces:
            name, g = nearest_glass(*_model_glass(systems[0].surfaces[i]), catalog)
            for s in systems:
                s.surfaces[i].material = g
                s.surfaces[i].label = name
            chosen[i] = name
        self.variables = [v for v in self.variables
                          if not isinstance(_unwrap(v), (GlassIndex, GlassAbbe))]
        self.constraints = [c for c in self.constraints
                            if not any(c.name == "glass_map{}".format(i) for i in surfaces)]
        self.system = self._solve(self.system)
        self._scale = np.array([v.typical(self.system) for v in self.variables])

        return chosen
