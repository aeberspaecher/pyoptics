#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Analytic derivatives of real ray traces with respect to the parameters of
a system (differential ray tracing), for exact Jacobians in lens
optimization.

A `Linearization` of a system belongs to a parametrization x -> system
(`DampedLeastSquares.system_at()` or `LensOptimizer.system_at()`, solves
included). For every parameter it first finds out what the parameter
changes, from the systems at x +- h e_k (no rays are traced for that, the
solves are paraxial):

- the shape of a surface (curvature, conic, even aspheric terms, Q-bfs and
  Q-con coefficients, Zernike terms, normalization radii...): the change of
  the sag and of its gradient at the points where the rays hit the
  surface; linear parameters (aspheric, Q and Zernike coefficients) give
  exact differences,
- the frames of the surfaces and of the image plane (thicknesses, air
  spaces, decentres and tilts, coordinate breaks, the image tilt),
- the refractive indices (model glasses),
- the semi diameter of the stop (the targets of ray aiming, e.g. by
  `solve_object_na()` or `solve_entrance_pupil()`), the object distance
  and the paraxial pupils (rays that are not aimed, the reference sphere
  of the OPD).

The rays are then traced once, carrying the derivatives of their points,
directions and optical paths along (forward mode). At a surface F(Q; p) =
Q_z - z(Q_x, Q_y; p) = 0 the intersection distance t of a ray P + t d
changes by

    dt = -(grad F . (dP + t dd) + dF/dp)/(grad F . d),

and the change of the normal (from the Hessian of the sag and the change
of its gradient) enters the derivative of the law of refraction or
reflection. Aimed rays are fixed by their stop coordinates, not by their
launch parameters: with the derivatives of the stop hit h with respect to
the parameters and to the two launch parameters (a, b), the implicit
function theorem gives d(a, b)/dp = -(dh/d(a, b))^-1 (dh/dp - d target/dp),
which is added at the stop. The derivatives agree with central
differences of complete traces to about 1e-7 (relative) and cost about one
trace with as many rays as there are parameters, without aiming them again.

Operands of `lens_optimization` with a method ``linearize(lin)`` returning
the residuals and their Jacobian (`off_axis.field_wavefront()`,
`lens_optimization.wavefront()`, `ray_errors()`,
`polychromatic_ray_errors()`, `spot_rms()`, `distortion()`, `ftheta()`,
`image_telecentricity()`, `object_telecentricity()`) are differentiated
analytically; paraxial operands (`seidel()`, `paraxial_target()`,
`clearances()`...) by `Linearization.derivative()`; other operands, parameters that change something the
differential trace does not cover (wavelength, field, surfaces other than
`raytrace.Surface`, Q-bfs, Q-con and Zernike surfaces and coordinate breaks)
and constraints get central differences of `Linearization.jacobian()`.
Rays are traced without vignetting (clip=False), as by the operands.

When analytic Jacobians pay off
-------------------------------
``jac="analytic"`` (`lens_optimization.DampedLeastSquares`,
`LensOptimizer.run()`) gives the same steps as differences, more
accurately; whether it is faster depends on the problem. A linearization
first builds the systems at x +- h for every variable (2V copies with the
solves, no rays), then traces the rays once with V derivatives each. A
forward difference Jacobian traces all operands V times. So:

- **Use it** for many variables (about 20 and more) and operands that trace
  many real rays and have a ``linearize`` method (`ray_errors()`,
  `wavefront()`, `polychromatic_ray_errors()`, `spot_rms()`,
  `off_axis.field_wavefront()`, the chief ray operands). Examples: the
  KrF lens of notebook 43 (Delano diagrams) with 96 variables and 308 residuals:
  1.3 s instead of 17 s per Jacobian (`LensOptimizer`), 1.5 s instead of
  12 s (`DampedLeastSquares` with 82 clearance constraints); the six mirror
  EUV projector: 0.85 s instead of 20 s.
- **Keep differences** ("2-point", the default) for small problems (fewer
  than about 10 variables: the 2V systems of the linearization cost as much
  as the V traces they save; notebooks 58 (lens optimization) and 154
  (global optimization and saddle points)), for merit functions without real
  rays (Seidel sums and other paraxial data: nothing to trace, the
  analytic mode takes central differences, twice the work), and when an
  expensive operand has no ``linearize`` (it is differentiated by central
  differences, twice the cost of forward differences; e.g. the
  clearance operand `off_axis.clearance_penalty()` in
  notebook 155 (off-axis mirrors and nodal aberrations): 16 s instead of
  4 s per Jacobian).
- Not with a `lens_optimization.MultiConfiguration` (forward differences).
- Parameters that the differential trace does not cover (see above) are
  differentiated column by column, so a few of them cost little.
- The more accurate Jacobian can change the path of an optimization: in
  ill conditioned problems the analytic run may go further (notebook 149
  (scan lenses): merit 494 instead of 561) or stop elsewhere. For a check of
  an operand's ``linearize``, compare with `Linearization.finite_difference()`.

Own operands use `Linearization.trace()`, which returns a `RayDerivatives`
with the image plane points and directions, the optical paths and the
object space directions and their derivatives, e.g. for a chief ray::

    def operand(system):
        b = system.trace(system.rays(f, 0.0, 0.0), clip=False)
        I, D = system._image_hits(b)
        return (I[0, :2] - target)/tol

    def linearize(lin):
        rd = lin.trace(f, 0.0, 0.0)
        return (rd.I[0, :2] - target)/tol, rd.dI[:, 0, :2].T/tol

    operand.linearize = linearize

References
----------
D. P. Feder, "Differentiation of ray-tracing equations with respect to
construction parameters of rotationally symmetric optics", J. Opt. Soc. Am.
58, 1494-1505 (1968).

O. Stavroudis, "A simpler derivation of the formulas for generalized ray
tracing", J. Opt. Soc. Am. 66, 1330-1333 (1976).

J. Nocedal and S. J. Wright, "Numerical Optimization", 2nd ed., Springer
(2006), chapter 8 (computing derivatives).
"""

from collections import namedtuple

import numpy as np

from . import raytrace as rt


FAILED = 1e3

RayDerivatives = namedtuple("RayDerivatives", ["I", "D", "opl", "d0", "P_last", "dI", "dD",
                                               "dopl", "dd0", "dP_last"])
RayDerivatives.__doc__ = """Rays traced by `Linearization.trace()` and their
derivatives with respect to the parameters of the linearization.

I, D : (N, 3) arrays
    Image plane points and directions in the frame of the image plane
    (as `raytrace.System._image_hits()`).
opl : (N,) array
    Optical path to the image plane.
d0 : (N, 3) array
    Object space directions (after aiming).
P_last : (N, 3) array
    Points on the last surface (global frame).
dI, dD, dd0, dP_last : (V, N, 3) arrays
    Their derivatives with respect to the V parameters.
dopl : (V, N) array
    Derivatives of the optical paths.
"""

# attributes of a surface that are not its shape:
_FRAME = {"thickness", "decenter", "tilt"}
_INDEX = {"n_after", "material", "index_offset"}
_IGNORED = {"label", "coating", "semi_diameter"}
_NOT_SHAPE = _FRAME | _INDEX | _IGNORED | {"mirror"}
# system attributes that the derivatives cover:
_SYSTEM_OK = {"object_distance", "image_decenter", "image_tilt", "background"}


def _shape_types():
    from .qaspheres import QbfsSurface, QconSurface

    return (rt.Surface, QbfsSurface, QconSurface, rt.ZernikeSurface)


def _differs(a, b):
    if a is b:
        return False
    try:
        eq = a == b
        if isinstance(eq, np.ndarray):
            return not bool(np.all(eq))
        return not bool(eq)
    except (ValueError, TypeError):
        return True


def _public(obj):
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _sag_gradient(s, x, y):
    """Sag z and gradient (z_x, z_y) of a surface at local points."""

    if isinstance(s, rt.ZernikeSurface):
        zx, zy = s._gradient(x, y)
        return s.sag_xy(x, y), zx, zy
    r2 = x**2 + y**2
    sr = s._slope_over_r(r2)

    return s.sag(r2), x*sr, y*sr


def _hessian(s, x, y, scale):
    """Second derivatives (z_xx, z_xy, z_yy) of the sag: rotationally
    symmetric part from a central difference of the slope over r in r**2,
    the Zernike departure from central differences of its analytic
    gradient."""

    r2 = x**2 + y**2
    h2 = 1e-5*scale**2
    sr = s._slope_over_r(r2)
    ds = (s._slope_over_r(r2 + h2) - s._slope_over_r(r2 - h2))/(2*h2)
    zxx, zxy, zyy = sr + 2*x**2*ds, 2*x*y*ds, sr + 2*y**2*ds
    if isinstance(s, rt.ZernikeSurface) and s.zernike:
        h = 1e-5*s.norm_radius
        gxp, gyp = s.departure_gradient(x + h, y)
        gxm, gym = s.departure_gradient(x - h, y)
        zxx = zxx + (gxp - gxm)/(2*h)
        zxy = zxy + (gyp - gym)/(2*h)
        gxp, gyp = s.departure_gradient(x, y + h)
        gxm, gym = s.departure_gradient(x, y - h)
        zyy = zyy + (gyp - gym)/(2*h)

    return zxx, zxy, zyy


def _vdot(A, b):
    """Dot product of tangents (m, N, 3) with vectors (N, 3) -> (m, N)."""

    return np.einsum("mnk,nk->mn", A, b)


class _Tangents:
    """Derivatives of the ray state for the active parameters (rows)."""

    def __init__(self, N):
        self.ids = np.zeros(0, int)
        self.P = np.zeros((0, N, 3))
        self.d = np.zeros((0, N, 3))
        self.opl = np.zeros((0, N))

    def activate(self, ids):
        new = np.setdiff1d(np.asarray(ids, int), self.ids)
        if len(new):
            N = self.P.shape[1]
            self.ids = np.concatenate([self.ids, new])
            self.P = np.concatenate([self.P, np.zeros((len(new), N, 3))])
            self.d = np.concatenate([self.d, np.zeros((len(new), N, 3))])
            self.opl = np.concatenate([self.opl, np.zeros((len(new), N))])

    def drop(self, ids):
        keep = ~np.isin(self.ids, ids)
        self.ids, self.P, self.d, self.opl = (self.ids[keep], self.P[keep], self.d[keep],
                                              self.opl[keep])

    def rows(self, ids):
        pos = {v: k for k, v in enumerate(self.ids)}

        return np.array([pos[v] for v in ids], int)


class Linearization:
    """Derivatives of ray traces of a system with respect to parameters x.

    Parameters
    ----------
    system_at : callable
        x -> raytrace.System (a new system, solves applied), e.g.
        `DampedLeastSquares.system_at`.
    x : array
        Values of the parameters.
    steps : array
        Steps h of the central differences that detect what each parameter
        changes, e.g. 1e-5 times the typical change of each variable
        (sags, frames and paraxial data are smooth and free of noise).
    system : raytrace.System, optional
        system_at(x), if already built.
    fd_steps : array, optional
        Steps of the central difference columns of operands and
        parameters without analytic derivatives (complete traces carry the
        noise of the ray aiming and the rounding of long optical paths, so
        larger steps; default 100 steps).
    """

    def __init__(self, system_at, x, steps, system=None, fd_steps=None):
        self.x = np.asarray(x, float)
        self.h = np.broadcast_to(np.asarray(steps, float), self.x.shape).copy()
        self.h_fd = (100*self.h if fd_steps is None
                     else np.broadcast_to(np.asarray(fd_steps, float), self.x.shape).copy())
        self._system_at = system_at
        self.system = system_at(self.x) if system is None else system
        self.plus, self.minus = self._perturbed(self.h)
        self._fd = None
        self._cache = {}
        self._analyse()

    def _perturbed(self, steps):
        plus, minus = [], []
        for k in range(len(self.x)):
            e = np.zeros(len(self.x))
            e[k] = steps[k]
            plus.append(self._system_at(self.x + e))
            minus.append(self._system_at(self.x - e))

        return plus, minus

    @classmethod
    def _from_systems(cls, parent, f):
        lin = cls.__new__(cls)
        lin.x, lin.h, lin.h_fd = parent.x, parent.h, parent.h_fd
        lin._system_at = lambda x: f(parent._system_at(x))
        lin.system = f(parent.system)
        lin.plus, lin.minus = [f(p) for p in parent.plus], [f(m) for m in parent.minus]
        lin._fd = None
        lin._cache = {}
        lin._analyse()

        return lin

    def mapped(self, f):
        """Linearization of f(system) (a function that returns a changed
        copy, e.g. the system at another wavelength), with the same
        parameters.

        Parameters
        ----------
        f : callable
            system -> system.

        Returns
        -------
        Linearization
        """

        return self._from_systems(self, f)

    @property
    def n_parameters(self):
        return len(self.x)

    # --- what the parameters change ---------------------------------------
    def derivative(self, f):
        """Central differences df/dx_k of a function of the system (without
        tracing rays: paraxial data, frames...).

        Parameters
        ----------
        f : callable
            system -> number or array.

        Returns
        -------
        (V, ...) array
        """

        return np.array([(np.asarray(f(p), float) - np.asarray(f(m), float))/(2*h)
                         for p, m, h in zip(self.plus, self.minus, self.h)])

    def _cached(self, key, f):
        if key not in self._cache:
            self._cache[key] = self.derivative(f)
        return self._cache[key]

    def _analyse(self):
        S = self.system
        V, n_s = self.n_parameters, len(S.surfaces)
        types = _shape_types()
        self.unsupported = np.zeros(V, bool)
        self.reasons = [[] for _ in range(V)]
        self.shape = [[] for _ in range(n_s)]  # parameters that change the shape of surface i
        sys_attrs = _public(S)
        for k, (p, m) in enumerate(zip(self.plus, self.minus)):
            if len(p.surfaces) != n_s or p.stop_index != S.stop_index:
                self.unsupported[k] = True
                self.reasons[k].append("surfaces")
                continue
            ap, am = _public(p), _public(m)
            for name in sys_attrs:
                if name in ("surfaces",) or name in _SYSTEM_OK:
                    continue
                if _differs(ap.get(name), am.get(name)):
                    self.unsupported[k] = True
                    self.reasons[k].append(name)
            if _differs(ap.get("object_distance"), am.get("object_distance")) \
                    and S.infinite_object:
                self.unsupported[k] = True
                self.reasons[k].append("object_distance")
            for i, (sp, sm) in enumerate(zip(p.surfaces, m.surfaces)):
                if type(sp) is not type(sm) or type(sp) is not type(S.surfaces[i]):
                    self.unsupported[k] = True
                    self.reasons[k].append("type of surface {}".format(i))
                    continue
                dp, dm = _public(sp), _public(sm)
                changed = {a for a in set(dp) | set(dm) if _differs(dp.get(a), dm.get(a))}
                if "mirror" in changed:
                    self.unsupported[k] = True
                    self.reasons[k].append("mirror {}".format(i))
                shape = changed - _NOT_SHAPE
                # the Zernike terms live in _terms (set_zernike):
                if isinstance(sp, rt.ZernikeSurface) and _differs(sp._terms, sm._terms):
                    shape.add("zernike")
                if shape and not isinstance(sp, rt.CoordinateBreak):
                    if type(sp) in types:
                        self.shape[i].append(k)
                    else:
                        self.unsupported[k] = True
                        self.reasons[k].append("shape of surface {} ({})".format(
                            i, type(sp).__name__))
        # frames, indices, stop, object:
        def frames(s):
            f, (o_i, R_i) = s.frames()
            o = np.array([o for o, _ in f] + [o_i])
            R = np.array([R for _, R in f] + [R_i])
            return np.concatenate([o.ravel(), R.ravel()])

        F = self.derivative(frames)
        n_f = n_s + 1
        self.do = F[:, :3*n_f].reshape(V, n_f, 3)
        self.dR = F[:, 3*n_f:].reshape(V, n_f, 3, 3)
        self.dn = self.derivative(lambda s: s.n)
        self.dsd = self.derivative(lambda s: s.surfaces[s.stop_index].semi_diameter)
        self.dobj = (np.zeros(V) if S.infinite_object
                     else self.derivative(lambda s: s.object_distance))
        types_ok = all(type(s) in types or isinstance(s, rt.CoordinateBreak) for s in S.surfaces)
        self.traceable = types_ok

    def _launch_effect(self):
        """Derivatives of the paraxial launch data (pupils, object slopes)."""

        S = self.system
        if S.infinite_object:
            return self._cached("pupils", lambda s: np.array(s._pupils()[:2]))

        return self._cached("slopes", lambda s: np.array(s._object_slopes()))

    # --- differential ray trace -------------------------------------------
    def trace(self, field, px=0.0, py=0.0, aim=None):
        """Trace rays (as `raytrace.System.rays()` and `trace()`, without
        vignetting) with their derivatives.

        Parameters
        ----------
        field : number or pair
            Field of the rays (see `raytrace.System.rays()`).
        px, py : numbers or arrays, optional
            Normalized pupil coordinates.
        aim : bool, optional
            Ray aiming (default: that of the system).

        Returns
        -------
        RayDerivatives
        """

        return self.trace_fields([field], px, py, aim)

    def trace_fields(self, fields, px=0.0, py=0.0, aim=None):
        """`trace()` of the same pupil points at several fields, in one
        differential trace (rays ordered field by field).

        Parameters
        ----------
        fields : sequence
            Fields (numbers or pairs).
        px, py : numbers or arrays, optional
            Normalized pupil coordinates.
        aim : bool, optional
            Ray aiming (default: that of the system).

        Returns
        -------
        RayDerivatives
            For F fields and K pupil points, the rays k + j*K belong to
            field j.
        """

        if not self.traceable:
            raise NotImplementedError("Surface types without analytic derivatives.")
        S = self.system
        aim = S.ray_aiming if aim is None else aim
        px_, py_ = np.broadcast_arrays(np.atleast_1d(np.asarray(px, float)),
                                       np.atleast_1d(np.asarray(py, float)))
        px_, py_ = px_.ravel(), py_.ravel()
        P, d, opl, fxy, pp = [], [], [], [], []
        for fl in fields:
            fl = tuple(fl) if np.ndim(fl) else fl
            r = S.rays(fl, px_, py_, aim=aim)
            P.append(np.reshape(r.points, (-1, 3)))
            d.append(np.reshape(r.directions, (-1, 3)))
            opl.append(np.reshape(r.opl, -1))
            fxy.append(np.tile(rt._field_xy(fl), (len(px_), 1)))
            pp.append(np.stack([px_, py_], axis=-1))

        return self._trace(np.concatenate(P), np.concatenate(d), np.concatenate(opl),
                           np.concatenate(fxy), np.concatenate(pp), aim)

    def _trace(self, P, d, opl, fxy, pp, aim):
        S = self.system
        V = self.n_parameters
        N = len(P)
        n = S.n
        n0 = S.n_object
        ex, ey, ez = np.eye(3)
        T = _Tangents(N)
        # launch: parameters that move the rays in object space
        L = self._launch_effect()
        if S.infinite_object:
            dep_z, dep_r = L[:, 0], L[:, 1]
            ids = np.flatnonzero((dep_z != 0) | ((dep_r != 0) & (not aim)))
            T.activate(ids)
            for row, v in zip(T.rows(ids), ids):
                T.P[row] = ez*dep_z[v]
                if not aim:
                    T.P[row] += (pp*dep_r[v]) @ np.eye(3)[:2]
                    T.opl[row] = n0*(pp[:, 0]*d[:, 0] + pp[:, 1]*d[:, 1])*dep_r[v]
        else:
            du_m, du_c = L[:, 0], L[:, 1]
            ids = np.flatnonzero((self.dobj != 0) | (((du_m != 0) | (du_c != 0)) & (not aim)))
            T.activate(ids)
            Da = d[:, 2:3]*(ex - d*d[:, :1])
            Db = d[:, 2:3]*(ey - d*d[:, 1:2])
            for row, v in zip(T.rows(ids), ids):
                T.P[row] = -ez*self.dobj[v]
                if not aim:
                    da = fxy[:, 0]*du_c[v] + pp[:, 0]*du_m[v]
                    db = fxy[:, 1]*du_c[v] + pp[:, 1]*du_m[v]
                    T.d[row] = da[:, None]*Da + db[:, None]*Db
        A, B = V, V + 1  # launch parameters of aimed rays
        if aim:
            T.activate([A, B])
            ra, rb = T.rows([A, B])
            if S.infinite_object:
                T.P[ra], T.P[rb] = np.tile(ex, (N, 1)), np.tile(ey, (N, 1))
                T.opl[ra], T.opl[rb] = n0*d[:, 0], n0*d[:, 1]
            else:
                T.d[ra] = d[:, 2:3]*(ex - d*d[:, :1])
                T.d[rb] = d[:, 2:3]*(ey - d*d[:, 1:2])
        d0 = d.copy()
        launch = (T.ids.copy(), T.d.copy())
        frames, (o_img, R_img) = S.frames()
        stop = S.stop_index
        correction = None
        P_last, TP_last = P, None
        for i, s in enumerate(S.surfaces):
            if isinstance(s, rt.CoordinateBreak):
                continue
            o, R = frames[i]
            moved = np.flatnonzero(np.any(self.do[:, i] != 0, axis=1)
                                   | np.any(self.dR[:, i] != 0, axis=(1, 2)))
            idx = np.flatnonzero((self.dn[:, i] != 0) | (self.dn[:, i + 1] != 0))
            new = np.union1d(np.union1d(moved, idx), self.shape[i])
            if i == stop and aim:
                new = np.union1d(new, np.flatnonzero(self.dsd != 0))
            T.activate(new.astype(int))
            ids = T.ids
            # local frame:
            Pl, dl = (P - o) @ R, d @ R
            TPl = (T.P - self._pad(self.do[:, i], ids)[:, None, :]) @ R
            Tdl = T.d @ R
            dR = self._pad(self.dR[:, i], ids)
            if np.any(dR):
                TPl += np.einsum("nj,mjk->mnk", P - o, dR)
                Tdl += np.einsum("nj,mjk->mnk", d, dR)
            with np.errstate(invalid="ignore", divide="ignore"):
                t = s.intersect(Pl, dl)
            Q = Pl + t[:, None]*dl
            x, y = Q[:, 0], Q[:, 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                z, zx, zy = _sag_gradient(s, x, y)
                scale = max(np.nanmax(np.hypot(x, y), initial=0.0),
                            getattr(s, "norm_radius", 0.0), s.semi_diameter or 0.0, 1e-6)
                zxx, zxy, zyy = _hessian(s, x, y, scale)
            Nn = np.stack([-zx, -zy, np.ones(N)], axis=-1)
            # change of the sag and its gradient by the parameters:
            dz_p = np.zeros((len(ids), N))
            dzx_p, dzy_p = np.zeros_like(dz_p), np.zeros_like(dz_p)
            if self.shape[i]:
                rows = T.rows(self.shape[i])
                for row, v in zip(rows, self.shape[i]):
                    with np.errstate(invalid="ignore", divide="ignore"):
                        zp = _sag_gradient(self.plus[v].surfaces[i], x, y)
                        zm = _sag_gradient(self.minus[v].surfaces[i], x, y)
                    h2 = 2*self.h[v]
                    dz_p[row] = (zp[0] - zm[0])/h2
                    dzx_p[row] = (zp[1] - zm[1])/h2
                    dzy_p[row] = (zp[2] - zm[2])/h2
            with np.errstate(invalid="ignore", divide="ignore"):
                den = dl @ np.array([0, 0, 1.0]) - zx*dl[:, 0] - zy*dl[:, 1]
                G = _vdot(TPl, Nn) + t*_vdot(Tdl, Nn) - dz_p
                dt = -G/den
            TQ = TPl + t[None, :, None]*Tdl + dt[:, :, None]*dl
            dn1 = self._pad(self.dn[:, i], ids)
            dn2 = self._pad(self.dn[:, i + 1], ids)
            opl = opl + n[i]*t
            T.opl = T.opl + dn1[:, None]*t + n[i]*dt
            if i == stop and aim:
                correction = self._aim_correction(T, Q, TQ, A, B)
            # normal and its change:
            dzx = zxx*TQ[..., 0] + zxy*TQ[..., 1] + dzx_p
            dzy = zxy*TQ[..., 0] + zyy*TQ[..., 1] + dzy_p
            TNn = np.stack([-dzx, -dzy, np.zeros_like(dzx)], axis=-1)
            nn = np.sqrt(np.sum(Nn**2, axis=-1))
            Nu = Nn/nn[:, None]
            TN = (TNn - _vdot(TNn, Nu)[..., None]*Nu)/nn[:, None]
            ci = np.sum(dl*Nu, axis=-1)
            Tci = _vdot(Tdl, Nu) + _vdot(TN, dl)
            if s.mirror:
                d2 = dl - 2*ci[:, None]*Nu
                Td2 = Tdl - 2*(Tci[..., None]*Nu + ci[:, None]*TN)
            else:
                sg = np.sign(ci)
                Nu, TN, ci, Tci = Nu*sg[:, None], TN*sg[:, None], ci*sg, Tci*sg
                mu = n[i]/n[i + 1]
                Tmu = (dn1*n[i + 1] - n[i]*dn2)/n[i + 1]**2
                with np.errstate(invalid="ignore"):
                    cos_t = np.sqrt(1 - mu**2*(1 - ci**2))
                    Tsin2 = 2*mu*Tmu[:, None]*(1 - ci**2) - 2*mu**2*ci*Tci
                    Tcos = -Tsin2/(2*cos_t)
                d2 = mu*dl + (cos_t - mu*ci)[:, None]*Nu
                Td2 = (Tmu[:, None, None]*dl + mu*Tdl
                       + (Tcos - Tmu[:, None]*ci - mu*Tci)[..., None]*Nu
                       + (cos_t - mu*ci)[:, None]*TN)
            P, d = o + Q @ R.T, d2 @ R.T
            T.P = self._pad(self.do[:, i], ids)[:, None, :] + TQ @ R.T
            T.d = Td2 @ R.T
            if np.any(dR):
                T.P += np.einsum("nk,mjk->mnj", Q, dR)
                T.d += np.einsum("nk,mjk->mnj", d2, dR)
            if correction is not None and i == stop:
                self._apply_correction(T, correction, A, B)
                launch = self._correct_launch(launch, correction, A, B)
                T.drop([A, B])
            P_last = P
            TP_last = (T.ids.copy(), T.P.copy())
        # image plane
        n_img = len(S.surfaces)
        moved = np.flatnonzero(np.any(self.do[:, n_img] != 0, axis=1)
                               | np.any(self.dR[:, n_img] != 0, axis=(1, 2))
                               | (self.dn[:, -1] != 0))
        T.activate(moved)
        ids = T.ids
        nrm = R_img[:, 2]
        do = self._pad(self.do[:, n_img], ids)
        dR = self._pad(self.dR[:, n_img], ids)
        dnrm = dR[:, :, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            den = d @ nrm
            t = ((o_img - P) @ nrm)/den
            dt = (((do @ nrm)[:, None] - (T.P @ nrm)) + dnrm @ (o_img - P).T
                  - t*((T.d @ nrm) + dnrm @ d.T))/den
        P_img = P + t[:, None]*d
        TP_img = T.P + dt[..., None]*d + t[None, :, None]*T.d
        opl = opl + n[-1]*t
        dn_img = self._pad(self.dn[:, -1], ids)
        Topl = T.opl + dn_img[:, None]*t + n[-1]*dt
        I, D = (P_img - o_img) @ R_img, d @ R_img
        TI = (TP_img - do[:, None, :]) @ R_img + np.einsum("nj,mjk->mnk", P_img - o_img, dR)
        TD = T.d @ R_img + np.einsum("nj,mjk->mnk", d, dR)
        # scatter to all parameters:
        out = []
        for arr, shape in ((TI, (N, 3)), (TD, (N, 3)), (Topl, (N,))):
            full = np.zeros((V,) + shape)
            keep = ids < V
            full[ids[keep]] = arr[keep]
            out.append(full)
        dd0 = np.zeros((V, N, 3))
        keep = launch[0] < V
        dd0[launch[0][keep]] = launch[1][keep]
        dPl = np.zeros((V, N, 3))
        if TP_last is not None:
            keep = TP_last[0] < V
            dPl[TP_last[0][keep]] = TP_last[1][keep]

        return RayDerivatives(I, D, opl, d0, P_last, out[0], out[1], out[2], dd0, dPl)

    @staticmethod
    def _pad(arr, ids):
        """Rows ids of a per parameter array (zero for the launch rows)."""

        V = len(arr)
        out = np.zeros((len(ids),) + arr.shape[1:])
        keep = ids < V
        out[keep] = arr[ids[keep]]

        return out

    def _aim_correction(self, T, Q, TQ, A, B):
        """Changes of the launch parameters (a, b) per parameter that keep
        the aimed rays on their stop coordinates."""

        ra, rb = T.rows([A, B])
        J = np.stack([TQ[ra, :, :2], TQ[rb, :, :2]], axis=-1)  # (N, 2, 2): d hit/d(a, b)
        a_s = self.system.surfaces[self.system.stop_index].semi_diameter
        dsd = self._pad(self.dsd, T.ids)
        target = Q[None, :, :2]*(dsd/a_s)[:, None, None]
        rhs = target - TQ[..., :2]  # (m, N, 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            det = J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0]
            ca = (J[:, 1, 1]*rhs[..., 0] - J[:, 0, 1]*rhs[..., 1])/det
            cb = (-J[:, 1, 0]*rhs[..., 0] + J[:, 0, 0]*rhs[..., 1])/det
        ca[[ra, rb]] = 0.0
        cb[[ra, rb]] = 0.0

        return T.ids.copy(), ca, cb

    @staticmethod
    def _apply_correction(T, correction, A, B):
        ids, ca, cb = correction
        ra, rb = T.rows([A, B])
        m = len(ids)
        T.P[:m] += ca[..., None]*T.P[ra] + cb[..., None]*T.P[rb]
        T.d[:m] += ca[..., None]*T.d[ra] + cb[..., None]*T.d[rb]
        T.opl[:m] += ca*T.opl[ra] + cb*T.opl[rb]

    @staticmethod
    def _correct_launch(launch, correction, A, B):
        """Object space directions: the same correction on the launch
        derivatives."""

        l_ids, l_d = launch
        ids, ca, cb = correction
        pos = {v: k for k, v in enumerate(l_ids)}
        N = l_d.shape[1]
        out = np.zeros((len(ids), N, 3))
        for k, v in enumerate(ids):
            if v in pos:
                out[k] = l_d[pos[v]]
        da, db = l_d[pos[A]], l_d[pos[B]]
        out += ca[..., None]*da + cb[..., None]*db

        return ids, out

    # --- OPD ---------------------------------------------------------------
    def opd(self, rd, start=0, count=None):
        """OPD (length units, as `raytrace.System._opd()`) of a group of rays
        relative to its first ray (the chief ray), with derivatives.

        Parameters
        ----------
        rd : RayDerivatives
        start, count : int, optional
            The rays start:start + count of rd (default: all), the first
            of them the chief ray.

        Returns
        -------
        W : (count,) array
        dW : (V, count) array
        """

        S = self.system
        sl = slice(start, None if count is None else start + count)
        I, D, opl = rd.I[sl], rd.D[sl], rd.opl[sl]
        TI, TD, Topl = rd.dI[:, sl], rd.dD[:, sl], rd.dopl[:, sl]
        Ic, dc, TIc, Tdc = I[0], D[0], TI[:, 0], TD[:, 0]
        xp_z = S._pupils()[2]
        size = max(np.sum(np.abs([s.thickness for s in S.surfaces])), 1.0)
        if np.isfinite(xp_z) and abs(xp_z) < 1e8*size:
            Txp = self._cached("xp_z", lambda s: s._pupils()[2])
            Tiz = self._cached("image_z", lambda s: s.image_z)
            k = (xp_z - S.image_z - Ic[2])/dc[2]
            Tk = (Txp - Tiz - TIc[:, 2])/dc[2] - k*Tdc[:, 2]/dc[2]
            Qc = Ic + k*dc
            TQc = TIc + Tk[:, None]*dc + k*Tdc
        else:
            o, R = S.frames()[1]
            Qc = (rd.P_last[sl][0] - o) @ R
            dR = self.dR[:, len(S.surfaces)]
            do = self.do[:, len(S.surfaces)]
            TQc = (rd.dP_last[:, sl][:, 0] - do) @ R + np.einsum("j,mjk->mk",
                                                                  rd.P_last[sl][0] - o, dR)
        L = np.dot(Ic - Qc, dc)
        side = np.sign(L) if L != 0 else 1.0
        Rr = np.linalg.norm(Ic - Qc)
        TR = (TIc - TQc) @ (Ic - Qc)/Rr
        n_img = S.n[-1]
        Tn = self.dn[:, -1]
        w = I - Ic
        Tw = TI - TIc[:, None, :]
        wd = np.sum(w*D, axis=-1)
        Twd = _vdot(Tw, D) + _vdot(TD, w)
        with np.errstate(invalid="ignore"):
            root = np.sqrt(wd**2 - np.sum(w*w, axis=-1) + Rr**2)
            Troot = (wd*Twd - _vdot(Tw, w) + Rr*TR[:, None])/root
        s = wd + side*root
        Ts = Twd + side*Troot
        W = (opl - n_img*s) - (opl[0] - n_img*side*Rr)
        TW = (Topl - n_img*Ts - Tn[:, None]*s) \
            - (Topl[:, :1] - n_img*side*TR[:, None] - Tn[:, None]*side*Rr)

        return W, TW

    # --- Jacobians of operands ---------------------------------------------
    def finite_difference(self, f, columns=None):
        """Central difference columns of a function of the system (steps
        fd_steps).

        Parameters
        ----------
        f : callable
            system -> residuals.
        columns : sequence of int, optional
            Parameters (default: all).

        Returns
        -------
        (residuals, len(columns)) array
        """

        columns = range(self.n_parameters) if columns is None else columns
        if self._fd is None:
            self._fd = self._perturbed(self.h_fd)
        plus, minus = self._fd
        cols = []
        with np.errstate(all="ignore"):
            for k in columns:
                cols.append((np.atleast_1d(np.asarray(f(plus[k]), float))
                             - np.atleast_1d(np.asarray(f(minus[k]), float)))/(2*self.h_fd[k]))

        return np.array(cols).T

    def jacobian(self, operand):
        """Residuals of an operand and their Jacobian: analytic if the
        operand has a method linearize(lin) (columns of parameters that the
        differential trace does not cover by central differences), else
        central differences.

        Parameters
        ----------
        operand : callable
            system -> residuals.

        Returns
        -------
        r : array
        J : (len(r), V) array
        """

        r = J = None
        if hasattr(operand, "linearize"):
            try:
                with np.errstate(all="ignore"):
                    r, J = operand.linearize(self)
            except NotImplementedError:
                r = J = None
        if r is None:
            with np.errstate(all="ignore"):
                r = np.atleast_1d(np.asarray(operand(self.system), float))
            return r, self.finite_difference(operand)
        r, J = np.atleast_1d(np.asarray(r, float)), np.array(J, float).reshape(len(r), -1)
        bad = np.flatnonzero(self.unsupported)
        if len(bad):
            J[:, bad] = self.finite_difference(operand, bad)

        return r, J


def clean(r, J, failed=FAILED):
    """Residuals with failed (non-finite) values set to `failed` and their
    rows (and non-finite derivatives) to zero."""

    r = np.asarray(r, float)
    bad = ~np.isfinite(r)
    J = np.where(np.isfinite(J), J, 0.0)
    J[bad] = 0.0

    return np.where(bad, failed, r), J
