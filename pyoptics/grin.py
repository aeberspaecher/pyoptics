#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Ray tracing through gradient-index (GRIN) media.

In a medium with a continuous index n(r) rays are curved. With the arc
length s the ray equation is d/ds (n dr/ds) = grad n. With the parameter
t, dt = ds/n, it becomes the equation of motion of a particle in the
potential -n**2/2 ::

    d^2 r/dt^2 = grad(n^2)/2,    T = dr/dt = n dr/ds,

where the optical direction vector T has the length n. The optical path
grows as dOPL/dt = n**2. trace() integrates this system with the
Runge-Kutta-Nyström scheme of Sharma, Kumar and Ghatak (1982), fourth
order in the step dt, for many rays at once, in 2D (x, z) or 3D (x, y, z),
and ends rays exactly (to rounding) on a surface g(r) = 0.

Media
-----
Every medium has n(r), n2(r) (= n**2) and force(r) (= grad(n^2)/2) for
positions of the shape (..., dim); `IndexField` takes any function (with a
numerical gradient if none is given). The others have closed form ray
paths, the checks for the tracer:

- Luneburg(R): n = sqrt(2 - (r/R)**2) inside the sphere, 1 outside.
  Inside, force = -r/R**2: the rays are ellipses about the centre,
  r(t) = r0 cos(t/R) + R T0 sin(t/R); a parallel bundle meets in the
  point of the surface opposite to where it entered.
- MaxwellFishEye(R, n0=2): n = n0/(1 + (r/R)**2). All rays are circles;
  the rays from r0 meet again in -r0 R**2/|r0|**2 with the same optical
  path (perfect imaging, Maxwell 1854).
- ParabolicRod(n0, g): n**2 = n0**2 (1 - (g rho)**2) with the distance rho
  from the z axis. Meridional rays are exact sines, x(t) = x0 cos(n0 g t)
  + T0x sin(n0 g t)/(n0 g), z = beta t (beta = T_z, the invariant); the
  pitch 2 pi beta/(n0 g) shrinks with the ray angle.
- SechRod(n0, g): n = n0 sech(g rho). Meridional rays obey
  sinh(g x) = sinh(g x0) cos(g z) + T0x/beta sin(g z): every meridional
  ray has the pitch 2 pi/g (Mikaelian 1980).
- ExponentialLayer(n_far, delta, h): n**2 = n_far**2 - delta exp(-y/h),
  the air above a hot road (delta > 0) or a cold sea (delta < 0); the
  ray paths follow from the invariant beta = n cos(elevation) in closed
  form.

Functions
---------
- trace(): trace rays (positions and directions) through a medium.
- index_map(): the index of a medium on a grid, for the wave solvers.

References
----------
A. Sharma, D. V. Kumar and A. K. Ghatak, "Tracing rays through
graded-index media: a new method", Appl. Opt. 21, 984-987 (1982).

R. K. Luneburg, "Mathematical Theory of Optics", University of California
Press (1964), section 29.

J. C. Maxwell, "Solutions of problems", Cambridge and Dublin Math. J. 8,
188-195 (1854).

A. L. Mikaelian, "Self-focusing media with variable index of refraction",
Progress in Optics 17, 279-345 (1980).

E. W. Marchand, "Gradient Index Optics", Academic Press (1978).

M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), sections 3.2 and 4.2.
"""

import numpy as np


__all__ = ["IndexField", "Luneburg", "MaxwellFishEye", "ParabolicRod", "SechRod",
           "ExponentialLayer", "GRINRays", "trace", "index_map"]


class IndexField(object):
    """A gradient-index medium given by a function.

    Parameters
    ----------
    n : callable
        n(r) for positions r of the shape (..., dim), returns (...).
    force : callable, optional
        grad(n**2)/2 at r, shape (..., dim). If not given, it is evaluated
        by central differences with the step `h`.
    h : number, optional
        Step of the numerical gradient.
    """

    def __init__(self, n=None, force=None, h=1e-6):
        if n is not None:
            self._n = n
        if force is not None:
            self._force = force
        self.h = h

    def n(self, r):
        """Refractive index at the positions r (..., dim)."""
        return self._n(np.asarray(r, dtype=float))

    def n2(self, r):
        """Square of the index at r."""
        return self.n(r)**2

    def force(self, r):
        """grad(n**2)/2 at r, the right hand side of the ray equation."""
        if hasattr(self, "_force"):
            return self._force(np.asarray(r, dtype=float))
        r = np.asarray(r, dtype=float)
        out = np.empty_like(r)
        for a in range(r.shape[-1]):
            e = np.zeros(r.shape[-1])
            e[a] = self.h
            out[..., a] = (self.n2(r + e) - self.n2(r - e))/(4*self.h)
        return out


class _Radial(IndexField):
    """A medium whose index depends on the distance from a centre."""

    def __init__(self, center=None):
        self.center = None if center is None else np.asarray(center, dtype=float)

    def _rel(self, r):
        r = np.asarray(r, dtype=float)
        return r if self.center is None else r - self.center[:r.shape[-1]]


class Luneburg(_Radial):
    """Luneburg lens of radius R: n = sqrt(2 - (r/R)**2) for r < R, 1
    outside (any dimension).

    Parameters
    ----------
    R : number
        Radius.
    center : array, optional
        Centre, default the origin.
    """

    def __init__(self, R=1.0, center=None):
        super(Luneburg, self).__init__(center)
        self.R = float(R)

    def n2(self, r):
        q = np.sum(self._rel(r)**2, axis=-1)/self.R**2
        return np.where(q < 1, 2 - q, 1.0)

    def n(self, r):
        return np.sqrt(self.n2(r))

    def force(self, r):
        d = self._rel(r)
        inside = np.sum(d**2, axis=-1) < self.R**2
        return np.where(inside[..., None], -d/self.R**2, 0.0)

    def exact(self, r0, T0, t):
        """Exact ray inside the lens: r(t) = r0 cos(t/R) + R T0 sin(t/R)
        (relative to the centre, plus the centre), for the start r0 and the
        optical direction T0 (length n(r0)) inside or on the lens."""
        r0, T0 = np.asarray(r0, dtype=float), np.asarray(T0, dtype=float)
        c = 0.0 if self.center is None else self.center[:r0.shape[-1]]
        t = np.asarray(t, dtype=float)[..., None]
        return c + (r0 - c)*np.cos(t/self.R) + self.R*T0*np.sin(t/self.R)


class MaxwellFishEye(_Radial):
    """Maxwell's fish-eye: n = n0/(1 + (r/R)**2) (any dimension). With
    n0 = 2, n = 1 on the sphere r = R.

    Parameters
    ----------
    R : number
    n0 : number, optional
        Index at the centre.
    center : array, optional
    """

    def __init__(self, R=1.0, n0=2.0, center=None):
        super(MaxwellFishEye, self).__init__(center)
        self.R, self.n0 = float(R), float(n0)

    def n(self, r):
        return self.n0/(1 + np.sum(self._rel(r)**2, axis=-1)/self.R**2)

    def force(self, r):
        d = self._rel(r)
        q = 1 + np.sum(d**2, axis=-1)/self.R**2
        # n^2 = n0^2/q^2: grad/2 = -2 n0^2 d/(R^2 q^3)
        return (-2*self.n0**2/(self.R**2*q**3))[..., None]*d

    def image(self, r0):
        """The point where all rays from r0 meet: -r0 R**2/|r0|**2 (relative
        to the centre)."""
        d = self._rel(r0)
        c = 0.0 if self.center is None else self.center[:d.shape[-1]]
        return c - d*self.R**2/np.sum(d**2, axis=-1, keepdims=True)

    def circle(self, r0, d0):
        """Centre and radius of the circular ray through r0 (2D, relative
        positions as for image()) with the direction d0.

        The circle passes through r0 and its image, and is tangent to d0 at
        r0; returns (centre, radius) with shapes (..., 2) and (...)."""
        r0 = np.asarray(r0, dtype=float)
        d0 = np.asarray(d0, dtype=float)
        d0 = d0/np.linalg.norm(d0, axis=-1, keepdims=True)
        nrm = np.stack([-d0[..., 1], d0[..., 0]], axis=-1)
        p = self.image(r0)
        # centre c = r0 + s nrm with |c - p| = |s|: s = |r0 - p|^2/(2 nrm.(p - r0))
        dp = p - r0
        s = np.sum(dp**2, axis=-1)/(2*np.sum(nrm*dp, axis=-1))
        return r0 + s[..., None]*nrm, np.abs(s)


class _Rod(IndexField):
    """Index depends on the distance rho from the z axis (the last
    coordinate)."""

    def _rho2(self, r):
        r = np.asarray(r, dtype=float)
        return np.sum(r[..., :-1]**2, axis=-1)


class ParabolicRod(_Rod):
    """GRIN rod with n**2 = n0**2 (1 - (g rho)**2) about the z axis (the
    last coordinate), clipped at n_min.

    Parameters
    ----------
    n0 : number
        Index on the axis.
    g : number
        Gradient constant; the paraxial pitch is 2 pi/g.
    n_min : number, optional
        Lower limit of the index (the profile would reach zero at
        rho = 1/g).
    """

    def __init__(self, n0, g, n_min=1.0):
        self.n0, self.g, self.n_min = float(n0), float(g), float(n_min)

    def n2(self, r):
        return np.maximum(self.n0**2*(1 - self.g**2*self._rho2(r)), self.n_min**2)

    def n(self, r):
        return np.sqrt(self.n2(r))

    def force(self, r):
        r = np.asarray(r, dtype=float)
        f = -self.n0**2*self.g**2*r
        f[..., -1] = 0
        inside = self.n0**2*(1 - self.g**2*self._rho2(r)) > self.n_min**2
        return np.where(inside[..., None], f, 0.0)

    def pitch(self, beta):
        """Exact pitch of a ray with the invariant beta = T_z = n cos(theta)
        (any ray, also skew ones, that stays above n_min)."""
        return 2*np.pi*np.asarray(beta)/(self.n0*self.g)

    def exact(self, r0, T0, t):
        """Exact ray (any dimension, also skew rays) for the start r0 and the
        optical direction T0, at the parameters t."""
        r0, T0 = np.asarray(r0, dtype=float), np.asarray(T0, dtype=float)
        t = np.asarray(t, dtype=float)[..., None]
        w = self.n0*self.g
        out = r0*np.cos(w*t) + T0*np.sin(w*t)/w
        out[..., -1] = r0[..., -1] + T0[..., -1]*t[..., 0]
        return out


class SechRod(_Rod):
    """GRIN rod with n = n0 sech(g rho) about the z axis (the last
    coordinate): every meridional ray has the pitch 2 pi/g.

    Parameters
    ----------
    n0 : number
    g : number
    """

    def __init__(self, n0, g):
        self.n0, self.g = float(n0), float(g)

    def n(self, r):
        return self.n0/np.cosh(self.g*np.sqrt(self._rho2(r)))

    def force(self, r):
        r = np.asarray(r, dtype=float)
        rho = np.sqrt(self._rho2(r))
        u = self.g*rho
        # n^2 = n0^2 sech^2(u): d(n^2)/2 drho = -g n0^2 sech^2(u) tanh(u)
        with np.errstate(invalid="ignore", divide="ignore"):
            fr = -self.g*self.n0**2*np.tanh(u)/np.cosh(u)**2/np.where(rho > 0, rho, 1.0)
        f = fr[..., None]*r
        f[..., -1] = 0
        return f

    def exact_meridional(self, x0, slope0, z):
        """Exact meridional ray x(z) for the start height x0 and the slope
        dx/dz = slope0 at z = 0: sinh(g x) = sinh(g x0) cos(g z)
        + cosh(g x0) slope0 sin(g z)."""
        x0, slope0, z = np.asarray(x0), np.asarray(slope0), np.asarray(z)
        g = self.g
        return np.arcsinh(np.sinh(g*x0)*np.cos(g*z) + np.cosh(g*x0)*slope0*np.sin(g*z))/g


class ExponentialLayer(IndexField):
    """Stratified medium n**2 = n_far**2 - delta exp(-y/h), y the height
    (coordinate `axis`).

    With delta > 0 (hot ground, lower index near y = 0) rays that leave the
    ground region at a small elevation are bent back upwards: the inferior
    mirage. delta < 0 bends rays down (a superior mirage over cold water).

    Parameters
    ----------
    n_far : number
        Index far above the ground.
    delta : number
        Drop of n**2 at y = 0.
    h : number
        Thickness of the layer.
    axis : int, optional
        Index of the height coordinate, default 1.
    """

    def __init__(self, n_far, delta, h, axis=1):
        self.n_far, self.delta, self.h, self.axis = float(n_far), float(delta), float(h), axis

    def n2(self, r):
        y = np.asarray(r, dtype=float)[..., self.axis]
        return self.n_far**2 - self.delta*np.exp(-y/self.h)

    def n(self, r):
        return np.sqrt(self.n2(r))

    def force(self, r):
        r = np.asarray(r, dtype=float)
        f = np.zeros_like(r)
        f[..., self.axis] = self.delta*np.exp(-r[..., self.axis]/self.h)/(2*self.h)
        return f

    def turning_height(self, beta):
        """Height where a ray with the invariant beta (n times the cosine of
        its elevation) is horizontal (delta > 0, n_far > beta)."""
        return self.h*np.log(self.delta/(self.n_far**2 - np.asarray(beta)**2))

    def horizontal_distance(self, beta, y):
        """Horizontal distance from the turning point at which a ray with the
        invariant beta reaches the height y (delta > 0):
        2 beta h/sqrt(c) artanh(sqrt(1 - delta exp(-y/h)/c)),
        c = n_far**2 - beta**2."""
        beta, y = np.asarray(beta, dtype=float), np.asarray(y, dtype=float)
        c = self.n_far**2 - beta**2
        q = np.clip(self.delta*np.exp(-y/self.h)/c, 0, 1)     # 1 - w**2
        w = np.sqrt(1 - q)
        # artanh(w) = ln((1 + w)**2/q)/2, accurate also for q -> 0
        with np.errstate(divide="ignore"):
            return beta*self.h/np.sqrt(c)*np.log((1 + w)**2/q)


class GRINRays(object):
    """Result of trace().

    Attributes
    ----------
    path : array (n_points, n_rays, dim)
        Positions after every step (if recorded); stopped rays repeat their
        end point.
    position, direction : arrays (n_rays, dim)
        End points and unit directions.
    T : array (n_rays, dim)
        Optical direction vectors n dr/ds at the end.
    opl : array (n_rays,)
        Optical path lengths from the start.
    t : array (n_rays,)
        Ray parameter at the end (sum of the steps).
    stopped : bool array (n_rays,)
        True for the rays that reached the stop surface.
    """

    def __init__(self, path, position, T, opl, t, stopped, opl_path=None):
        self.path, self.position, self.T = path, position, T
        self.opl, self.t, self.stopped = opl, t, stopped
        self.opl_path = opl_path

    @property
    def direction(self):
        return self.T/np.linalg.norm(self.T, axis=-1, keepdims=True)


def _step(medium, R, T, dt):
    """One Runge-Kutta-Nyström step (Sharma et al. 1982); returns the new
    position, optical direction and the optical path of the step."""
    dt = np.asarray(dt, dtype=float)[..., None]
    A = dt*medium.force(R)
    R2 = R + dt/2*T + dt/8*A
    B = dt*medium.force(R2)
    R3 = R + dt*T + dt/2*B
    C = dt*medium.force(R3)
    Rn = R + dt*(T + (A + 2*B)/6)
    Tn = T + (A + 4*B + C)/6
    # optical path: dOPL/dt = n^2 (Simpson rule over the three nodes)
    dopl = dt[..., 0]/6*(medium.n2(R) + 4*medium.n2(R2) + medium.n2(R3))
    return Rn, Tn, dopl


def trace(medium, position, direction, step, t_max=None, max_steps=100000, stop=None,
          record=True, tol=1e-12):
    """Trace rays through a gradient-index medium.

    Integrates d^2r/dt^2 = grad(n^2)/2 (dt = ds/n) with the fourth order
    Runge-Kutta-Nyström scheme of Sharma, Kumar and Ghatak.

    Parameters
    ----------
    medium : IndexField
        The medium (n, n2, force).
    position : array (n_rays, dim) or (dim,)
        Start points.
    direction : array like position
        Start directions (normalized here; the optical direction is n times
        the unit vector).
    step : number
        Step dt of the ray parameter, a geometric length of about step*n.
    t_max : number, optional
        End of the ray parameter (the last step is shortened to end there).
    max_steps : int, optional
    stop : callable, optional
        g(r) -> (n_rays,) for the positions r (n_rays, dim) of all rays: a
        ray ends where its g changes the sign it had at the start, e.g.
        lambda r: r[:, -1] - z_end for a plane. The end point
        is found by shortening the last step (secant iteration).
    record : bool, optional
        Keep every point (GRINRays.path, opl_path).
    tol : number, optional
        Tolerance of |g| at the end point.

    Returns
    -------
    rays : GRINRays
    """

    R = np.atleast_2d(np.asarray(position, dtype=float)).copy()
    d = np.atleast_2d(np.asarray(direction, dtype=float))
    d = np.broadcast_to(d, R.shape)
    T = d/np.linalg.norm(d, axis=-1, keepdims=True)*medium.n(R)[:, None]
    nr = R.shape[0]
    opl = np.zeros(nr)
    t = np.zeros(nr)
    active = np.ones(nr, dtype=bool)
    stopped = np.zeros(nr, dtype=bool)
    if t_max is None and stop is None:
        raise ValueError("give t_max or a stop function")
    if stop is not None:
        g0 = np.broadcast_to(np.asarray(stop(R), dtype=float), (nr,))
        sgn = np.where(g0 >= 0, 1.0, -1.0)
        active &= g0 != 0
        stopped |= g0 == 0
    path, opl_path = ([R.copy()], [opl.copy()]) if record else (None, None)

    def g_of(sub, Rsub):
        # stop() sees all rays (it may have per ray parameters)
        full = R.copy()
        full[sub] = Rsub
        return sgn[sub]*np.asarray(stop(full), dtype=float)[sub]

    for _ in range(int(max_steps)):
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        dt = np.full(len(idx), float(step))
        if t_max is not None:
            dt = np.minimum(dt, t_max - t[idx])
        Rn, Tn, dopl = _step(medium, R[idx], T[idx], dt)
        cross = np.zeros(len(idx), dtype=bool)
        if stop is not None:
            gn = g_of(idx, Rn)
            cross = gn <= 0
            if cross.any():
                # secant iteration on the length of the last step
                c = np.nonzero(cross)[0]
                R0, T0 = R[idx[c]], T[idx[c]]
                lo, hi = np.zeros(len(c)), dt[c].copy()
                glo, ghi = g_of(idx[c], R0), gn[c]
                for _ in range(60):
                    s = hi - ghi*(hi - lo)/(ghi - glo)
                    s = np.where(np.isfinite(s) & (s > lo) & (s < hi), s, (lo + hi)/2)
                    Rs, Ts, ds = _step(medium, R0, T0, s)
                    gs = g_of(idx[c], Rs)
                    pos = gs > 0
                    lo, glo = np.where(pos, s, lo), np.where(pos, gs, glo)
                    hi, ghi = np.where(pos, hi, s), np.where(pos, ghi, gs)
                    # Illinois: halve the retained end to avoid stagnation
                    glo = np.where(~pos, glo/2, glo)
                    ghi = np.where(pos, ghi/2, ghi)
                    if np.all(np.abs(gs) < tol):
                        break
                Rn[c], Tn[c], dopl[c], dt[c] = Rs, Ts, ds, s
        R[idx], T[idx] = Rn, Tn
        opl[idx] += dopl
        t[idx] += dt
        done = cross.copy()
        if t_max is not None:
            done |= t[idx] >= t_max - 1e-12*abs(t_max)
        stopped[idx[cross]] = True
        active[idx[done]] = False
        if record:
            path.append(R.copy())
            opl_path.append(opl.copy())
    if record:
        path, opl_path = np.array(path), np.array(opl_path)
    return GRINRays(path, R, T, opl, t, stopped, opl_path)


def index_map(medium, *coords):
    """The index of a medium on the grid of the coordinate arrays (indexing
    "ij"): index_map(m, x, z) has the shape (len(x), len(z))."""
    grids = np.meshgrid(*coords, indexing="ij")
    return medium.n(np.stack(grids, axis=-1))
