#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""The merit function landscape of a lens: global optimization and saddle
points.

A damped least squares run (`lens_optimization.DampedLeastSquares`) goes
downhill to the nearest local minimum. Which minimum that is depends on the
start, and a lens design problem has many: the merit function of even a
Cooke triplet with fixed glasses and thicknesses has a dozen or more. This
module looks at the landscape as a whole:

- `random_starts()` draws starting points in a box of the variables that
  trace (no failed rays), `multistart()` runs damped least squares from
  each and `cluster_minima()` merges the end points into distinct local
  minima, with the number of starts that ended in each (the relative size
  of its basin of attraction);
- `stationarity()` gives the gradient and the Hessian of the merit function
  (central differences, in scaled variables) and its eigenvalues: the Morse
  index, the number of negative eigenvalues, is 0 at a minimum and 1 at a
  saddle point that links two minima; `refine_stationary()` converges onto
  the nearest stationary point of any kind with Newton's method;
- `insert_null_lens()`, `null_lens_gradient()` and `null_lens_curvatures()`
  build the saddle points of Bociort's saddle-point construction: a lens of
  zero thickness with two equal curvatures changes nothing, and at the
  right curvature it is a saddle point of the system with one lens more,
  from which `descend_from_saddle()` runs downhill on both sides into two
  new minima;
- `minimum_energy_path()`: a climbing image nudged elastic band between two
  minima, whose highest point is the saddle point between them;
- `basin_hopping()`: a Monte Carlo walk from minimum to minimum (random jump,
  local optimization, Metropolis acceptance of the new minimum).

Every local optimization counts the merit function evaluations it used
(`LocalRun.evaluations`; a Jacobian costs one evaluation per variable plus
one), so that global strategies can be compared at equal cost. Notebook 154
(global optimization and saddle points).

References
----------
F. Bociort, A. Serebriakov and M. van Turnhout, "Saddle points in the merit
function landscape of systems of thin lenses in contact", Proc. SPIE 5523,
174-184 (2004).

F. Bociort, E. van Driel and A. Serebriakov, "Networks of local minima in
optical system optimization", Opt. Lett. 29, 189-191 (2004).

F. Bociort and M. van Turnhout, "Finding new local minima in lens design
landscapes by constructing saddle points", Opt. Eng. 48, 063001 (2009).

M. van Turnhout and F. Bociort, "Instabilities and fractal basins of
attraction in optical system optimization", Opt. Express 17, 314-328 (2009).

J. Baker, "An algorithm for the location of transition states", J. Comput.
Chem. 7, 385-395 (1986).

D. J. Wales and J. P. K. Doye, "Global optimization by basin-hopping and the
lowest energy structures of Lennard-Jones clusters containing up to 110
atoms", J. Phys. Chem. A 101, 5111-5116 (1997).

G. Henkelman, B. P. Uberuaga and H. Jonsson, "A climbing image nudged elastic
band method for finding saddle points and minimum energy paths", J. Chem.
Phys. 113, 9901-9904 (2000).
"""

import copy
from collections import namedtuple

import numpy as np

from .lens_optimization import FAILED, _uncached

LocalRun = namedtuple("LocalRun", ["start", "x", "merit", "iterations", "evaluations", "success",
                                   "message"])
LocalRun.__doc__ = """One local optimization: starting variables, final
variables, final merit function, number of iterations, number of merit
function evaluations, whether it converged and the message of the
optimizer (merit inf for a run that failed)."""

LocalMinimum = namedtuple("LocalMinimum", ["x", "merit", "count", "members", "at_bound"])
LocalMinimum.__doc__ = """A distinct local minimum from `cluster_minima()`:
variables and merit function of its best member, the number of runs that
ended in it, their indices and whether a variable is at a bound."""

Stationarity = namedtuple("Stationarity", ["gradient", "hessian", "eigenvalues", "eigenvectors",
                                           "scale", "morse_index"])
Stationarity.__doc__ = """Gradient and Hessian of the merit function in the
scaled variables x/scale, the eigenvalues (ascending) and eigenvectors
(columns) of the Hessian, the scales, and the Morse index (the number of
eigenvalues below -tol times the largest absolute one)."""

BasinHoppingResult = namedtuple("BasinHoppingResult", ["x", "merit", "runs", "accepted", "current",
                                                       "evaluations"])
BasinHoppingResult.__doc__ = """Result of `basin_hopping()`: best variables and
merit function, all local runs (the first from the start), whether each hop
was accepted, the merit function of the current minimum after each hop, and
the cumulative number of merit function evaluations after each run."""

Path = namedtuple("Path", ["x", "merit", "saddle", "converged"])
Path.__doc__ = """Result of `minimum_energy_path()`: the images (rows of
variables, from the first minimum to the second), their merit function, the
index of the highest image (the saddle point with a climbing image) and
whether the band converged."""


# --- helpers -----------------------------------------------------------------

def _scale(optimizer):
    return np.asarray(optimizer._scale, float)


def _bounds(optimizer, bounds):
    if bounds is None:
        bounds = [v.bounds for v in optimizer.variables]
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds], float)
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds], float)

    return lo, hi


def merit_at(optimizer, x):
    """Merit function of the optimizer's system with the variables x (inf
    if the system cannot be built; a failed ray counts as in the optimizer,
    with the residual `lens_optimization.FAILED`).

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
    x : array

    Returns
    -------
    float
    """

    try:
        r = optimizer.residuals(optimizer.system_at(np.asarray(x, float)))
    except (ValueError, ArithmeticError, NotImplementedError, np.linalg.LinAlgError):
        return np.inf
    if not np.all(np.isfinite(r)):
        return np.inf

    return float(np.sum(r**2))


def traces(optimizer, x):
    """Whether all rays of all operands trace with the variables x (no
    residual of a failed ray, see `lens_optimization.FAILED`).

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
    x : array

    Returns
    -------
    bool
    """

    try:
        r = optimizer.residuals(optimizer.system_at(np.asarray(x, float)))
    except (ValueError, ArithmeticError, NotImplementedError, np.linalg.LinAlgError):
        return False
    w = np.min(np.abs(optimizer.weights), initial=1.0)

    return bool(np.all(np.isfinite(r)) and np.all(np.abs(r) < 0.999*FAILED*w))


def random_starts(optimizer, n, bounds=None, rng=None, max_tries=None):
    """Random starting points, uniform in a box of the variables, that trace.

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
    n : int
        Number of starting points.
    bounds : sequence of (lower, upper), optional
        Box per variable (default: the bounds of the variables, which must
        then be finite).
    rng : numpy.random.Generator or int, optional
    max_tries : int, optional
        Number of draws before giving up (default 200 n).

    Returns
    -------
    (m, variables) array
        The m <= n points found.
    """

    rng = np.random.default_rng(rng)
    lo, hi = _bounds(optimizer, bounds)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("random_starts() needs finite bounds for all variables")
    out = []
    for _ in range(200*n if max_tries is None else max_tries):
        x = rng.uniform(lo, hi)
        if traces(optimizer, x):
            out.append(x)
            if len(out) == n:
                break

    return np.array(out).reshape(-1, len(lo))


class _Counted:
    """The residuals of an optimizer, counting the calls."""

    def __init__(self, residuals):
        self.residuals = residuals
        self.calls = 0

    def __call__(self, system=None):
        self.calls += 1
        return self.residuals(system)


def local_search(optimizer, x, max_iterations=80, ftol=1e-6, **kwargs):
    """Damped least squares from the variables x (the optimizer is not
    changed).

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
    x : array
        Starting variables.
    max_iterations, ftol : optional
        See `DampedLeastSquares.run()`.
    **kwargs
        Passed on to `DampedLeastSquares.run()`.

    Returns
    -------
    LocalRun
    """

    x = np.asarray(x, float)
    other = copy.copy(optimizer)
    counter = _Counted(other.residuals)
    other.residuals = counter
    try:
        other.system = optimizer.system_at(x)
        res = other.run(max_iterations=max_iterations, ftol=ftol, **kwargs)
    except (ValueError, ArithmeticError, NotImplementedError, np.linalg.LinAlgError) as e:
        return LocalRun(x, x, np.inf, 0, counter.calls, False, "failed: {}".format(e))
    merit = res.merit if traces(optimizer, res.x) else np.inf

    return LocalRun(x, np.asarray(res.x, float), merit, len(res.history) - 1, counter.calls,
                    res.success, res.message)


def multistart(optimizer, starts, **kwargs):
    """Damped least squares from each of several starting points.

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
    starts : (n, variables) array
    **kwargs
        Passed on to `local_search()`.

    Returns
    -------
    list of LocalRun
        In the order of the starts.
    """

    return [local_search(optimizer, x, **kwargs) for x in np.asarray(starts, float)]


def cluster_minima(runs, optimizer, xtol=0.05, merit_rtol=1e-3):
    """Merge the end points of local runs into distinct local minima.

    Two end points belong to the same minimum if no variable differs by
    more than xtol times its scale (the optimizer's typical change) and
    their merit functions agree to merit_rtol. Failed runs (merit inf) are
    left out.

    Parameters
    ----------
    runs : sequence of LocalRun
    optimizer : lens_optimization.DampedLeastSquares
        For the scales and bounds of the variables.
    xtol, merit_rtol : number, optional

    Returns
    -------
    list of LocalMinimum
        Sorted by merit function, best first.
    """

    scale = _scale(optimizer)
    lo, hi = _bounds(optimizer, None)
    order = sorted((k for k, r in enumerate(runs) if np.isfinite(r.merit)),
                   key=lambda k: runs[k].merit)
    groups = []
    for k in order:
        r = runs[k]
        for g in groups:
            best = runs[g[0]]
            if (np.max(np.abs(r.x - best.x)/scale) <= xtol
                    and abs(r.merit - best.merit) <= merit_rtol*max(best.merit, 1e-300)):
                g.append(k)
                break
        else:
            groups.append([k])
    out = []
    for g in groups:
        x = runs[g[0]].x
        tol = 1e-9*scale
        at_bound = bool(np.any((x <= lo + tol) | (x >= hi - tol)))
        out.append(LocalMinimum(x, runs[g[0]].merit, len(g), g, at_bound))

    return out


# --- stationary points ---------------------------------------------------------

def stationarity(optimizer, x=None, h=1e-3, tol=1e-6):
    """Gradient and Hessian of the merit function (central differences in
    the scaled variables x/scale) and the eigenvalues of the Hessian.

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
    x : array, optional
        Variables (default: the optimizer's current ones).
    h : number, optional
        Step in scaled variables (a fraction of the typical change).
    tol : number, optional
        Eigenvalues below -tol times the largest absolute eigenvalue count
        as negative (for the Morse index). The accuracy of the finite
        differences is limited by the largest eigenvalue: an eigenvalue
        that is zero (a flat direction) can come out with either sign, a
        few 1e-4 of the largest.

    Returns
    -------
    Stationarity
    """

    x = optimizer.x if x is None else np.asarray(x, float)
    scale = _scale(optimizer)
    n = len(x)
    E = np.eye(n)*h

    def f(dz):
        return merit_at(optimizer, x + dz*scale)

    f0 = f(np.zeros(n))
    fp = np.array([f(E[i]) for i in range(n)])
    fm = np.array([f(-E[i]) for i in range(n)])
    g = (fp - fm)/(2*h)
    H = np.diag((fp - 2*f0 + fm)/h**2)
    for i in range(n):
        for j in range(i + 1, n):
            d = (f(E[i] + E[j]) - f(E[i] - E[j]) - f(-E[i] + E[j]) + f(-E[i] - E[j]))/(4*h*h)
            H[i, j] = H[j, i] = d
    w, V = np.linalg.eigh(H)
    big = max(np.max(np.abs(w), initial=0.0), 1e-300)
    index = int(np.sum(w < -tol*big))

    return Stationarity(g, H, w, V, scale, index)


def descend_from_saddle(optimizer, x=None, step=0.1, stationary=None, **kwargs):
    """Local optimizations from both sides of a saddle point, along the
    eigenvector of the most negative eigenvalue of the Hessian.

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
    x : array, optional
        The saddle point (default: the optimizer's current variables).
    step : number, optional
        Distance of the two starts from the saddle point, in scaled
        variables.
    stationary : Stationarity, optional
        Of the saddle point, to reuse.
    **kwargs
        Passed on to `local_search()`.

    Returns
    -------
    runs : (LocalRun, LocalRun)
        From x - step v and from x + step v.
    stationary : Stationarity
    """

    x = optimizer.x if x is None else np.asarray(x, float)
    st = stationarity(optimizer, x) if stationary is None else stationary
    v = st.eigenvectors[:, 0]*st.scale
    runs = tuple(local_search(optimizer, x + s*step*v, **kwargs) for s in (-1.0, 1.0))

    return runs, st


def insert_null_lens(system, surface, material=None, curvature=None, distance=0.0):
    """Insert a lens of zero thickness with two equal curvatures, which
    changes nothing, behind a surface.

    The gap behind `surface` is split: the lens sits `distance` behind the
    surface, and the rest of the gap follows it. Both new surfaces have the
    curvature `curvature`; the medium between them is `material`, the
    medium behind them that behind `surface`.

    In contact with the surface (distance 0), with the curvature of the
    surface and the material in front of it (the default), the new system
    is a stationary point of the merit function in the variables of the
    old system plus the two new curvatures (Bociort and van Turnhout): any
    change of the new curvatures is, to first order, a change of the
    curvature of `surface`, so a minimum of the old system becomes a
    saddle point (or a minimum) of the new one. Elsewhere, a curvature at
    which the system is stationary is found with `null_lens_gradient()`.

    Parameters
    ----------
    system : raytrace.System
    surface : int
        Index of the surface behind which the lens is inserted.
    material : optional
        Material of the lens (a glass of `lens_optimization.GLASSES`, any
        material of `raytrace.Surface`, or a number for an index). Default:
        the medium in front of the surface.
    curvature : number, optional
        Default: the curvature of the surface.
    distance : number, optional
        Distance of the lens behind the surface.

    Returns
    -------
    system : raytrace.System
        With two more surfaces (stop and other indices behind `surface`
        shift by two).
    surfaces : (int, int)
        Indices of the two new surfaces.
    """

    from .raytrace import Surface

    new = _uncached(system)
    S = new.surfaces
    s = S[surface]
    if distance < 0 or distance > s.thickness:
        raise ValueError("the lens must lie inside the gap behind the surface")
    lens = _medium(system, surface - 1) if material is None else _spec(material)
    behind = _medium(system, surface)
    c = s.c if curvature is None else curvature
    R = np.inf if c == 0 else 1/c
    rest = s.thickness - distance
    s.thickness = distance
    S.insert(surface + 1, Surface(R, 0.0, **lens))
    S.insert(surface + 2, Surface(R, rest, **behind))
    if new.stop_index > surface:
        new.stop_index += 2
    for k in [k for k in new.__dict__ if k.startswith("_")]:
        del new.__dict__[k]

    return new, (surface + 1, surface + 2)


def _spec(material):
    """Keyword arguments of raytrace.Surface for a medium."""

    if np.isscalar(material):
        return {"n_after": float(material)}

    return {"material": material}


def _medium(system, i):
    """Keyword arguments of raytrace.Surface for the medium behind surface i
    (-1: object space)."""

    for s in system.surfaces[i::-1] if i >= 0 else []:
        if s.material is not None:
            return {"material": s.material}
        if s.n_after is not None:
            return {"n_after": s.n_after}

    return {"n_after": system.n_object}


def null_lens_gradient(optimizer, surfaces, curvatures, h=1e-3):
    """Derivative of the merit function with respect to the power of a
    null lens, against its common curvature.

    For each curvature c, both surfaces of the lens are set to c and the
    merit function is differentiated (central differences, h times the
    curvature scale) along (c - h/2, c + h/2): the bending of the null lens
    does not change the merit function, its power does. A zero of the
    derivative, with the gradient of the other variables zero (the old
    system at a minimum), is a stationary point of the new system,
    usually a saddle point of Morse index 1.

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
        Of the system with the null lens (its variables are not used).
    surfaces : (int, int)
        The two surfaces of the lens.
    curvatures : array
    h : number, optional

    Returns
    -------
    array
        d merit / d (c2 - c1), one value per curvature.
    """

    i, j = surfaces
    base = optimizer.system
    scale = 1/np.sum(np.abs([s.thickness for s in base.surfaces]))
    d = h*scale
    out = []
    for c in np.atleast_1d(curvatures):
        vals = []
        for sgn in (-1, 1):
            s = _uncached(base)
            for k, cc in ((i, c - sgn*d/2), (j, c + sgn*d/2)):
                s.surfaces[k].R = np.inf if cc == 0 else 1/cc
            s = optimizer._solve(s)
            try:
                vals.append(optimizer.merit(s))
            except (ValueError, ArithmeticError, NotImplementedError, np.linalg.LinAlgError):
                vals.append(np.nan)
        out.append((vals[1] - vals[0])/(2*d))

    return np.array(out)


def null_lens_curvatures(optimizer, surfaces, curvatures, h=1e-3):
    """Curvatures at which a null lens makes the system stationary: the
    zeros of `null_lens_gradient()`, bracketed on a grid of curvatures and
    refined by Brent's method.

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
        Of the system with the null lens.
    surfaces : (int, int)
    curvatures : array
        Grid on which sign changes are looked for.
    h : number, optional

    Returns
    -------
    array
        The zeros (usually none or two: to third order the derivative is
        quadratic in the curvature).
    """

    from scipy.optimize import brentq

    c = np.asarray(curvatures, float)
    g = null_lens_gradient(optimizer, surfaces, c, h)
    roots = []
    for k in range(len(c) - 1):
        if np.isfinite(g[k]) and np.isfinite(g[k + 1]) and g[k]*g[k + 1] < 0:
            roots.append(brentq(lambda cc: null_lens_gradient(optimizer, surfaces, [cc], h)[0],
                                c[k], c[k + 1], xtol=1e-10))

    return np.array(roots)


def refine_stationary(optimizer, x=None, iterations=10, tol=1e-8, h=1e-3, max_step=0.05,
                      index=None):
    """Converge onto a stationary point near x: Newton's method on the
    gradient (the nearest stationary point of any kind), or eigenvector
    following onto a saddle point of a given Morse index.

    With index=None, Newton steps go to the nearest stationary point, e.g.
    to follow a saddle point while a parameter that is not a variable
    changes. With index=k, partitioned rational function steps (Baker 1986)
    go uphill along the k lowest eigenvectors of the Hessian and downhill
    along all others, which converges onto a saddle point of index k from
    much farther away (e.g. from the highest image of an elastic band).

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
    x : array, optional
        Start (default: the current variables).
    iterations : int, optional
    tol : number, optional
        Stop when the step (scaled) is below this.
    h : number, optional
        Finite difference step of `stationarity()`.
    max_step : number, optional
        Largest step (scaled; longer ones are shortened).
    index : int, optional
        Morse index of the saddle point to converge onto.

    Returns
    -------
    x : array
    stationary : Stationarity
        At x.
    """

    x = optimizer.x if x is None else np.asarray(x, float)
    st = stationarity(optimizer, x, h)
    for _ in range(iterations):
        if index is None:
            dz = -np.linalg.pinv(st.hessian, rcond=1e-10) @ st.gradient
        else:
            b, V = st.eigenvalues, st.eigenvectors
            F = V.T @ st.gradient
            dz_e = np.zeros(len(b))
            up, down = slice(0, index), slice(index, len(b))
            for part, pick in ((up, -1), (down, 0)):
                bp, Fp = b[part], F[part]
                if len(bp) == 0:
                    continue
                aug = np.zeros((len(bp) + 1, len(bp) + 1))
                aug[:-1, :-1] = np.diag(bp)
                aug[:-1, -1] = aug[-1, :-1] = Fp
                lam = np.linalg.eigvalsh(aug)[pick]
                d = bp - lam
                d = np.where(np.abs(d) < 1e-12*np.max(np.abs(b)), 1e-12*np.max(np.abs(b)), d)
                dz_e[part] = -Fp/d
            dz = V @ dz_e
        dz *= min(1.0, max_step/max(np.max(np.abs(dz)), 1e-300))
        x = x + dz*st.scale
        st = stationarity(optimizer, x, h)
        if np.max(np.abs(dz)) < tol:
            break

    return x, st


# --- paths between minima -------------------------------------------------------

def _gradient(optimizer, x):
    """Gradient of the merit function 2 J^T r at x (forward difference
    Jacobian of the optimizer)."""

    s = optimizer.system_at(x)
    J, _, r, _ = optimizer.jacobian(s)

    return 2*J.T @ r, float(np.sum(r**2))


def minimum_energy_path(optimizer, xa, xb, images=11, iterations=200, step=0.02, climbing=True,
                        gtol=None, path=None):
    """Climbing image nudged elastic band between two minima.

    A chain of images between the minima xa and xb relaxes under the
    component of the merit function gradient perpendicular to the path
    plus springs along it (in scaled variables); with a climbing image, the
    highest image climbs up along the path and ends in the saddle point
    (Henkelman, Uberuaga and Jonsson). The images move by steepest descent
    with a step limited to `step` (scaled) per iteration.

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
    xa, xb : array
        The two minima.
    images : int, optional
        Number of images including the two ends.
    iterations : int, optional
    step : number, optional
        Largest move of an image per iteration, in scaled variables.
    climbing : bool, optional
        Turn the highest image into a climbing image after a third of the
        iterations.
    gtol : number, optional
        Stop when the largest projected force (scaled) is below this
        (default: 1e-3 times the largest at the start).
    path : (images, variables) array, optional
        Starting chain (default: a straight line).

    Returns
    -------
    Path
    """

    scale = _scale(optimizer)
    za, zb = np.asarray(xa, float)/scale, np.asarray(xb, float)/scale
    if path is None:
        t = np.linspace(0, 1, images)[:, None]
        Z = za + t*(zb - za)
    else:
        Z = np.asarray(path, float)/scale
        images = len(Z)
    E = np.zeros(images)
    E[0], E[-1] = merit_at(optimizer, Z[0]*scale), merit_at(optimizer, Z[-1]*scale)
    f0 = None
    converged = False
    ci = None
    for it in range(iterations):
        G = np.zeros_like(Z)
        for k in range(1, images - 1):
            try:
                g, E[k] = _gradient(optimizer, Z[k]*scale)
            except (ValueError, ArithmeticError, NotImplementedError, np.linalg.LinAlgError):
                g, E[k] = np.zeros(len(scale)), np.inf
            G[k] = g*scale  # gradient in scaled variables
        # spring constant from the merit function scale
        F = np.zeros_like(Z)
        if climbing and it >= iterations//3:
            ci = 1 + int(np.argmax(E[1:-1]))
        for k in range(1, images - 1):
            # upwind tangent
            if E[k + 1] > E[k] > E[k - 1]:
                tau = Z[k + 1] - Z[k]
            elif E[k + 1] < E[k] < E[k - 1]:
                tau = Z[k] - Z[k - 1]
            else:
                dmax = max(abs(E[k + 1] - E[k]), abs(E[k - 1] - E[k]))
                dmin = min(abs(E[k + 1] - E[k]), abs(E[k - 1] - E[k]))
                if E[k + 1] > E[k - 1]:
                    tau = (Z[k + 1] - Z[k])*dmax + (Z[k] - Z[k - 1])*dmin
                else:
                    tau = (Z[k + 1] - Z[k])*dmin + (Z[k] - Z[k - 1])*dmax
            tau = tau/max(np.linalg.norm(tau), 1e-300)
            g = G[k]
            if k == ci:
                F[k] = -g + 2*(g @ tau)*tau
            else:
                f_perp = -(g - (g @ tau)*tau)
                # spring force along the path keeps the images equally spaced
                d = np.linalg.norm(Z[k + 1] - Z[k]) - np.linalg.norm(Z[k] - Z[k - 1])
                F[k] = f_perp + d*tau*np.linalg.norm(g)/max(
                    np.linalg.norm(Z[-1] - Z[0])/(images - 1), 1e-300)
        fmax = np.max(np.linalg.norm(F[1:-1], axis=1))
        if f0 is None:
            f0 = fmax
        if fmax < (1e-3*f0 if gtol is None else gtol) and (ci is not None or not climbing):
            converged = True
            break
        norms = np.linalg.norm(F, axis=1, keepdims=True)
        move = F/np.maximum(norms, 1e-300)*np.minimum(norms/max(fmax, 1e-300), 1.0)*step
        Z[1:-1] += move[1:-1]
    for k in range(1, images - 1):
        E[k] = merit_at(optimizer, Z[k]*scale)
    saddle = 1 + int(np.argmax(E[1:-1])) if images > 2 else 0

    return Path(Z*scale, E, saddle, converged)


# --- basin hopping ----------------------------------------------------------------

def basin_hopping(optimizer, x0, hops=30, step=0.5, temperature=1.0, bounds=None, rng=None,
                  **kwargs):
    """Basin hopping: a Monte Carlo walk between local minima.

    From the current minimum, the variables jump by a normal random step
    (`step` times their scale, redrawn until the rays trace), damped least
    squares runs into the next minimum, and the new minimum is accepted
    with the Metropolis probability ``min(1, exp(-(ln Phi_new -
    ln Phi)/T))`` (Wales and Doye). The temperature acts on the logarithm
    of the merit function, so that T = 1 accepts a minimum e times worse
    with probability 1/e at any level of correction.

    Parameters
    ----------
    optimizer : lens_optimization.DampedLeastSquares
    x0 : array
        Start (optimized first).
    hops : int, optional
    step : number, optional
        Size of a jump in scaled variables.
    temperature : number, optional
    bounds : sequence of (lower, upper), optional
        Jumps are clipped to these (default: the bounds of the variables).
    rng : numpy.random.Generator or int, optional
    **kwargs
        Passed on to `local_search()`.

    Returns
    -------
    BasinHoppingResult
    """

    rng = np.random.default_rng(rng)
    scale = _scale(optimizer)
    lo, hi = _bounds(optimizer, bounds)
    run = local_search(optimizer, x0, **kwargs)
    runs, accepted, current, evals = [run], [True], [run.merit], [run.evaluations]
    x, phi = run.x, run.merit
    best = run
    for _ in range(hops):
        trial = None
        n_tries = 0
        for _ in range(50):
            n_tries += 1
            xt = np.clip(x + step*scale*rng.standard_normal(len(x)), lo, hi)
            if traces(optimizer, xt):
                trial = xt
                break
        if trial is None:
            runs.append(LocalRun(x, x, np.inf, 0, n_tries, False, "no jump traces"))
            accepted.append(False)
            current.append(phi)
            evals.append(evals[-1] + n_tries)
            continue
        r = local_search(optimizer, trial, **kwargs)
        runs.append(r)
        evals.append(evals[-1] + r.evaluations + n_tries)
        ok = np.isfinite(r.merit) and (
            not np.isfinite(phi) or r.merit <= phi
            or rng.random() < np.exp(-np.log(r.merit/max(phi, 1e-300))/temperature))
        accepted.append(bool(ok))
        if ok:
            x, phi = r.x, r.merit
        current.append(phi)
        if r.merit < best.merit:
            best = r

    return BasinHoppingResult(best.x, best.merit, runs, np.array(accepted), np.array(current),
                              np.array(evals))
