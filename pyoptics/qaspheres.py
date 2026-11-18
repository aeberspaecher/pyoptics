#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Forbes Q-type aspheres: slope-orthogonal (Q-bfs) and sag-orthogonal
(Q-con) descriptions of rotationally symmetric surfaces.

The usual even asphere adds a power series to a conic ::

    z(r) = c r**2/(1 + sqrt(1 - (1 + K) c**2 r**2)) + A4 r**4 + A6 r**6 + ...

The monomials r**4, r**6, ... look more and more alike over the aperture,
so high order designs carry huge coefficients of alternating sign that
cancel each other to a few micrometres, tens of significant digits are
needed to reproduce a surface, and the coefficients say nothing about how
hard the surface is to make or to test. G. W. Forbes (2007) replaced them
by two families of orthogonal polynomials in the normalized radius
u = r/r_norm:

Q-bfs (strong aspheres, slope orthogonal)
    The surface is a *best fit sphere* of curvature c_bfs through the
    vertex and the rim (u = 1) plus a departure measured nearly along the
    normal of that sphere ::

        z(r) = c_bfs r**2/(1 + sqrt(1 - c_bfs**2 r**2))
               + u**2 (1 - u**2)/sqrt(1 - c_bfs**2 r**2) * sum_m a_m Q_m^bfs(u**2).

    The factor u**2 (1 - u**2) pins the departure to zero at the vertex
    and at the rim, and the polynomials Q_m^bfs (degree m in x = u**2) are
    orthonormal in the *slope* of the departure, ::

        (2/pi) int_0^1 S_m'(u) S_n'(u) du/sqrt(1 - u**2) = delta_mn,
        S_m(u) = u**2 (1 - u**2) Q_m^bfs(u**2),

    so that the mean square slope of the departure (with the weight
    1/sqrt(1 - u**2), which stresses the rim where the slopes are largest)
    is simply sum_m a_m**2 / r_norm**2 (for c_bfs r small; exact for the
    term in the brackets). The coefficients a_m are lengths: the
    contribution of each term to the departure, directly comparable with
    the wavelength and with the fringe density in an interferometer.
    Q_0 = 1, Q_1 = (13 - 16 x)/sqrt(19), Q_2 = sqrt(2/95) (29 - 4 x (25 - 19 x)).

Q-con (mild aspheres on a conic, sag orthogonal)
    ::

        z(r) = c r**2/(1 + sqrt(1 - (1 + K) c**2 r**2)) + u**4 sum_m a_m Q_m^con(u**2),

    with the Jacobi polynomials Q_m^con(x) = P_m^(0,4)(2 x - 1). They are
    orthogonal in the sag, int_0^1 u**8 Q_m Q_n u du ~ delta_mn, so the
    terms do not fight each other and the coefficients decay.

The Q-bfs polynomials are built here by Gram-Schmidt orthonormalization of
Chebyshev polynomials in x with Gauss-Chebyshev quadrature (exact for the
polynomial integrands), stable to more than 30 terms; the published
recurrence of Forbes gives the same polynomials. Both families are summed
as one Chebyshev series (`numpy.polynomial.chebyshev`).

`QbfsSurface` and `QconSurface` are `raytrace.Surface` subclasses: exact
ray intersections by Newton steps on the full sag, normals from the
analytic derivative, the vertex curvature for the paraxial data and the
r**4 term of the departure for the Seidel sums. `fit_qbfs()` and
`fit_qcon()` convert any sag (e.g. an even asphere from a patent) into Q
coefficients; `best_fit_sphere()` gives Forbes' sphere through vertex and
rim. `QCoefficient` is a `lens_optimization.Variable` for the damped least
squares optimizer.

References
----------
G. W. Forbes, "Shape specification for axially symmetric optical
surfaces", Opt. Express 15, 5218-5226 (2007).

G. W. Forbes, "Robust, efficient computational methods for axially
symmetric optical aspheres", Opt. Express 18, 19700-19712 (2010).

G. W. Forbes, "Manufacturability estimates for optical aspheres", Opt.
Express 19, 9923-9941 (2011).

ISO 10110-12:2019, "Optics and photonics - Preparation of drawings for
optical elements and systems - Part 12: Aspheric surfaces" (includes the Q
forms).
"""

from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import eval_jacobi

from . import abcd
from .raytrace import Surface, _norm


__all__ = ["qbfs", "qbfs_series", "qcon", "qcon_series", "sag_qbfs", "sag_qcon",
           "QbfsSurface", "QconSurface", "QconQ2DSurface", "best_fit_sphere", "fit_qbfs", "fit_qcon",
           "even_asphere_sag", "rms_slope", "QCoefficient", "q2d", "Q2DSurface",
           "Q2DCoefficient"]


def _x_to_t(x):
    """x in [0, 1] to the Chebyshev variable t in [-1, 1]."""

    return 2*np.asarray(x, dtype=float) - 1


@lru_cache(maxsize=None)
def _qbfs_coefficients(n_terms):
    """Chebyshev coefficients (in t = 2 x - 1) of Q_0^bfs ... Q_{n-1}^bfs,
    shape (n_terms, n_terms), rows are the polynomials."""

    # Gauss-Chebyshev nodes on [-1, 1]: (1/N) sum f(u_k) = (1/pi) int f/sqrt(1-u^2),
    # exact for polynomials of degree < 2N
    N = 4*n_terms + 16
    u = np.cos(np.pi*(np.arange(N) + 0.5)/N)
    x = u**2

    def slope(coef):
        # d/du [u^2 (1 - u^2) q(u^2)] = (2u - 4u^3) q + u^2 (1 - u^2) q'(x) 2u
        q = C.chebval(_x_to_t(x), coef)
        dq = 2*C.chebval(_x_to_t(x), C.chebder(coef))  # dq/dx = 2 dq/dt
        return (2*u - 4*u**3)*q + 2*u**3*(1 - u**2)*dq

    def inner(a, b):
        # (2/pi) int_0^1 ... = (1/pi) int_{-1}^{1} (even integrand)
        return np.mean(slope(a)*slope(b))

    rows = []
    for m in range(n_terms):
        v = np.zeros(n_terms)
        v[m] = 1.0
        for _ in range(2):  # Gram-Schmidt, twice for stability
            for r in rows:
                v = v - inner(v, r)*r
        v = v/np.sqrt(inner(v, v))
        if C.chebval(-1.0, v) < 0:  # sign: Q_m(0) > 0 (Forbes)
            v = -v
        rows.append(v)

    return np.array(rows)


def qbfs_series(a):
    """Chebyshev coefficients (in t = 2 x - 1) of sum_m a_m Q_m^bfs(x)."""

    a = np.atleast_1d(np.asarray(a, dtype=float))
    if len(a) == 0:
        return np.zeros(1)
    return a @ _qbfs_coefficients(len(a))


def qbfs(m, x):
    """Forbes' slope-orthonormal polynomial Q_m^bfs at x = u**2.

    Parameters
    ----------
    m : int
        Order (degree in x).
    x : array_like
        Squared normalized radius u**2.

    Returns
    -------
    Q : array
    """

    return C.chebval(_x_to_t(x), _qbfs_coefficients(m + 1)[m])


def qcon(m, x):
    """Forbes' sag-orthogonal polynomial Q_m^con(x) = P_m^(0,4)(2 x - 1).

    Parameters
    ----------
    m : int
        Order (degree in x).
    x : array_like
        Squared normalized radius u**2.

    Returns
    -------
    Q : array
    """

    return eval_jacobi(m, 0, 4, _x_to_t(x))


@lru_cache(maxsize=None)
def _qcon_coefficients(n_terms):
    """Chebyshev coefficients (in t) of Q_0^con ... Q_{n-1}^con."""

    t = np.cos(np.pi*(np.arange(n_terms) + 0.5)/n_terms)
    rows = [C.chebfit(t, eval_jacobi(m, 0, 4, t), n_terms - 1) for m in range(n_terms)]

    return np.array(rows)


def qcon_series(a):
    """Chebyshev coefficients (in t = 2 x - 1) of sum_m a_m Q_m^con(x)."""

    a = np.atleast_1d(np.asarray(a, dtype=float))
    if len(a) == 0:
        return np.zeros(1)
    return a @ _qcon_coefficients(len(a))


def _sphere(c, K, r2):
    with np.errstate(invalid="ignore"):
        return c*r2/(1 + np.sqrt(1 - (1 + K)*c**2*r2))


def sag_qbfs(r, c_bfs, norm_radius, a):
    """Sag of a Q-bfs asphere (see the module docstring).

    Parameters
    ----------
    r : array_like
        Radial coordinate.
    c_bfs : number
        Curvature of the best fit sphere.
    norm_radius : number
        Normalization radius r_norm (u = r/r_norm).
    a : sequence
        Coefficients a_0, a_1, ... (lengths).

    Returns
    -------
    z : array
    """

    return QbfsSurface(1/c_bfs if c_bfs else np.inf, 0.0, a, norm_radius).sag(np.asarray(r)**2)


def sag_qcon(r, c, K, norm_radius, a):
    """Sag of a Q-con asphere (see the module docstring).

    Parameters
    ----------
    r : array_like
        Radial coordinate.
    c, K : numbers
        Vertex curvature and conic constant.
    norm_radius : number
        Normalization radius r_norm (u = r/r_norm).
    a : sequence
        Coefficients a_0, a_1, ... (lengths).

    Returns
    -------
    z : array
    """

    return QconSurface(1/c if c else np.inf, 0.0, a, norm_radius, conic=K).sag(np.asarray(r)**2)


def even_asphere_sag(r, c, K=0.0, aspheric=()):
    """Sag of the even asphere of `raytrace.Surface`: conic plus A4 r**4 + A6 r**6 + ...

    Parameters
    ----------
    r : array_like
        Radial coordinate.
    c : number
        Vertex curvature.
    K : number, optional
        Conic constant.
    aspheric : sequence, optional
        (A4, A6, ...).

    Returns
    -------
    z : array
    """

    return Surface(1/c if c else np.inf, conic=K, aspheric=aspheric).sag(np.asarray(r, float)**2)


class _QSurface(Surface):
    """Common part of the Q surfaces: Newton intersection on the full sag."""

    def _series(self):
        raise NotImplementedError

    def _base(self, r2):
        raise NotImplementedError

    def intersect(self, P, d):
        """Distance along the rays to the surface: from the base sphere or
        conic by Newton steps on the full sag."""

        base = Surface(self.R, conic=self._base_conic)
        t = base.intersect(P, d)
        t = np.where(np.isfinite(t), t, -P[:, 2]/d[:, 2])
        prev = np.full(len(t), np.inf)
        for _ in range(60):
            Q = P + t[:, None]*d
            r2 = Q[:, 0]**2 + Q[:, 1]**2
            G = Q[:, 2] - self.sag(r2)
            dG = d[:, 2] - self._slope_over_r(r2)*(Q[:, 0]*d[:, 0] + Q[:, 1]*d[:, 1])
            with np.errstate(invalid="ignore", divide="ignore"):
                step = G/dG
            t = t - step
            tol = 1 + np.abs(t)
            with np.errstate(invalid="ignore"):
                done = (~np.isfinite(step) | (np.abs(step) < 1e-15*tol)
                        | ((np.abs(step) >= prev) & (np.abs(step) < 1e-12*tol)))
            if np.all(done):
                break
            prev = np.abs(step)
        Q = P + t[:, None]*d

        return np.where(np.isfinite(self.sag(Q[:, 0]**2 + Q[:, 1]**2)), t, np.nan)

    def normal(self, Q):
        """Unit normals (pointing into +z) at the points Q relative to the vertex."""

        s = self._slope_over_r(Q[:, 0]**2 + Q[:, 1]**2)
        N = np.stack([-Q[:, 0]*s, -Q[:, 1]*s, np.ones(len(Q))], axis=-1)

        return N/_norm(N)[:, None]

    def abcd_matrix(self, n1, n2, sigma):
        """Unfolded ABCD matrix with the vertex curvature (see `Surface.abcd_matrix`)."""

        R = 1/self.c if self.c != 0 else np.inf
        if self.mirror:
            return abcd.mirror(-sigma*R, n=n1)

        return abcd.refraction(sigma*R, n1, n2)

    def departure(self, r):
        """Sag minus the base sphere (Q-bfs) or base conic (Q-con) at the radii r."""

        r2 = np.asarray(r, float)**2
        return self.sag(r2) - self._base(r2)


class QbfsSurface(_QSurface):
    """Forbes Q-bfs asphere: best fit sphere plus a slope-orthogonal
    departure (see the module docstring).

    The attribute R is the radius of the best fit sphere (the sphere through
    the vertex and the surface at r_norm); `c` is the vertex curvature,
    which the paraxial data use. The surface is defined for r < 1/abs(c_bfs).

    Parameters
    ----------
    R : number
        Radius of the best fit sphere (np.inf: plane).
    thickness : number
        As for `raytrace.Surface`.
    q : sequence
        Coefficients a_0, a_1, ... in length units.
    norm_radius : number
        Normalization radius r_norm (usually the clear aperture radius).
    **kwargs
        Passed on to `raytrace.Surface` (n_after, material, semi_diameter,
        mirror, decenter, tilt, ...); conic and aspheric are not allowed.
    """

    _base_conic = 0.0

    def __init__(self, R=np.inf, thickness=0.0, q=(), norm_radius=1.0, **kwargs):
        if kwargs.get("conic") or kwargs.get("aspheric"):
            raise ValueError("A Q-bfs surface has no conic or aspheric terms.")
        super().__init__(R, thickness, **kwargs)
        self.q = tuple(float(v) for v in q)
        self.norm_radius = float(norm_radius)

    def __repr__(self):
        return "QbfsSurface(R={}, thickness={}, q={}, norm_radius={})".format(
            self.R, self.thickness, self.q, self.norm_radius)

    @property
    def c_bfs(self):
        """Curvature of the best fit sphere."""

        return 0.0 if np.isinf(self.R) else 1.0/self.R

    @property
    def c(self):
        """Vertex curvature, c_bfs + 2 sum_m a_m Q_m(0)/r_norm**2."""

        s0 = C.chebval(-1.0, qbfs_series(self.q)) if self.q else 0.0
        return self.c_bfs + 2*s0/self.norm_radius**2

    def _base(self, r2):
        return _sphere(self.c_bfs, 0.0, r2)

    def sag(self, r2):
        """Sag z at the squared radius r2 = x**2 + y**2."""

        r2 = np.asarray(r2, dtype=float)
        z = self._base(r2)
        if not self.q:
            return z
        cb = self.c_bfs
        x = r2/self.norm_radius**2
        with np.errstate(invalid="ignore"):
            root = np.sqrt(1 - cb**2*r2)
        S = C.chebval(_x_to_t(x), qbfs_series(self.q))

        return z + x*(1 - x)*S/root

    def _slope_over_r(self, r2):
        """(dz/dr)/r."""

        r2 = np.asarray(r2, dtype=float)
        cb = self.c_bfs
        with np.errstate(invalid="ignore"):
            root = np.sqrt(1 - cb**2*r2)
            s = cb/root
        if not self.q:
            return s
        rn2 = self.norm_radius**2
        x = r2/rn2
        coef = qbfs_series(self.q)
        S = C.chebval(_x_to_t(x), coef)
        dS = 2*C.chebval(_x_to_t(x), C.chebder(coef))       # dS/dx
        # D = x (1 - x) S / root; dD/dr = 2 r dD/d(r^2)
        # d/d(r2): [(1 - 2x) S + x (1 - x) dS]/rn2/root + x (1 - x) S cb^2/(2 root^3)
        with np.errstate(invalid="ignore"):
            dD = ((1 - 2*x)*S + x*(1 - x)*dS)/rn2/root + x*(1 - x)*S*cb**2/(2*root**3)

        return s + 2*dD

    def fourth_order_departure(self):
        """Coefficient of r**4 of the departure from the vertex sphere (Seidel sums)."""

        cb, rn = self.c_bfs, self.norm_radius
        coef = qbfs_series(self.q) if self.q else np.zeros(1)
        s0 = C.chebval(-1.0, coef)
        s1 = 2*C.chebval(-1.0, C.chebder(coef)) if len(coef) > 1 else 0.0
        total4 = cb**3/8 + (s1 - s0)/rn**4 + s0*cb**2/(2*rn**2)

        return total4 - self.c**3/8

    def rms_slope(self):
        """Forbes' rms slope of the departure, sqrt(sum a_m**2)/r_norm (radians)."""

        return np.sqrt(np.sum(np.square(self.q)))/self.norm_radius


class QconSurface(_QSurface):
    """Forbes Q-con asphere: a conic plus u**4 times a sag-orthogonal series
    of Jacobi polynomials (see the module docstring).

    Parameters
    ----------
    R : number
        Vertex radius.
    thickness : number
        As for `raytrace.Surface`.
    q : sequence
        Coefficients a_0, a_1, ... in length units.
    norm_radius : number
        Normalization radius r_norm.
    conic : number, optional
        Conic constant K.
    **kwargs
        Passed on to `raytrace.Surface`; aspheric is not allowed.
    """

    def __init__(self, R=np.inf, thickness=0.0, q=(), norm_radius=1.0, conic=0.0, **kwargs):
        if kwargs.get("aspheric"):
            raise ValueError("A Q-con surface has no monomial aspheric terms.")
        super().__init__(R, thickness, conic=conic, **kwargs)
        self.q = tuple(float(v) for v in q)
        self.norm_radius = float(norm_radius)

    def __repr__(self):
        return "QconSurface(R={}, thickness={}, conic={}, q={}, norm_radius={})".format(
            self.R, self.thickness, self.conic, self.q, self.norm_radius)

    @property
    def _base_conic(self):
        return self.conic

    def _base(self, r2):
        return _sphere(self.c, self.conic, r2)

    def sag(self, r2):
        """Sag z at the squared radius r2 = x**2 + y**2."""

        r2 = np.asarray(r2, dtype=float)
        z = self._base(r2)
        if not self.q:
            return z
        x = r2/self.norm_radius**2

        return z + x**2*C.chebval(_x_to_t(x), qcon_series(self.q))

    def _slope_over_r(self, r2):
        """(dz/dr)/r."""

        r2 = np.asarray(r2, dtype=float)
        c, K = self.c, self.conic
        with np.errstate(invalid="ignore"):
            s = c/np.sqrt(1 - (1 + K)*c**2*r2)
        if not self.q:
            return s
        rn2 = self.norm_radius**2
        x = r2/rn2
        coef = qcon_series(self.q)
        S = C.chebval(_x_to_t(x), coef)
        dS = 2*C.chebval(_x_to_t(x), C.chebder(coef))

        return s + 2*(2*x*S + x**2*dS)/rn2

    def fourth_order_departure(self):
        """Coefficient of r**4 of the departure from the vertex sphere (Seidel sums)."""

        s0 = C.chebval(-1.0, qcon_series(self.q)) if self.q else 0.0

        return self.conic*self.c**3/8 + s0/self.norm_radius**4


def best_fit_sphere(sag, norm_radius):
    """Forbes' best fit sphere: the sphere through the vertex and the
    surface at r_norm.

    Parameters
    ----------
    sag : callable
        z(r) of the surface (z(0) = 0).
    norm_radius : number
        Normalization radius.

    Returns
    -------
    c_bfs : number
        Curvature of the sphere.
    """

    z = float(sag(np.array([norm_radius]))[0])
    # sphere through (0, 0) and (r, z): c = 2 z/(r^2 + z^2)
    return 2*z/(norm_radius**2 + z**2)


def _nodes(n):
    """Radii u in (0, 1) for the fits: Gauss-Chebyshev nodes (u = cos)."""

    k = np.arange(n)
    return np.cos(np.pi*(k + 0.5)/(2*n))


def fit_qbfs(sag, norm_radius, n_terms=10, c_bfs=None):
    """Q-bfs coefficients of a rotationally symmetric sag.

    The coefficients are the projections of the slope of the normal
    departure from the base sphere onto the orthonormal slopes of the
    Q-bfs terms (Gauss-Chebyshev quadrature, slopes by central
    differences), i.e. the best fit in Forbes' slope norm. Adding terms
    leaves the earlier coefficients unchanged; for a sag that is a Q-bfs
    series of at most n_terms the fit is exact.

    Parameters
    ----------
    sag : callable
        z(r) (array in, array out), e.g. ``lambda r: even_asphere_sag(r, c, K, A)``.
    norm_radius : number
        Normalization radius r_norm.
    n_terms : int, optional
        Number of coefficients.
    c_bfs : number, optional
        Curvature of the base sphere; default `best_fit_sphere()`.

    Returns
    -------
    surface : QbfsSurface
        With R = 1/c_bfs, the coefficients and norm_radius (thickness 0).
    """

    if c_bfs is None:
        c_bfs = best_fit_sphere(sag, norm_radius)

    def D(u):
        r = u*norm_radius
        with np.errstate(invalid="ignore"):
            return (np.asarray(sag(r), float) - _sphere(c_bfs, 0.0, r**2))*np.sqrt(1 - c_bfs**2*r**2)

    # slope projection with Gauss-Chebyshev quadrature: a_m = <D', S_m'>
    n = 16*n_terms + 64
    u = _nodes(n)
    h = 1e-5
    dD = (D(u + h) - D(u - h))/(2*h)
    t = _x_to_t(u**2)
    a = []
    for m in range(n_terms):
        coef = _qbfs_coefficients(n_terms)[m]
        q = C.chebval(t, coef)
        dq = 2*C.chebval(t, C.chebder(coef))
        dS = (2*u - 4*u**3)*q + 2*u**3*(1 - u**2)*dq
        a.append(np.mean(dD*dS))
    R = 1/c_bfs if c_bfs else np.inf

    return QbfsSurface(R, 0.0, a, norm_radius)


def fit_qcon(sag, norm_radius, c, K=0.0, n_terms=10):
    """Q-con coefficients of a rotationally symmetric sag on a given conic.

    Parameters
    ----------
    sag : callable
        z(r).
    norm_radius : number
        Normalization radius r_norm.
    c, K : numbers
        Vertex curvature and conic constant of the base conic.
    n_terms : int, optional
        Number of coefficients.

    Returns
    -------
    surface : QconSurface
    """

    u = _nodes(8*n_terms + 32)
    r = u*norm_radius
    x = u**2
    D = np.asarray(sag(r), float) - _sphere(c, K, r**2)
    A = np.stack([x**2*qcon(m, x) for m in range(n_terms)], axis=1)
    a = np.linalg.lstsq(A, D, rcond=None)[0]

    return QconSurface(1/c if c else np.inf, 0.0, a, norm_radius, conic=K)


def rms_slope(sag, norm_radius, c_bfs=None, n=2000):
    """Forbes-weighted rms slope of the departure of a sag from its best fit sphere.

    Parameters
    ----------
    sag : callable
        z(r).
    norm_radius : number
        Normalization radius.
    c_bfs : number, optional
        Base sphere curvature (default `best_fit_sphere()`).
    n : int, optional
        Number of quadrature nodes.

    Returns
    -------
    slope : number
        sqrt((2/pi) int_0^1 (dD/dr)^2 du/sqrt(1 - u^2)) of the normal
        departure D (radians).
    """

    if c_bfs is None:
        c_bfs = best_fit_sphere(sag, norm_radius)
    u = _nodes(n)
    h = 1e-6
    r = u*norm_radius

    def D(r):
        with np.errstate(invalid="ignore"):
            return (np.asarray(sag(r), float) - _sphere(c_bfs, 0.0, r**2))*np.sqrt(1 - c_bfs**2*r**2)

    dD = (D(r + h*norm_radius) - D(r - h*norm_radius))/(2*h*norm_radius)

    return np.sqrt(np.mean(dD**2))



# ---------------------------------------------------------------------------
# Q2D freeforms
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _q2d_coefficients(m, n_terms):
    """Chebyshev coefficients (in t = 2 x - 1) of the radial polynomials
    P_0^m ... P_{n-1}^m of the azimuthal order m >= 1, shape (n_terms,
    n_terms): the terms u**m P_n^m(u**2) cos(m phi) are orthonormal in the
    gradient with the weight of Q-bfs (see `q2d()`)."""

    N = 4*(n_terms + m) + 16
    u = np.cos(np.pi*(np.arange(N) + 0.5)/N)
    x = u**2

    def parts(coef):
        # S = u^m P(x): dS/du = m u^(m-1) P + 2 u^(m+1) P'(x); S/u = u^(m-1) P
        P = C.chebval(_x_to_t(x), coef)
        dP = 2*C.chebval(_x_to_t(x), C.chebder(coef)) if len(coef) > 1 else 0.0*P
        return m*u**(m - 1)*P + 2*u**(m + 1)*dP, m*u**(m - 1)*P

    def inner(a, b):
        # (1/pi) int_0^1 [S_a' S_b' + m^2 S_a S_b/u^2] du/sqrt(1 - u^2)
        da, sa = parts(a)
        db, sb = parts(b)
        return 0.5*np.mean(da*db + sa*sb)

    rows = []
    for k in range(n_terms):
        v = np.zeros(n_terms)
        v[k] = 1.0
        for _ in range(2):
            for r in rows:
                v = v - inner(v, r)*r
        v = v/np.sqrt(inner(v, v))
        if C.chebval(1.0, v) < 0:  # sign: positive at the rim
            v = -v
        rows.append(v)

    return np.array(rows)


def q2d(m, n, x):
    """Radial polynomial Q_n^m(x) of the Q2D freeform terms at x = u**2.

    For m = 0 it is Forbes' Q-bfs polynomial (the term is u**2 (1 - u**2)
    Q_n^bfs(u**2)); for m >= 1 the terms u**m Q_n^m(u**2) cos(m phi) and
    u**m Q_n^m(u**2) sin(m phi) are built here by Gram-Schmidt
    orthonormalization so that, like the Q-bfs terms, their gradients are
    orthonormal in ::

        <f, g> = (1/pi**2) int int grad f . grad g du dphi/sqrt(1 - u**2),

    which for m = 0 is the Q-bfs inner product. The sum of the squared
    coefficients is then the mean square slope of the whole departure
    (times r_norm**2), as in Forbes' Q2D freeform description (Forbes 2012);
    his published m >= 1 polynomials come from a recurrence and may differ
    from these in normalization.

    Parameters
    ----------
    m : int
        Azimuthal order.
    n : int
        Radial order (degree in x).
    x : array_like
        Squared normalized radius u**2.

    Returns
    -------
    Q : array
    """

    if m == 0:
        return qbfs(n, x)

    return C.chebval(_x_to_t(x), _q2d_coefficients(m, n + 1)[n])


class Q2DSurface(Surface):
    """Q2D freeform: a best fit sphere plus a slope-orthonormal departure
    with azimuthal terms (Forbes 2012) ::

        z = c_bfs r**2/(1 + sqrt(1 - c_bfs**2 r**2)) + D(u, phi)/sqrt(1 - c_bfs**2 r**2),
        D = u**2 (1 - u**2) sum_n a_n^0 Q_n^0(u**2)
            + sum_{m>=1} u**m sum_n [a_n^m cos(m phi) + b_n^m sin(m phi)] Q_n^m(u**2),

    with u = r/r_norm, the azimuth phi counted from +x towards +y in the
    local frame of the surface, and the polynomials of `q2d()`. The m = 0
    terms are those of `QbfsSurface`; m = 1 holds tilt and coma-like
    terms, m = 2 astigmatism-like ones (a cylinder departure), m = 3
    trefoil, and so on. The paraxial data use the vertex curvature of the
    rotationally symmetric part (as for `QbfsSurface`); the other terms only
    enter the real ray trace, which is done with NumPy (Newton steps on the
    full sag).

    Parameters
    ----------
    R : number
        Radius of the base sphere.
    thickness : number
    cos, sin : dict, optional
        {(m, n): coefficient} of the cos(m phi) and sin(m phi) terms, in
        length units (m = 0 only in cos).
    norm_radius : number
        Normalization radius r_norm.
    **kwargs
        Passed on to `raytrace.Surface` (no conic or aspheric terms).
    """

    _base_conic = 0.0

    def __init__(self, R=np.inf, thickness=0.0, cos=None, sin=None, norm_radius=1.0, **kwargs):
        if kwargs.get("conic") or kwargs.get("aspheric"):
            raise ValueError("A Q2D surface has no conic or aspheric terms.")
        super().__init__(R, thickness, **kwargs)
        self.norm_radius = float(norm_radius)
        self.cos = tuple(sorted(((int(m), int(n)), float(v)) for (m, n), v in (cos or {}).items()))
        self.sin = tuple(sorted(((int(m), int(n)), float(v)) for (m, n), v in (sin or {}).items()
                                if m > 0))

    def __repr__(self):
        return "Q2DSurface(R={}, thickness={}, cos={}, sin={}, norm_radius={})".format(
            self.R, self.thickness, dict(self.cos), dict(self.sin), self.norm_radius)

    def set_term(self, kind, m, n, value):
        """Set one coefficient (in place); kind is "cos" or "sin"."""

        terms = dict(getattr(self, kind))
        terms[(int(m), int(n))] = float(value)
        setattr(self, kind, tuple(sorted(terms.items())))

    def get_term(self, kind, m, n):
        """One coefficient (0 if absent)."""

        return dict(getattr(self, kind)).get((int(m), int(n)), 0.0)

    @property
    def c_bfs(self):
        """Curvature of the base sphere."""

        return 0.0 if np.isinf(self.R) else 1.0/self.R

    def _axial(self):
        return [self.get_term("cos", 0, n) for n in range(1 + max(
            [n for (m, n), _ in self.cos if m == 0], default=-1))]

    @property
    def c(self):
        """Vertex curvature of the rotationally symmetric part."""

        q = self._axial()
        s0 = C.chebval(-1.0, qbfs_series(q)) if q else 0.0
        return self.c_bfs + 2*s0/self.norm_radius**2

    def fourth_order_departure(self):
        """r**4 term of the rotationally symmetric part (Seidel sums)."""

        return QbfsSurface(self.R, 0.0, self._axial(), self.norm_radius).fourth_order_departure()

    def abcd_matrix(self, n1, n2, sigma):
        """Unfolded ABCD matrix with the vertex curvature."""

        return _QSurface.abcd_matrix(self, n1, n2, sigma)

    def _groups(self):
        """{(kind, m): Chebyshev series of the radial polynomial}."""

        out = {}
        for kind in ("cos", "sin"):
            by_m = {}
            for (m, n), v in getattr(self, kind):
                by_m.setdefault(m, {})[n] = v
            for m, terms in by_m.items():
                a = np.zeros(max(terms) + 1)
                for n, v in terms.items():
                    a[n] = v
                out[kind, m] = qbfs_series(a) if m == 0 else a @ _q2d_coefficients(m, len(a))
        return out

    def _departure_and_gradient(self, x, y):
        """D and its gradient (dD/dx, dD/dy) in the local frame."""

        rn = self.norm_radius
        w = (np.asarray(x, float) + 1j*np.asarray(y, float))/rn
        X = np.abs(w)**2
        t = _x_to_t(X)
        D = np.zeros(w.shape)
        Dx, Dy = np.zeros(w.shape), np.zeros(w.shape)
        groups = self._groups()
        if not groups:
            return D, Dx, Dy
        # one Chebyshev Vandermonde matrix for all groups, powers of w by recurrence
        T = C.chebvander(t, max(len(c) for c in groups.values()) - 1)
        powers = [np.ones(w.shape, complex)]
        for _ in range(max(m for _, m in groups)):
            powers.append(powers[-1]*w)
        for (kind, m), coef in groups.items():
            P = T[..., :len(coef)] @ coef
            dP = (2*(T[..., :len(coef) - 1] @ C.chebder(coef)) if len(coef) > 1
                  else np.zeros(w.shape))   # dP/dX
            if m == 0:
                # X (1 - X) P(X)
                f = X*(1 - X)*P
                df = (1 - 2*X)*P + X*(1 - X)*dP
                D = D + f
                Dx, Dy = Dx + df*2*w.real/rn, Dy + df*2*w.imag/rn
                continue
            wm = powers[m]
            wm1 = m*powers[m - 1]
            if kind == "cos":
                A, Ax, Ay = wm.real, wm1.real, -wm1.imag
            else:
                A, Ax, Ay = wm.imag, wm1.imag, wm1.real
            D = D + P*A
            Dx = Dx + dP*2*w.real/rn*A + P*Ax/rn
            Dy = Dy + dP*2*w.imag/rn*A + P*Ay/rn

        return D, Dx, Dy

    def departure(self, x, y):
        """Normal departure D(x, y) from the base sphere (before the
        division by the square root)."""

        return self._departure_and_gradient(x, y)[0]

    def sag_xy(self, x, y):
        """Sag z(x, y)."""

        x, y = np.asarray(x, float), np.asarray(y, float)
        r2 = x**2 + y**2
        cb = self.c_bfs
        with np.errstate(invalid="ignore"):
            root = np.sqrt(1 - cb**2*r2)
        return _sphere(cb, 0.0, r2) + self.departure(x, y)/root

    def sag(self, r2):
        """Sag of the rotationally symmetric part (base sphere and m = 0
        terms) at r2 (used for drawings and paraxial checks)."""

        return QbfsSurface(self.R, 0.0, self._axial(), self.norm_radius).sag(r2)

    def _gradient(self, x, y):
        return self._sag_and_gradient(x, y)[1:]

    def _sag_and_gradient(self, x, y):
        """Sag and its gradient from one evaluation of the polynomials."""

        x, y = np.asarray(x, float), np.asarray(y, float)
        cb = self.c_bfs
        r2 = x**2 + y**2
        with np.errstate(invalid="ignore"):
            root = np.sqrt(1 - cb**2*r2)
        D, Dx, Dy = self._departure_and_gradient(x, y)
        s = cb/root                       # (dz_sphere/dr)/r
        k = D*cb**2/root**3
        return _sphere(cb, 0.0, r2) + D/root, x*s + Dx/root + k*x, y*s + Dy/root + k*y

    def intersect(self, P, d):
        """Distance along the rays to the surface: from the base sphere by
        Newton steps on the full sag."""

        t = Surface(self.R).intersect(P, d)
        t = np.where(np.isfinite(t), t, -P[:, 2]/d[:, 2])
        for _ in range(60):
            Q = P + t[:, None]*d
            z, gx, gy = self._sag_and_gradient(Q[:, 0], Q[:, 1])
            G = Q[:, 2] - z
            dG = d[:, 2] - gx*d[:, 0] - gy*d[:, 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                step = G/dG
            t = t - step
            with np.errstate(invalid="ignore"):
                if np.all(~np.isfinite(step) | (np.abs(step) < 1e-14*(1 + np.abs(t)))):
                    break
        Q = P + t[:, None]*d

        return np.where(np.isfinite(self.sag_xy(Q[:, 0], Q[:, 1])), t, np.nan)

    def normal(self, Q):
        """Unit normals (pointing into +z) at the points Q relative to the vertex."""

        gx, gy = self._gradient(Q[:, 0], Q[:, 1])
        N = np.stack([-gx, -gy, np.ones(len(Q))], axis=-1)

        return N/_norm(N)[:, None]

    def rms_slope(self):
        """sqrt(sum of all squared coefficients)/r_norm (radians)."""

        v = [c for _, c in self.cos] + [c for _, c in self.sin]
        return np.sqrt(np.sum(np.square(v)))/self.norm_radius


class QconQ2DSurface(QconSurface):
    """Q-con asphere with a Q2D freeform departure centred off axis: an
    off-axis mirror segment of a rotationally symmetric parent with a
    freeform correction over its footprint ::

        z(x, y) = z_Qcon(x**2 + y**2) + D((x - x0)/r_f, (y - y0)/r_f),

    with the Q-con sag of `QconSurface` about the vertex and the Q2D
    departure D of `Q2DSurface` (same coefficients, polynomials and
    azimuth convention) centred on the point (x0, y0) of the local frame
    with its own normalization radius r_f, usually the footprint centre
    and radius (`off_axis.freeform_mirrors`). D is added to the sag as it
    is: it is not divided by the square root of the base sphere, since the
    base is not a sphere about (x0, y0). The paraxial data and the Seidel
    sums use the Q-con part; the departure only enters the real ray trace.

    Parameters
    ----------
    R, thickness, q, norm_radius, conic : optional
        As for `QconSurface` (norm_radius is that of the Q-con terms).
    cos, sin : dict, optional
        {(m, n): coefficient} of the Q2D terms, as for `Q2DSurface`.
    freeform_radius : number, optional
        Normalization radius r_f of the Q2D terms.
    centre : (x0, y0), optional
        Centre of the Q2D terms in the local frame.
    **kwargs
        Passed on to `raytrace.Surface`.
    """

    def __init__(self, R=np.inf, thickness=0.0, q=(), norm_radius=1.0, conic=0.0,
                 cos=None, sin=None, freeform_radius=1.0, centre=(0.0, 0.0), **kwargs):
        super().__init__(R, thickness, q, norm_radius, conic, **kwargs)
        self.freeform_radius = float(freeform_radius)
        self.centre = (float(centre[0]), float(centre[1]))
        self.cos = tuple(sorted(((int(m), int(n)), float(v)) for (m, n), v in (cos or {}).items()))
        self.sin = tuple(sorted(((int(m), int(n)), float(v)) for (m, n), v in (sin or {}).items()
                                if m > 0))

    def __repr__(self):
        return ("QconQ2DSurface(R={}, thickness={}, conic={}, q={}, norm_radius={}, cos={}, "
                "sin={}, freeform_radius={}, centre={})").format(
            self.R, self.thickness, self.conic, self.q, self.norm_radius, dict(self.cos),
            dict(self.sin), self.freeform_radius, self.centre)

    set_term = Q2DSurface.set_term
    get_term = Q2DSurface.get_term
    _groups = Q2DSurface._groups

    def departure_gradient(self, x, y):
        """Analytic (dD/dx, dD/dy) of the Q2D departure."""

        return self._departure(x, y)[1:]

    def _departure(self, x, y):
        x0, y0 = self.centre

        # the Q2D departure with its own normalization radius:
        patch = SimpleNamespace(norm_radius=self.freeform_radius, _groups=self._groups)

        return Q2DSurface._departure_and_gradient(
            patch, np.asarray(x, float) - x0, np.asarray(y, float) - y0)

    def departure_xy(self, x, y):
        """Q2D part of the sag at the points (x, y) of the local frame."""

        return self._departure(x, y)[0]

    def sag_xy(self, x, y):
        """Sag z(x, y), Q-con sag plus the Q2D departure."""

        x, y = np.asarray(x, float), np.asarray(y, float)
        return self.sag(x**2 + y**2) + self.departure_xy(x, y)

    def _gradient(self, x, y):
        return self._sag_and_gradient(x, y)[1:]

    def _sag_and_gradient(self, x, y):
        """Sag and its gradient from one evaluation of the polynomials."""

        x, y = np.asarray(x, float), np.asarray(y, float)
        r2 = x**2 + y**2
        s = self._slope_over_r(r2)
        D, gx, gy = self._departure(x, y)

        return self.sag(r2) + D, x*s + gx, y*s + gy

    def intersect(self, P, d):
        """Distance along the rays to the surface: from the Q-con surface by
        Newton steps on the full sag."""

        t = super().intersect(P, d)
        if not (self.cos or self.sin):
            return t
        t = np.where(np.isfinite(t), t, -P[:, 2]/d[:, 2])
        for _ in range(60):
            Q = P + t[:, None]*d
            z, gx, gy = self._sag_and_gradient(Q[:, 0], Q[:, 1])
            G = Q[:, 2] - z
            dG = d[:, 2] - gx*d[:, 0] - gy*d[:, 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                step = G/dG
            t = t - step
            with np.errstate(invalid="ignore"):
                if np.all(~np.isfinite(step) | (np.abs(step) < 1e-14*(1 + np.abs(t)))):
                    break
        Q = P + t[:, None]*d

        return np.where(np.isfinite(self.sag_xy(Q[:, 0], Q[:, 1])), t, np.nan)

    def normal(self, Q):
        """Unit normals (pointing into +z) at the points Q relative to the vertex."""

        gx, gy = self._gradient(Q[:, 0], Q[:, 1])
        N = np.stack([-gx, -gy, np.ones(len(Q))], axis=-1)

        return N/_norm(N)[:, None]

    def rms_slope(self):
        """sqrt(sum of the squared Q2D coefficients)/r_f (radians)."""

        v = [c for _, c in self.cos] + [c for _, c in self.sin]
        return np.sqrt(np.sum(np.square(v)))/self.freeform_radius


try:
    from .lens_optimization import Variable as _Variable
except ImportError:  # pragma: no cover
    _Variable = object


class QCoefficient(_Variable):
    """Q coefficient a_m of a `QbfsSurface` or `QconSurface`, a variable of
    `lens_optimization.DampedLeastSquares`.

    Parameters
    ----------
    surface : int
        Index of the surface.
    m : int
        Order of the coefficient.
    bounds : (lower, upper), optional
        Limits of the value.
    """

    def __init__(self, surface, m, bounds=(None, None)):
        super().__init__(bounds)
        self.surface, self.m = surface, m
        self.name = "q{}_{}".format(m, surface)

    def _coefficients(self, system):
        q = list(system.surfaces[self.surface].q)
        return q + [0.0]*(self.m + 1 - len(q))

    def get(self, system):
        """Value of the coefficient."""

        return self._coefficients(system)[self.m]

    def set(self, system, value):
        """Set the coefficient (in place)."""

        q = self._coefficients(system)
        q[self.m] = value
        system.surfaces[self.surface].q = tuple(q)

    def typical(self, system):
        """Typical change: 100 wavelengths."""

        return 100*system.wavelength


class Q2DCoefficient(_Variable):
    """Coefficient a_n^m (kind "cos") or b_n^m (kind "sin") of a
    `Q2DSurface`, a variable of `lens_optimization.DampedLeastSquares`.

    Parameters
    ----------
    surface : int
        Index of the surface.
    m, n : int
        Azimuthal and radial order.
    kind : {"cos", "sin"}, optional
    bounds : (lower, upper), optional
    """

    def __init__(self, surface, m, n, kind="cos", bounds=(None, None)):
        super().__init__(bounds)
        if kind not in ("cos", "sin") or (kind == "sin" and m == 0):
            raise ValueError("kind is 'cos' or 'sin' (sin needs m > 0).")
        self.surface, self.m, self.n, self.kind = surface, m, n, kind
        self.name = "{}{}_{}_{}".format("a" if kind == "cos" else "b", m, n, surface)

    def get(self, system):
        """Value of the coefficient."""

        return system.surfaces[self.surface].get_term(self.kind, self.m, self.n)

    def set(self, system, value):
        """Set the coefficient (in place)."""

        system.surfaces[self.surface].set_term(self.kind, self.m, self.n, value)

    def typical(self, system):
        """Typical change: 100 wavelengths."""

        return 100*system.wavelength
