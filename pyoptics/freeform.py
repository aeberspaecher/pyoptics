#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Freeform beam shaping with geometric optics: ray maps, optimal transport
and the construction of lens surfaces.

A refractive beam shaper redistributes the power of a beam: the ray that
leaves the source point x must land on the target point T(x) such that the
power is conserved, I_in(x) dx = E_out(T(x)) dT, i.e. ::

    E_out(T(x)) det(grad T(x)) = I_in(x)

(the Monge-Ampère equation when T is the gradient of a potential). Every
such map T is a ray map, and a surface realizes it if its normals refract
the rays accordingly (Snell's law) and are integrable, i.e. the normals
come from a single sag z(x, y). Two steps follow:

1. A ray map. With rotational symmetry, the map R(r) follows from equal
   enclosed power, int_0^r I_in 2 pi r dr = int_0^R E_out 2 pi R dR
   (`radial_map()`). In 2D there are infinitely many maps; the one of
   optimal transport for the quadratic cost (Brenier), T = grad phi with a
   convex phi, is the gradient of a potential and so gives nearly
   integrable surfaces. `ot_map()` computes it with entropic optimal
   transport (Sinkhorn iterations in the log domain, with the Gibbs kernel
   of a regular grid applied separably along x and y, and the barycentric
   projection of the transport plan).
2. The surfaces.

   - `flattop_shaper()`: the classic two-surface shaper (Kreuzer; Hoffnagle
     and Jefferson) that turns a collimated beam into another collimated
     beam with a new profile, e.g. a Gaussian into a flat top. The equal
     optical path between the two surfaces fixes their distance, Snell's
     law the slope of the first; both follow from the radial ray map in
     closed form.
   - `freeform_lens()`: a plano-freeform lens that sends a collimated beam
     to a target plane at the distance L. Snell's law gives the required
     surface gradient at every point, a least squares integration
     (`integrate_gradient()`, a Poisson equation solved with discrete
     cosine transforms) the sag, and a few fixed-point iterations account
     for the sag in the ray directions.

`TabulatedSurface` puts a sag given on a grid (radial or 2D, interpolated
by cubic splines) into `raytrace.System`, so the designs are checked by an
exact ray trace. `fermi_dirac()` is the flat-top family of Hoffnagle and
Jefferson whose soft edges survive propagation.

References
----------
J. L. Kreuzer, "Coherent light optical system yielding an output beam of
desired intensity distribution at a desired equiphase surface", US patent
3,476,463 (1969).

J. A. Hoffnagle and C. M. Jefferson, "Design and performance of a
refractive optical system that converts a Gaussian to a flattop beam",
Appl. Opt. 39, 5488-5499 (2000).

F. M. Dickey (ed.), "Laser Beam Shaping: Theory and Techniques", 2nd ed.,
CRC Press (2014).

C. Bösel and H. Gross, "Single freeform surface design for prescribed
input wavefront and target irradiance", J. Opt. Soc. Am. A 34, 1490-1499
(2017).

R. Wu, L. Xu, P. Liu, Y. Zhang, Z. Zheng, H. Li and X. Liu, "Freeform
illumination design: a nonlinear boundary problem for the elliptic
Monge-Ampère equation", Opt. Lett. 38, 229-231 (2013).

M. Cuturi, "Sinkhorn distances: lightspeed computation of optimal
transport", Advances in Neural Information Processing Systems 26 (2013).

G. Peyré and M. Cuturi, "Computational optimal transport", Foundations
and Trends in Machine Learning 11, 355-607 (2019).

R. T. Frankot and R. Chellappa, "A method for enforcing integrability in
shape from shading algorithms", IEEE Trans. Pattern Anal. Mach. Intell.
10, 439-451 (1988).
"""

import numpy as np
from scipy.fft import dctn, idctn
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.special import logsumexp

from .raytrace import Surface


__all__ = ["radial_map", "fermi_dirac", "flattop_shaper", "ot_map", "integrate_gradient",
           "freeform_lens", "TabulatedSurface"]


def radial_map(r, I_in, R, E_out):
    """Ray map R(r) of a rotationally symmetric beam shaper from equal
    enclosed power: int_0^r I_in 2 pi r dr = int_0^R E_out 2 pi R dR.

    Parameters
    ----------
    r : 1d array
        Radii of the input beam, increasing from 0.
    I_in : 1d array
        Input irradiance at r.
    R : 1d array
        Radii of the target, increasing from 0.
    E_out : 1d array
        Target irradiance at R (any scale: both are normalized to the same
        total power).

    Returns
    -------
    R_of_r : 1d array
        Target radius for every input radius.
    """

    def cumulative(x, y):
        c = np.concatenate([[0.0], np.cumsum(0.5*(y[1:]*x[1:] + y[:-1]*x[:-1])*np.diff(x))])
        return c/c[-1]

    P_in = cumulative(np.asarray(r, float), np.asarray(I_in, float))
    P_out = cumulative(np.asarray(R, float), np.asarray(E_out, float))
    return np.interp(P_in, P_out, R)


def fermi_dirac(R, R_0, beta):
    """Flat-top profile 1/(1 + exp(beta (R/R_0 - 1))) of Hoffnagle and
    Jefferson: flat inside R_0, with an edge of the width ~ 4 R_0/beta
    (from 88 % to 12 %); beta -> infinity is the sharp flat top."""

    with np.errstate(over="ignore"):
        return 1/(1 + np.exp(beta*(np.asarray(R, float)/R_0 - 1)))


def flattop_shaper(r, R, n, distance):
    """Two surfaces that turn a collimated beam into a collimated beam with
    the radial ray map R(r) (Kreuzer, Hoffnagle and Jefferson).

    The beam travels in glass of index n up to the first surface z_1(r)
    (vertex at z = 0), crosses an air gap and enters glass again at the
    second surface z = distance + z_2(R). The optical path between the
    planes before and after the surfaces is the same for all rays; with
    u = z_2(R) - z_1(r) and D = distance it reads
    sqrt((R - r)**2 + (D + u)**2) - n u = D, which gives ::

        u = (sqrt(D**2 (n - 1)**2 + (n**2 - 1)(R - r)**2) - D (n - 1))/(n**2 - 1).

    Snell's law at the first surface for the ray direction
    d = (R - r, D + u)/norm(R - r, D + u) in air gives its slope
    dz_1/dr = d_r/(n - d_z), integrated from the axis.

    Parameters
    ----------
    r : 1d array
        Input radii from 0.
    R : 1d array
        Target radii R(r) (e.g. from `radial_map()`).
    n : number
        Index of the two lenses (air in between).
    distance : number
        Distance D of the two vertices.

    Returns
    -------
    z_1 : 1d array
        Sag of the first surface at r.
    z_2 : 1d array
        Sag of the second surface at R.
    """

    r, R = np.asarray(r, float), np.asarray(R, float)
    D = float(distance)
    delta = R - r
    u = (np.sqrt(D**2*(n - 1)**2 + (n**2 - 1)*delta**2) - D*(n - 1))/(n**2 - 1)
    L = np.hypot(delta, D + u)
    slope = (delta/L)/(n - (D + u)/L)
    z_1 = np.concatenate([[0.0], np.cumsum(0.5*(slope[1:] + slope[:-1])*np.diff(r))])
    return z_1, z_1 + u


def _lse_kernel(h, x, y, eps):
    # log of sum_j exp(-(x_i - y_j)**2/(2 eps) + h_j) along the first axis of h
    c = -(x[:, None] - y[None, :])**2/(2*eps)                  # (n_x, n_y)
    return logsumexp(c[:, :, None] + h[None, :, :], axis=1)


def _apply(h, xs, ys, eps):
    # log K exp(h) for the separable Gibbs kernel of two regular grids,
    # h on the grid (ys[0], ys[1]), result on (xs[0], xs[1])
    A = _lse_kernel(h, xs[0], ys[0], eps)                       # (nx0, ny1)
    return _lse_kernel(A.T, xs[1], ys[1], eps).T                # (nx0, nx1)


def ot_map(a, x, b, y, eps=None, n_iter=300, eps_start=None, tol=1e-9):
    """Optimal transport map (quadratic cost) between two densities on
    regular 2D grids, by entropic optimal transport.

    Sinkhorn's iterations for the dual potentials f, g run in the log
    domain with the Gibbs kernel exp(-dist(x, y)**2/(2 eps)) applied
    separably (first along one axis, then the other), so one iteration
    costs O(n**3) for n x n grids. eps is lowered from eps_start to eps
    (epsilon scaling). The map is the barycentric projection of the
    transport plan, T(x_i) = sum_j P_ij y_j/sum_j P_ij, which tends to the
    Brenier map grad phi for eps -> 0; its blur is about sqrt(eps).

    Parameters
    ----------
    a : (n0, n1) array
        Source weights on the grid x (any scale, normalized here).
    x : (1d array, 1d array)
        Grid coordinates of the source along its two axes.
    b : (m0, m1) array
        Target weights on the grid y.
    y : (1d array, 1d array)
        Grid coordinates of the target.
    eps : number, optional
        Final regularization (default: a quarter of the product of the
        source and target grid spacings, which keeps the blur of the map
        below a target cell where the map expands the beam).
    n_iter : int, optional
        Iterations at the final eps.
    eps_start : number, optional
        Start of the epsilon scaling (default: a hundredth of the squared
        extent of the target).
    tol : number, optional
        Stop when the target marginal is matched to this relative accuracy.

    Returns
    -------
    Tx, Ty : (n0, n1) arrays
        Target coordinates of every source grid point.
    err : float
        Remaining relative error of the target marginal.
    """

    a = np.asarray(a, float)/np.sum(a)
    b = np.asarray(b, float)/np.sum(b)
    x = [np.asarray(v, float) for v in x]
    y = [np.asarray(v, float) for v in y]
    h_x = max(x[0][1] - x[0][0], x[1][1] - x[1][0])
    h_y = max(y[0][1] - y[0][0], y[1][1] - y[1][0])
    eps = 0.25*h_x*h_y if eps is None else eps
    extent = max(np.ptp(y[0]), np.ptp(y[1]))
    eps_start = max(eps, 0.01*extent**2) if eps_start is None else eps_start
    with np.errstate(divide="ignore"):
        la, lb = np.log(a), np.log(b)
    f = np.zeros_like(a)
    g = np.zeros_like(b)
    schedule = np.geomspace(eps_start, eps, max(2, int(np.ceil(np.log(eps_start/eps)/np.log(2))) + 1))
    err = np.inf
    for k, e in enumerate(schedule):
        for it in range(n_iter if k == len(schedule) - 1 else 20):
            f = -e*_apply(g/e + lb, x, y, e)
            g = -e*_apply(f/e + la, y, x, e)
            if k == len(schedule) - 1 and it % 10 == 9:
                # marginal of the plan on the target side
                lf = _apply(f/e + la, y, x, e) + g/e + lb
                err = np.max(np.abs(np.exp(lf) - b))/np.max(b)
                if err < tol:
                    break
    # barycentric projection: sum_j P_ij y_j with P_ij = exp((f_i + g_j - c_ij)/e) a_i b_j
    e = schedule[-1]
    lg = g/e + lb
    norm = _apply(lg, x, y, e)
    Y0, Y1 = np.meshgrid(y[0], y[1], indexing="ij")
    shift0 = y[0].min() - 1.0
    shift1 = y[1].min() - 1.0
    Tx = np.exp(_apply(lg + np.log(Y0 - shift0), x, y, e) - norm) + shift0
    Ty = np.exp(_apply(lg + np.log(Y1 - shift1), x, y, e) - norm) + shift1
    return Tx, Ty, float(err)


def integrate_gradient(gx, gy, dx, dy=None):
    """Least squares surface z with the gradient (gx, gy) on a regular grid
    (Frankot-Chellappa with Neumann boundaries): the Poisson equation
    laplace(z) = div(g) solved with discrete cosine transforms. The mean of
    z is zero; any curl of g is discarded.

    Parameters
    ----------
    gx, gy : (n0, n1) arrays
        dz/dx along axis 0 and dz/dy along axis 1.
    dx, dy : numbers
        Grid spacings along axis 0 and axis 1 (dy defaults to dx).

    Returns
    -------
    z : (n0, n1) array
    """

    dy = dx if dy is None else dy
    gx, gy = np.asarray(gx, float), np.asarray(gy, float)
    n0, n1 = gx.shape
    # least squares over the differences between neighbours, with the
    # gradient averaged onto the faces between them; the normal equations
    # are the Neumann Poisson problem laplace(z) = div(g)
    gxf = np.concatenate([np.zeros((1, n1)), 0.5*(gx[1:] + gx[:-1]), np.zeros((1, n1))])
    gyf = np.concatenate([np.zeros((n0, 1)), 0.5*(gy[:, 1:] + gy[:, :-1]), np.zeros((n0, 1))],
                         axis=1)
    rhs = np.diff(gxf, axis=0)/dx + np.diff(gyf, axis=1)/dy
    k0 = np.arange(n0)
    k1 = np.arange(n1)
    lam = ((2*np.cos(np.pi*k0/n0) - 2)/dx**2)[:, None] + ((2*np.cos(np.pi*k1/n1) - 2)/dy**2)[None, :]
    R = dctn(rhs, type=2, norm="ortho")
    lam[0, 0] = 1.0
    Z = R/lam
    Z[0, 0] = 0.0
    return idctn(Z, type=2, norm="ortho")


def freeform_lens(x, y, Tx, Ty, n, distance, iterations=4):
    """Freeform surface of a plano-freeform lens that sends a collimated
    beam (along +z, inside glass of index n) through the surface into air
    and on to the points (Tx, Ty) of the target plane z = distance.

    For the sag z(x, y), the ray from (x, y, z) to (Tx, Ty, distance) has
    the direction d; Snell's law requires the surface gradient
    (d_x, d_y)/(n - d_z), which is integrated by least squares
    (`integrate_gradient()`). The sag enters d, so this is repeated a few
    times starting from z = 0.

    Parameters
    ----------
    x, y : 1d arrays
        Regular grid of the lens aperture (axis 0 and axis 1).
    Tx, Ty : (len(x), len(y)) arrays
        Target points of the rays through the grid points (a ray map, e.g.
        from `ot_map()`).
    n : number
        Index of the lens.
    distance : number
        Distance of the target plane from the vertex plane of the surface.
    iterations : int, optional

    Returns
    -------
    z : (len(x), len(y)) array
        Sag, zero at the centre of the grid (vertex).
    """

    X, Y = np.meshgrid(x, y, indexing="ij")
    z = np.zeros_like(X)
    for _ in range(iterations):
        dvec = np.stack([Tx - X, Ty - Y, distance - z])
        dvec /= np.linalg.norm(dvec, axis=0)
        gx = dvec[0]/(n - dvec[2])
        gy = dvec[1]/(n - dvec[2])
        z = integrate_gradient(gx, gy, x[1] - x[0], y[1] - y[0])
        z -= RectBivariateSpline(x, y, z)(0.0, 0.0)[0, 0]
    return z


class TabulatedSurface(Surface):
    """Surface whose sag is given on a grid and interpolated with cubic
    splines: rotationally symmetric (z(r) on radii from 0) or a 2D freeform
    (z(x, y) on a rectangular grid), for `raytrace.System`.

    Rays are intersected by Newton steps from the vertex plane, like
    `raytrace.ZernikeSurface`, and refracted with the spline normals.
    Points outside the table are lost. The paraxial data of the system see
    a plane.

    Parameters
    ----------
    r : 1d array, optional
        Radii (from 0) for a rotationally symmetric sag.
    z : array
        Sag at r, or on the grid (x, y).
    x, y : 1d arrays, optional
        Grid of a 2D sag (axis 0 is x).
    thickness, n_after, semi_diameter, ... : optional
        As for `raytrace.Surface`.
    """

    def __init__(self, r=None, z=None, x=None, y=None, thickness=0.0, **kwargs):
        super().__init__(np.inf, thickness, **kwargs)
        if r is not None:
            r = np.asarray(r, float)
            z = np.asarray(z, float)
            # even extension keeps the slope zero on the axis
            self._radial = CubicSpline(np.r_[-r[:0:-1], r], np.r_[z[:0:-1], z])
            self._r_max = r[-1]
            self._spline = None
        else:
            self._spline = RectBivariateSpline(np.asarray(x, float), np.asarray(y, float),
                                               np.asarray(z, float), kx=3, ky=3)
            self._box = (x[0], x[-1], y[0], y[-1])
            self._radial = None

    def __repr__(self):
        return "TabulatedSurface(thickness={}, {})".format(
            self.thickness, "radial" if self._radial is not None else "2D")

    def _inside(self, x, y):
        if self._radial is not None:
            return np.hypot(x, y) <= self._r_max
        x0, x1, y0, y1 = self._box
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    def sag_xy(self, x, y):
        """Sag z(x, y) (NaN outside the table)."""

        x, y = np.asarray(x, float), np.asarray(y, float)
        if self._radial is not None:
            z = self._radial(np.hypot(x, y))
        else:
            z = self._spline(x, y, grid=False)
        return np.where(self._inside(x, y), z, np.nan)

    def _gradient(self, x, y):
        if self._radial is not None:
            r = np.hypot(x, y)
            s = self._radial(r, 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                f = np.where(r > 0, s/np.where(r > 0, r, 1.0), 0.0)
            return x*f, y*f
        return self._spline(x, y, dx=1, grid=False), self._spline(x, y, dy=1, grid=False)

    def intersect(self, P, d):
        """Distance along the rays to the surface (Newton steps)."""

        t = -P[:, 2]/d[:, 2]
        for _ in range(60):
            Q = P + t[:, None]*d
            xq = np.clip(Q[:, 0], -1e12, 1e12)
            yq = np.clip(Q[:, 1], -1e12, 1e12)
            gx, gy = self._gradient(xq, yq)
            if self._radial is not None:
                zq = self._radial(np.hypot(xq, yq))
            else:
                zq = self._spline(xq, yq, grid=False)
            G = Q[:, 2] - zq
            dG = d[:, 2] - gx*d[:, 0] - gy*d[:, 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                step = G/dG
            t = t - step
            with np.errstate(invalid="ignore"):
                if np.all(~np.isfinite(step) | (np.abs(step) < 1e-13*(1 + np.abs(t)))):
                    break
        Q = P + t[:, None]*d
        return np.where(self._inside(Q[:, 0], Q[:, 1]), t, np.nan)

    def normal(self, Q):
        """Unit normals (pointing into +z) at the points Q."""

        gx, gy = self._gradient(Q[:, 0], Q[:, 1])
        N = np.stack([-gx, -gy, np.ones(len(Q))], axis=-1)
        return N/np.linalg.norm(N, axis=1)[:, None]
