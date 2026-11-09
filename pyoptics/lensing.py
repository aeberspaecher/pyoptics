#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Gravitational lensing: a mass as a gradient-index medium and as a thin
lens, Einstein rings, arcs, caustics and magnification (notebook 99).

Weak field: in the isotropic coordinates of a static mass the light rays
are those of a medium with the refractive index n = 1 - 2 Phi/c^2, for a
point mass n = 1 + R_s/r with the Schwarzschild radius R_s = 2 G M/c^2.
A ray with the impact parameter b is deflected by alpha = 2 R_s/b =
4 G M/(c^2 b), twice the Newtonian value. The exact Schwarzschild metric
is an isotropic medium too, n = (1 + R_s/(4 rho))^3/(1 - R_s/(4 rho))
(`schwarzschild_medium()`), which captures rays with b < 3 sqrt(3) R_s/2
and deflects the others by 2 R_s/b + 15 pi R_s^2/(16 b^2) + ... Both can
be traced with `grin.trace()`.

Thin lens: when the deflection happens in a thin layer, a source at the
angle beta is seen at the angles theta that solve the lens equation
beta = theta - alpha(theta), with the scaled deflection angle alpha (the
true deflection times D_ls/D_s). Angles here are in units of the Einstein
radius theta_E = sqrt(4 G M D_ls/(c^2 D_l D_s)) unless stated, and
lenses are objects with deflection(theta) and potential(theta) for
angles of the shape (..., 2):

- `PointMass`: alpha = theta_E^2 theta/abs(theta)^2. Two images at
  (u +- sqrt(u^2 + 4))/2 for the source at u = beta/theta_E, with the
  magnifications (u^2 + 2)/(2 u sqrt(u^2 + 4)) +- 1/2 and the total
  (u^2 + 2)/(u sqrt(u^2 + 4)) (`point_mass_images()`, `paczynski()`);
  a uniform disk source of radius rho centred behind the lens is
  magnified by sqrt(1 + 4/rho^2).
- `SIS` (singular isothermal sphere, a galaxy with a flat rotation
  curve): alpha = theta_E theta/abs(theta), images at beta +- theta_E
  with the magnifications abs(theta)/(abs(theta) - theta_E)
  (`sis_images()`), total 2 theta_E/beta inside the Einstein radius.
- `ExternalShear`: the tidal field of the surroundings, potential
  gamma/2 (x^2 - y^2) rotated by an angle. With an SIS it turns the
  Einstein ring into an astroid caustic (four images: the Einstein
  cross) with cusps at 2 gamma theta_E/(1 + gamma) and
  2 gamma theta_E/(1 - gamma) on its axes.
- `Composite`: sum of lenses.

The Fermat potential tau(theta) = abs(theta - beta)^2/2 - psi(theta)
(`fermat_potential()`) is the arrival time up to a constant factor; the
images are its stationary points (minima, saddles, maxima), and for the
SIS the delay between the two images is 2 beta theta_E.

Functions:

- schwarzschild_radius(), einstein_radius(): physical scales (SI units).
- weak_field_medium(), schwarzschild_medium(): GRIN media for grin.trace();
  deflection_weak(), deflection_second_order(), schwarzschild_deflection().
- lens_map(), jacobian(), magnification(), find_images(),
  critical_curve(), caustic(), fermat_potential(), ray_shoot(): thin lens
  tools for any lens.
- point_mass_images(), sis_images(), paczynski(),
  finite_source_magnification(), sis_shear_cusps(): closed forms.

References
----------
A. Einstein, "Lens-like action of a star by the deviation of light in the
gravitational field", Science 84, 506-507 (1936).

S. Refsdal, "The gravitational lens effect", Mon. Not. R. Astron. Soc.
128, 307-310 (1964).

B. Paczyński, "Gravitational microlensing by the galactic halo",
Astrophys. J. 304, 1-5 (1986).

R. Kormann, P. Schneider and M. Bartelmann, "Isothermal elliptical
gravitational lens models", Astron. Astrophys. 284, 285-299 (1994).

P. Schneider, J. Ehlers and E. E. Falco, "Gravitational Lenses",
Springer (1992).

H. J. Witt and S. Mao, "Can lensed stars be regarded as pointlike for
microlensing by MACHOs?", Astrophys. J. 430, 505-510 (1994).

R. Narayan and M. Bartelmann, "Lectures on gravitational lensing",
arXiv:astro-ph/9606001 (1996).
"""

import numpy as np

from .grin import IndexField


__all__ = ["G", "C", "M_SUN", "PC", "KPC", "MPC", "schwarzschild_radius", "einstein_radius",
           "weak_field_medium", "schwarzschild_medium", "deflection_weak",
           "deflection_second_order", "schwarzschild_deflection", "PointMass", "SIS", "ExternalShear", "Composite",
           "lens_map", "jacobian", "magnification", "find_images", "critical_curve", "caustic",
           "fermat_potential", "ray_shoot", "point_mass_images", "sis_images", "paczynski",
           "finite_source_magnification", "sis_shear_cusps"]

G = 6.67430e-11          # m^3 kg^-1 s^-2
C = 299792458.0          # m/s
M_SUN = 1.98892e30       # kg
PC = 3.0856775814913673e16   # m
KPC = 1e3*PC
MPC = 1e6*PC


def schwarzschild_radius(M):
    """Schwarzschild radius 2 G M/c^2 (m) of the mass M (kg)."""
    return 2*G*M/C**2


def einstein_radius(M, D_l, D_s, D_ls=None):
    """Einstein radius sqrt(4 G M D_ls/(c^2 D_l D_s)) in radians.

    Parameters
    ----------
    M : float
        Lens mass (kg).
    D_l, D_s : float
        Distances to the lens and to the source (m; angular diameter
        distances in cosmology).
    D_ls : float, optional
        Lens-source distance (default D_s - D_l, the static Euclidean
        case).

    Returns
    -------
    theta_E : float
    """

    D_ls = D_s - D_l if D_ls is None else D_ls
    return np.sqrt(4*G*M*D_ls/(C**2*D_l*D_s))


def weak_field_medium(R_s=1.0, center=(0.0, 0.0)):
    """Weak field index n = 1 + R_s/r of a point mass as a GRIN medium.

    Parameters
    ----------
    R_s : float, optional
        Schwarzschild radius (length unit of the trace).
    center : sequence, optional

    Returns
    -------
    grin.IndexField
    """

    c = np.asarray(center, float)

    def n(r):
        return 1 + R_s/np.linalg.norm(r - c[:r.shape[-1]], axis=-1)

    def force(r):
        d = r - c[:r.shape[-1]]
        rr = np.linalg.norm(d, axis=-1)[..., None]
        return -(1 + R_s/rr)*R_s*d/rr**3            # grad(n^2)/2 = n grad n

    return IndexField(n, force)


def schwarzschild_medium(R_s=1.0, center=(0.0, 0.0)):
    """The exact Schwarzschild geometry as an isotropic medium.

    In isotropic coordinates (radius rho) the null geodesics of the
    Schwarzschild metric are the rays of n = (1 + R_s/(4 rho))^3/
    (1 - R_s/(4 rho)) (outside the horizon rho = R_s/4).

    Parameters
    ----------
    R_s : float, optional
        Schwarzschild radius.
    center : sequence, optional

    Returns
    -------
    grin.IndexField
    """

    c = np.asarray(center, float)
    a = R_s/4

    def n(r):
        rho = np.linalg.norm(r - c[:r.shape[-1]], axis=-1)
        return (1 + a/rho)**3/(1 - a/rho)

    def force(r):
        d = r - c[:r.shape[-1]]
        rho = np.linalg.norm(d, axis=-1)
        q = a/rho
        nn = (1 + q)**3/(1 - q)
        # dn/drho = dn/dq * dq/drho, dq/drho = -q/rho
        dn_dq = (3*(1 + q)**2*(1 - q) + (1 + q)**3)/(1 - q)**2
        dn = dn_dq*(-q/rho)
        return (nn*dn/rho)[..., None]*d

    return IndexField(n, force)


def deflection_weak(b, R_s=1.0):
    """First order deflection 2 R_s/b = 4 G M/(c^2 b) (radians)."""
    return 2*R_s/np.asarray(b, float)


def schwarzschild_deflection(b, R_s=1.0):
    """Exact deflection of a light ray by a Schwarzschild mass.

    alpha = 2 int_0^u0 du/sqrt(1/b^2 - u^2 + R_s u^3) - pi with u = 1/r
    and u0 the smallest positive root (the closest approach), evaluated by
    Gauss-Legendre quadrature after the substitution u = u0 (1 - t^2),
    which removes the inverse square root singularity. Rays with
    b <= 3 sqrt(3) R_s/2 are captured (NaN).

    Parameters
    ----------
    b : float or array
        Impact parameter.
    R_s : float, optional

    Returns
    -------
    alpha : array (radians)
    """

    b = np.atleast_1d(np.asarray(b, float))
    t, w = np.polynomial.legendre.leggauss(200)
    t, w = (t + 1)/2, w/2                       # nodes on [0, 1]
    out = np.full(b.shape, np.nan)
    for i, bb in enumerate(b):
        if bb <= 1.5*np.sqrt(3)*R_s*(1 + 1e-12):
            continue
        roots = np.roots([R_s, -1.0, 0.0, 1/bb**2])
        roots = np.real(roots[(np.abs(roots.imag) < 1e-12) & (roots.real > 0)])
        u0 = np.min(roots)
        u = u0*(1 - t**2)
        f = 1/bb**2 - u**2 + R_s*u**3
        out[i] = 2*np.sum(w*2*u0*t/np.sqrt(f)) - np.pi
    return out if out.size > 1 else out[0]


def deflection_second_order(b, R_s=1.0):
    """Deflection to second order in R_s/b, 2 R_s/b + 15 pi R_s^2/(16 b^2)."""
    b = np.asarray(b, float)
    return 2*R_s/b + 15*np.pi*R_s**2/(16*b**2)


class _Lens(object):
    def _rel(self, theta):
        return np.asarray(theta, float) - np.asarray(self.center, float)


class PointMass(_Lens):
    """Point mass lens, alpha = theta_E^2 theta/abs(theta)^2.

    Parameters
    ----------
    theta_E : float, optional
        Einstein radius.
    center : (float, float), optional
    """

    def __init__(self, theta_E=1.0, center=(0.0, 0.0)):
        self.theta_E, self.center = float(theta_E), tuple(center)

    def deflection(self, theta):
        t = self._rel(theta)
        r2 = np.sum(t**2, axis=-1, keepdims=True)
        return self.theta_E**2*t/r2

    def potential(self, theta):
        t = self._rel(theta)
        return self.theta_E**2*np.log(np.linalg.norm(t, axis=-1))


class SIS(_Lens):
    """Singular isothermal sphere, alpha = theta_E theta/abs(theta).

    Parameters
    ----------
    theta_E : float, optional
    center : (float, float), optional
    """

    def __init__(self, theta_E=1.0, center=(0.0, 0.0)):
        self.theta_E, self.center = float(theta_E), tuple(center)

    def deflection(self, theta):
        t = self._rel(theta)
        return self.theta_E*t/np.linalg.norm(t, axis=-1, keepdims=True)

    def potential(self, theta):
        return self.theta_E*np.linalg.norm(self._rel(theta), axis=-1)


class ExternalShear(_Lens):
    """External shear, potential gamma/2 (x'^2 - y'^2) in axes rotated by
    angle.

    Parameters
    ----------
    gamma : float
    angle : float, optional
        Direction of the x' axis (radians).
    """

    def __init__(self, gamma, angle=0.0):
        self.gamma, self.angle = float(gamma), float(angle)
        self.center = (0.0, 0.0)

    def deflection(self, theta):
        t = np.asarray(theta, float)
        c, s = np.cos(2*self.angle), np.sin(2*self.angle)
        x, y = t[..., 0], t[..., 1]
        return self.gamma*np.stack([c*x + s*y, s*x - c*y], axis=-1)

    def potential(self, theta):
        t = np.asarray(theta, float)
        c, s = np.cos(2*self.angle), np.sin(2*self.angle)
        x, y = t[..., 0], t[..., 1]
        return self.gamma/2*(c*(x**2 - y**2) + 2*s*x*y)


class Composite(_Lens):
    """Sum of lenses (deflections and potentials add)."""

    def __init__(self, lenses):
        self.lenses = list(lenses)

    def deflection(self, theta):
        return sum(L.deflection(theta) for L in self.lenses)

    def potential(self, theta):
        return sum(L.potential(theta) for L in self.lenses)


def lens_map(lens, theta):
    """Source position beta = theta - alpha(theta)."""
    theta = np.asarray(theta, float)
    return theta - lens.deflection(theta)


def jacobian(lens, theta, h=1e-6):
    """Jacobian d beta/d theta (..., 2, 2) by central differences."""
    theta = np.asarray(theta, float)
    J = np.empty(theta.shape + (2,))
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in range(2):
            e = np.zeros(2)
            e[k] = h
            J[..., :, k] = (lens_map(lens, theta + e) - lens_map(lens, theta - e))/(2*h)
    return J


def magnification(lens, theta, h=1e-6):
    """Signed magnification 1/det(d beta/d theta) at the image positions."""
    J = jacobian(lens, theta, h)
    return 1/(J[..., 0, 0]*J[..., 1, 1] - J[..., 0, 1]*J[..., 1, 0])


def find_images(lens, beta, extent=3.0, n=101, tol=1e-12, avoid=1e-3):
    """All images of a point source: Newton iterations from a grid.

    Parameters
    ----------
    lens : lens object
    beta : (float, float)
        Source position.
    extent : float, optional
        Half width of the square of start points.
    n : int, optional
        Start points per axis.
    tol : float, optional
        Required abs(beta(theta) - beta).
    avoid : float, optional
        Distance below which two solutions count as one, and around lens
        centres (singularities) where solutions are rejected.

    Returns
    -------
    theta : array (k, 2)
        Image positions, sorted by distance from the origin.
    mu : array (k,)
        Signed magnifications.
    """

    beta = np.asarray(beta, float)
    g = np.linspace(-extent, extent, n)
    T = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = T[np.all(np.isfinite(lens_map(lens, T)), axis=-1)]
    for _ in range(60):
        with np.errstate(invalid="ignore", divide="ignore"):
            F = lens_map(lens, T) - beta
        J = jacobian(lens, T)
        det = J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0]
        ok = np.isfinite(det) & (np.abs(det) > 1e-14)
        inv = np.zeros_like(J)
        inv[ok, 0, 0], inv[ok, 1, 1] = J[ok, 1, 1]/det[ok], J[ok, 0, 0]/det[ok]
        inv[ok, 0, 1], inv[ok, 1, 0] = -J[ok, 0, 1]/det[ok], -J[ok, 1, 0]/det[ok]
        step = np.nan_to_num(np.einsum("nij,nj->ni", inv, F))
        lim = 0.5*extent/n*10
        sn = np.linalg.norm(step, axis=-1, keepdims=True)
        T = T - step*np.minimum(1, lim/np.maximum(sn, 1e-300))
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.linalg.norm(lens_map(lens, T) - beta, axis=-1)
    T = T[np.isfinite(F) & (F < max(tol, 1e-9))]
    centres = [np.asarray(L.center) for L in getattr(lens, "lenses", [lens])
               if not isinstance(L, ExternalShear)]
    for c in centres:
        T = T[np.linalg.norm(T - c, axis=-1) > avoid]
    out = []
    for t in T:
        if all(np.linalg.norm(t - o) > avoid for o in out):
            out.append(t)
    out = np.array(out).reshape(-1, 2)
    out = out[np.argsort(np.linalg.norm(out, axis=-1))]
    return out, magnification(lens, out) if len(out) else np.zeros(0)


def critical_curve(lens, r_min=1e-3, r_max=3.0, n_angles=721, n_r=600, center=(0.0, 0.0)):
    """Critical curves (det of the lens map = 0) of a lens whose critical
    curves are star shaped about the centre: for every polar angle, the
    radii where det changes sign (bisection).

    Returns
    -------
    curves : list of arrays (n_angles, 2)
        One closed curve per root (ordered by radius); angles where a root
        is missing are NaN.
    """

    c = np.asarray(center, float)
    phi = np.linspace(0, 2*np.pi, n_angles)
    r = np.geomspace(r_min, r_max, n_r)
    u = np.stack([np.cos(phi), np.sin(phi)], axis=-1)

    def det(points):
        J = jacobian(lens, points)
        return J[..., 0, 0]*J[..., 1, 1] - J[..., 0, 1]*J[..., 1, 0]

    P = c + r[None, :, None]*u[:, None, :]
    D = det(P)
    roots = []
    for i in range(n_angles):
        idx = np.nonzero(np.sign(D[i, :-1]) != np.sign(D[i, 1:]))[0]
        rr = []
        for k in idx:
            a, b = r[k], r[k + 1]
            fa = D[i, k]
            for _ in range(50):
                m = (a + b)/2
                fm = det(c + m*u[i])
                if np.sign(fm) == np.sign(fa):
                    a, fa = m, fm
                else:
                    b = m
            rr.append((a + b)/2)
        roots.append(rr)
    n_c = max(len(x) for x in roots)
    curves = []
    for j in range(n_c):
        pts = np.full((n_angles, 2), np.nan)
        for i, rr in enumerate(roots):
            if j < len(rr):
                pts[i] = c + rr[j]*u[i]
        curves.append(pts)
    return curves


def caustic(lens, curves):
    """Caustics: the critical curves mapped to the source plane."""
    return [lens_map(lens, cc) for cc in curves]


def fermat_potential(lens, theta, beta):
    """tau = abs(theta - beta)^2/2 - psi(theta), the arrival time up to the
    factor (1 + z_l) D_l D_s/(c D_ls) (in units of theta_E^2)."""
    theta = np.asarray(theta, float)
    return 0.5*np.sum((theta - np.asarray(beta, float))**2, axis=-1) - lens.potential(theta)


def ray_shoot(lens, source, x, y):
    """Lensed image of an extended source by inverse ray shooting:
    I(theta) = S(beta(theta)) (surface brightness is conserved).

    Parameters
    ----------
    lens : lens object
    source : callable
        Surface brightness S(bx, by) in the source plane.
    x, y : arrays
        Image plane coordinates.

    Returns
    -------
    I : array (len(y), len(x))
    """

    X, Y = np.meshgrid(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        B = lens_map(lens, np.stack([X, Y], axis=-1))
    return np.nan_to_num(source(B[..., 0], B[..., 1]))


def point_mass_images(u):
    """Images of a point source at u (in theta_E) behind a point mass.

    Returns
    -------
    theta_plus, theta_minus : arrays
        Image positions along the source direction, (u +- sqrt(u^2 + 4))/2.
    mu_plus, mu_minus : arrays
        Magnifications (u^2 + 2)/(2 u sqrt(u^2 + 4)) +- 1/2 (the minus image
        has negative parity; its absolute value is returned).
    """

    u = np.asarray(u, float)
    s = np.sqrt(u**2 + 4)
    m = (u**2 + 2)/(2*u*s)
    return (u + s)/2, (u - s)/2, m + 0.5, m - 0.5


def paczynski(u):
    """Total point-lens magnification (u^2 + 2)/(u sqrt(u^2 + 4))."""
    u = np.asarray(u, float)
    return (u**2 + 2)/(u*np.sqrt(u**2 + 4))


def finite_source_magnification(rho):
    """Magnification of a uniform disk source of radius rho (theta_E)
    centred behind a point mass, sqrt(1 + 4/rho^2)."""
    rho = np.asarray(rho, float)
    return np.sqrt(1 + 4/rho**2)


def sis_images(beta, theta_E=1.0):
    """Images of a point source at beta > 0 behind an SIS.

    Returns
    -------
    theta : array
        Image positions along the source direction: beta + theta_E and,
        for beta < theta_E, beta - theta_E.
    mu : array
        Signed magnifications abs(theta)/(abs(theta) - theta_E).
    """

    th = [beta + theta_E] + ([beta - theta_E] if beta < theta_E else [])
    th = np.array(th)
    return th, np.abs(th)/(np.abs(th) - theta_E)


def sis_shear_cusps(gamma, theta_E=1.0):
    """Cusps of the tangential caustic of an SIS with external shear gamma
    (shear axis along x): at 2 gamma theta_E/(1 + gamma) on the x axis and
    2 gamma theta_E/(1 - gamma) on the y axis."""
    return 2*gamma*theta_E/(1 + gamma), 2*gamma*theta_E/(1 - gamma)
