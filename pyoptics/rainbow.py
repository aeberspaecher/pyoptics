#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Rainbows: rays through a spherical drop, Descartes' angles and Airy's
theory of the bow.

A ray hits a drop of radius a (index n in air) at the impact parameter
b = a sin(i), is refracted to the angle r (sin(i) = n sin(r)), reflected k
times inside (each time again at the angle r) and leaves after k + 1 chords.
Its deviation from the incident direction is ::

    D_k(i) = k pi + 2 i - 2 (k + 1) r,

k = 0 is the directly transmitted ray, k = 1 the primary bow, k = 2 the
secondary bow; k = -1 here stands for the external reflection,
D = 2 i - pi. The scattering angle (between the incident and the outgoing
directions, 0 forward, pi backward) is theta = arccos(cos(D)); the angle on
the sky measured from the antisolar point (the shadow of the observer's
head) is pi - theta.

For k >= 1, D_k(i) has a minimum (Descartes 1637), where many rays leave in
nearly the same direction, a fold caustic: the bow. The stationary point
dD/di = 0 lies at ::

    cos(i_c)**2 = (n**2 - 1)/(k (k + 2)),

for water (n = 1.333) at theta = 137.9 deg (the primary bow 42.1 deg from
the antisolar point) and 129.1 deg (the secondary bow at 50.9 deg). No rays
of the first two orders leave between the two: Alexander's dark band.

Geometric optics gives the differential cross section of one branch of
rays (rays per solid angle, normalized to the incident irradiance) ::

    dsigma/dOmega = b |db/dtheta|/sin(theta) eps,

eps the Fresnel power factor T(i) R(r)**k T(r) of the polarization s
(perpendicular to the scattering plane, the plane of the ray and the axis
through the drop centre, which every ray keeps in a sphere) or p (parallel);
it diverges at the caustic. Airy (1838) replaced the wavefront near the
rainbow ray by a cubic, theta(b) = theta_c + theta''_c (b - b_c)**2/2, and
integrated it in the diffraction approximation ::

    dsigma/dOmega = 2 pi b_c eps_c k**(1/3) (2/|theta''_c|)**(2/3) Ai(z)**2/sin(theta),
    z = -(theta - theta_c) sgn(theta''_c) k**(2/3) (2/|theta''_c|)**(1/3),

(k = 2 pi/wavelength), finite at the caustic, with a main peak just on the
bright side and the supernumerary fringes inside the primary bow (outside
the secondary). Their angular spacing scales as (wavelength/a)**(2/3).
Averaged over the fringes, Ai**2 approaches the geometric result.
Lorenz-Mie theory (`analytical.mie_amplitudes()`) is exact; it adds the
interference with the other orders (fine ripple) and the polarization
effects that Airy's scalar theory leaves out.

Functions
---------
- refraction_angle(), deviation(), scattering_angle(), descartes():
  the geometry of the k-th order and the Descartes angles.
- fresnel_factors(): eps_s and eps_p of the k-th order.
- trace_drop(): exact vector ray trace through a drop (any k), with all
  intersection points, the outgoing directions and the optical path.
- geometric_cross_section(): dsigma/dOmega of one order, per ray or summed
  over the branches on a grid of scattering angles.
- airy_parameters(): the rainbow ray and the curvature of theta(b) there.
- airy_cross_section(): Airy's theory of the k-th bow.
- drop_system(): the drop as a sequential `raytrace.System` for a band of
  impact parameters (one tilted spherical surface per intersection).

Lengths in one unit (radius, wavelength), angles in radians.

References
----------
R. Descartes, "Les Météores", discourse 8, Leiden (1637).

H. C. van de Hulst, "Light Scattering by Small Particles", Wiley (1957),
chapter 13.

H. M. Nussenzveig, "The theory of the rainbow", Sci. Am. 236(4), 116-127
(1977).

J. A. Adam, "The mathematical physics of rainbows and glories", Phys. Rep.
356, 229-365 (2002).

G. B. Airy, "On the intensity of light in the neighbourhood of a caustic",
Trans. Camb. Phil. Soc. 6, 379-403 (1838).
"""

from collections import namedtuple

import numpy as np

from .interfaces import R_p, R_s, T_p, T_s


__all__ = ["refraction_angle", "deviation", "scattering_angle", "descartes",
           "fresnel_factors", "trace_drop", "DropTrace", "geometric_cross_section",
           "airy_parameters", "airy_cross_section", "drop_system"]


DropTrace = namedtuple("DropTrace", ["points", "directions", "opl", "theta", "eps_s",
                                     "eps_p", "valid"])
DropTrace.__doc__ = """Rays traced through a drop (see trace_drop()).

points : (k + 4, N, 3) array (3 for k = -1)
    Start points, the intersections with the drop surface (entry, the k
    internal reflections, exit) and an end point one radius behind the exit
    (for drawing).
directions : (N, 3) array
    Outgoing unit directions.
opl : (N,) array
    Optical path from the start point to the exit point.
theta : (N,) array
    Scattering angles (radians, angle between +z and the outgoing direction).
eps_s, eps_p : (N,) arrays
    Fresnel power factors for s (perpendicular to the scattering plane) and
    p (parallel).
valid : (N,) bool array
    Rays that hit the drop.
"""


def refraction_angle(i, n):
    """Refraction angle r inside the drop, sin(i) = n sin(r)."""

    return np.arcsin(np.sin(i)/n)


def deviation(i, n, k):
    """Deviation D = k pi + 2 i - 2 (k + 1) r of the order k (k = -1:
    external reflection, D = 2 i - pi)."""

    return k*np.pi + 2*i - 2*(k + 1)*refraction_angle(i, n)


def scattering_angle(i, n, k):
    """Scattering angle arccos(cos(D)) in [0, pi] of the order k."""

    return np.arccos(np.cos(deviation(i, n, k)))


def _dD(i, n, k):
    r = refraction_angle(i, n)
    return 2 - 2*(k + 1)*np.cos(i)/(n*np.cos(r))


def _d2D(i, n, k):
    r = refraction_angle(i, n)
    dr = np.cos(i)/(n*np.cos(r))
    return 2*(k + 1)/n*(np.sin(i)*np.cos(r) - np.cos(i)*np.sin(r)*dr)/np.cos(r)**2


def descartes(n, k=1):
    """Descartes' rainbow ray of the order k >= 1.

    Parameters
    ----------
    n : number or array
        Refractive index of the drop (relative to the surrounding medium).
    k : int, optional
        Number of internal reflections.

    Returns
    -------
    i_c, r_c : numbers or arrays
        Angles of incidence and refraction of the rainbow ray.
    theta_c : number or array
        Its scattering angle (pi - theta_c from the antisolar point).
    """

    if k < 1:
        raise ValueError("Only orders k >= 1 have a rainbow.")
    n = np.asarray(n, dtype=float)
    c2 = (n**2 - 1)/(k*(k + 2))
    if np.any(c2 > 1):
        raise ValueError("No rainbow of order {} for n > {:.3f}.".format(k, np.sqrt(1 + k*(k + 2))))
    i_c = np.arccos(np.sqrt(c2))
    r_c = refraction_angle(i_c, n)

    return i_c[()], r_c[()], scattering_angle(i_c, n, k)[()]


def fresnel_factors(i, n, k):
    """Fresnel power factors (eps_s, eps_p) of the order k at incidence i.

    T(i) R(r)**k T(r) for k >= 0, the external reflectance R(i) for k = -1;
    s is perpendicular to the scattering plane, p parallel to it.
    """

    i = np.asarray(i, dtype=float)
    if k == -1:
        return R_s(i, 1.0, n), R_p(i, 1.0, n)
    r = refraction_angle(i, n)
    eps_s = T_s(i, 1.0, n)*R_s(r, n, 1.0)**k*T_s(r, n, 1.0)
    eps_p = T_p(i, 1.0, n)*R_p(r, n, 1.0)**k*T_p(r, n, 1.0)

    return np.real(eps_s), np.real(eps_p)


def _refract(d, N, mu):
    """Refract the directions d at normals N facing the incident rays
    (N.d < 0), mu = n1/n2."""

    cos_i = -np.sum(d*N, axis=-1)
    with np.errstate(invalid="ignore"):
        cos_t = np.sqrt(1 - mu**2*(1 - cos_i**2))

    return mu*d + (mu*cos_i - cos_t)[:, None]*N


def trace_drop(x, y, n, k=1, radius=1.0):
    """Exact ray trace of rays parallel to +z through a drop at the origin.

    Parameters
    ----------
    x, y : arrays
        Transverse start coordinates of the rays (impact parameter
        b = hypot(x, y)); the rays start at z = -2 radius.
    n : number
        Refractive index of the drop.
    k : int, optional
        Number of internal reflections (-1: external reflection).
    radius : number, optional

    Returns
    -------
    DropTrace
    """

    x, y = np.broadcast_arrays(np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel())
    N_rays = len(x)
    P = np.stack([x, y, np.full(N_rays, -2.0*radius)], axis=-1)
    d = np.tile([0.0, 0.0, 1.0], (N_rays, 1))
    a = radius
    pd = np.sum(P*d, axis=-1)
    disc = pd**2 - (np.sum(P**2, axis=-1) - a**2)
    valid = disc >= 0
    with np.errstate(invalid="ignore"):
        t = -pd - np.sqrt(disc)
    Q = P + t[:, None]*d
    opl = t.copy()
    points = [P, Q]
    N_out = Q/a
    if k == -1:
        d = d - 2*np.sum(d*N_out, axis=-1)[:, None]*N_out
    else:
        d = _refract(d, N_out, 1.0/n)
        for j in range(k + 1):
            t = -2*np.sum(Q*d, axis=-1)
            Q = Q + t[:, None]*d
            opl = opl + n*t
            points.append(Q)
            N_out = Q/a
            if j < k:
                d = d - 2*np.sum(d*N_out, axis=-1)[:, None]*N_out
            else:
                d = _refract(d, -N_out, n)
    points.append(Q + a*d)
    b = np.hypot(x, y)
    i = np.arcsin(np.clip(b/a, 0, 1))
    eps_s, eps_p = fresnel_factors(i, n, k)
    theta = np.arccos(np.clip(d[:, 2], -1, 1))
    nan = ~valid
    theta[nan] = np.nan
    opl[nan] = np.nan

    return DropTrace(np.array(points), d, opl, theta, eps_s, eps_p, valid)


def geometric_cross_section(n, k, radius=1.0, theta=None, i=None, num=20001):
    """Differential cross section dsigma/dOmega of the order k in geometric
    optics.

    Parameters
    ----------
    n : number
        Refractive index of the drop.
    k : int
        Number of internal reflections (-1: external reflection).
    radius : number, optional
    theta : array, optional
        Scattering angles: the cross section summed over all branches of
        rays that leave at theta (interpolated from num incidence angles).
    i : array, optional
        Angles of incidence instead (not with theta): the contribution of
        each of these rays.
    num : int, optional
        Number of incidence angles in (0, pi/2) for theta.

    Returns
    -------
    With i: theta(i), sigma_s(i), sigma_p(i). With theta: sigma_s(theta),
    sigma_p(theta). sigma is in units of radius**2 (per steradian).
    """

    def per_ray(i):
        th = scattering_angle(i, n, k)
        eps_s, eps_p = fresnel_factors(i, n, k)
        dD = np.abs(_dD(i, n, k)) if k != -1 else 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            g = radius**2*np.sin(i)*np.cos(i)/(np.sin(th)*dD)
        return th, g*eps_s, g*eps_p

    if i is not None:
        if theta is not None:
            raise ValueError("Give either theta or i.")
        return per_ray(np.asarray(i, dtype=float))
    if theta is None:
        raise ValueError("Give theta or i.")
    theta = np.asarray(theta, dtype=float)
    ii = np.linspace(0, np.pi/2, num + 2)[1:-1]
    th, s_s, s_p = per_ray(ii)
    # monotonic branches of theta(i):
    dth = np.sign(np.diff(th))
    breaks = np.flatnonzero(dth[1:] != dth[:-1]) + 1
    sig_s = np.zeros(theta.shape)
    sig_p = np.zeros(theta.shape)
    for idx in np.split(np.arange(len(ii)), breaks):
        if len(idx) < 2:
            continue
        t, a_s, a_p = th[idx], s_s[idx], s_p[idx]
        if t[-1] < t[0]:
            t, a_s, a_p = t[::-1], a_s[::-1], a_p[::-1]
        inside = (theta >= t[0]) & (theta <= t[-1])
        sig_s[inside] += np.interp(theta[inside], t, a_s)
        sig_p[inside] += np.interp(theta[inside], t, a_p)

    return sig_s, sig_p


def airy_parameters(n, k=1, radius=1.0):
    """The rainbow ray of the order k and the curvature of theta(b) there.

    Parameters
    ----------
    n : number
        Refractive index of the drop.
    k : int, optional
        Order of the bow (>= 1).
    radius : number, optional

    Returns
    -------
    theta_c : number
        Scattering angle of the rainbow ray.
    b_c : number
        Its impact parameter.
    theta2 : number
        d**2 theta/db**2 at b_c (per length**2): theta(b) = theta_c +
        theta2 (b - b_c)**2/2 near the bow. Negative for the secondary bow
        (the rays lie at theta < theta_c). Near the rainbow ray, the
        outgoing wavefront is Airy's cubic w = theta2 s**3/6 (s: transverse
        coordinate from the rainbow ray, ds/db = 1 there).
    """

    i_c, _, theta_c = descartes(n, k)
    side = np.sign(scattering_angle(i_c + 1e-3, n, k) - theta_c)
    theta2 = side*abs(_d2D(i_c, n, k))/(radius*np.cos(i_c))**2

    return theta_c, radius*np.sin(i_c), theta2


def airy_cross_section(theta, radius, wavelength, n, k=1):
    """Airy's theory of the rainbow of order k: dsigma/dOmega near the bow.

    Parameters
    ----------
    theta : array
        Scattering angles.
    radius, wavelength : numbers
        Drop radius and (vacuum) wavelength in the same unit.
    n : number
        Refractive index of the drop.
    k : int, optional
        Order of the bow (1 primary, 2 secondary, ...).

    Returns
    -------
    sigma_s, sigma_p : arrays
        Cross sections per steradian (in the length unit squared) for s and
        p polarization, with the Fresnel factors of the rainbow ray. Valid
        within a few fringes of the bow, compare `analytical.mie_amplitudes()`
        (``|S1|**2/k**2`` and ``|S2|**2/k**2``).
    """

    from scipy.special import airy

    theta = np.asarray(theta, dtype=float)
    theta_c, b_c, d2 = airy_parameters(n, k, radius)
    wn = 2*np.pi/wavelength
    q = 2/abs(d2)
    z = -np.sign(d2)*(theta - theta_c)*wn**(2/3)*q**(1/3)
    Ai = airy(z)[0]
    eps_s, eps_p = fresnel_factors(np.arcsin(b_c/radius), n, k)
    g = 2*np.pi*b_c*wn**(1/3)*q**(2/3)*Ai**2/np.sin(theta)

    return g*eps_s, g*eps_p


def drop_system(radius, n, k, i_band, wavelength=550e-6, units="mm"):
    """A drop as a sequential `raytrace.System` (k internal reflections).

    A sequential surface is one hemisphere of the sphere (the branch through
    its vertex), so a ray path that goes round a drop needs one surface per
    intersection, each tilted about x such that it contains that
    intersection but not the previous one. For rays in the meridional y-z
    plane near the angle of incidence i_band, the j-th intersection lies at
    the polar angle phi_j = i + j (pi - 2 r) (seen from the drop centre,
    from the front pole towards +y); surface j (j >= 1) is centred 45 deg
    beyond phi_j. This works for rays within about +-10 deg of i_band.

    Parameters
    ----------
    radius : number
        Drop radius; the front pole is the vertex of the first surface at
        z = 0, the centre lies at z = radius.
    n : number or callable
        Refractive index of the drop (or n(wavelength_nm)).
    k : int
        Number of internal reflections (>= 0).
    i_band : number
        Angle of incidence (radians) of the band of rays.
    wavelength : number, optional
        Wavelength in units.
    units : str, optional
        Length unit of the system.

    Returns
    -------
    raytrace.System
        Entry surface (the stop, semi diameter = radius), k mirrors and the
        exit surface, the image plane at z = 3 radius (rays may be
        travelling away from it). Trace rays parallel to +z with
        `System.trace()`; the directions after the last surface are the
        outgoing ones.
    """

    from .raytrace import NM_PER_UNIT, Surface, System

    if k < 0:
        raise ValueError("k >= 0 internal reflections.")
    a = radius
    n_c = float(n(wavelength*NM_PER_UNIT[units])) if callable(n) else n    # nominal index for the angles
    r = np.arcsin(np.sin(i_band)/n_c)
    medium = dict(material=n) if callable(n) else dict(n_after=n)
    surfaces = [Surface(a, 0.0, semi_diameter=a, **medium)]
    z_prev = 0.0
    for j in range(1, k + 2):
        phi = i_band + j*(np.pi - 2*r) + np.pi/4
        p_y, p_z = a*np.sin(phi), a - a*np.cos(phi)
        kw = dict(n_after=1.0) if j == k + 1 else dict(mirror=True)
        surfaces[-1].thickness = p_z - z_prev
        surfaces.append(Surface(a, 0.0, decenter=(0.0, p_y), tilt=(phi, 0.0, 0.0), **kw))
        z_prev = p_z
    surfaces[-1].thickness = 3*a - z_prev

    return System(surfaces, 0, wavelength, units=units)
