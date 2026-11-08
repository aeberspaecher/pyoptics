#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Illumination optics: Monte Carlo sources, phase space and étendue.

Illumination optics moves power, not images. Its central quantity is the
étendue, the volume that a beam occupies in phase space. A ray at the
point (x, y) of a plane z = const with the unit direction (L, M, N) in a
medium of index n has the canonical coordinates ::

    (x, y, p, q) = (x, y, n*L, n*M),

the optical direction cosines p and q being the transverse components of
the wave vector over k_0. The étendue of a beam is the volume it fills in
this four dimensional space, ::

    G = int dx dy dp dq = n**2 int cos(theta) dA dOmega,

and for a Lambertian source of area A that emits into the cone of
numerical aperture NA = n*sin(theta) it is G = pi*A*NA**2 (in 2D, a strip
of width w into +-NA: G = 2*w*NA). Lossless optics maps phase space onto
itself symplectically (Hamilton's equations), with a unit Jacobian
determinant. So the étendue of a beam can never shrink, and the basic
radiance L/n**2 (power per phase space volume) is carried unchanged along
every ray: the brightness theorem. An image is never brighter than its
source, and the illuminance at a point is bounded by
E <= L*pi*n**2*sin(theta)**2, which limits the concentration of light
from a source of angular radius alpha to ::

    C_max = n**2/sin(alpha)**2  (3D),    C_max = n/sin(alpha)  (2D),

about 46,000 for sunlight in air.

The module provides

- Monte Carlo ray sources: points on disks and squares, Lambertian
  directions (uniform in the direction sines, which is exactly a source of
  constant radiance), and `LambertianSource`, which returns
  `raytrace.Rays` for `raytrace.System.trace()`;
- the canonical coordinates of rays (`phase_space()`), the skew invariant
  of rotationally symmetric systems (`skew_invariant()`) and the Jacobian
  of a ray map (`phase_space_jacobian()`, `system_phase_map()`);
- étendue: closed forms for Lambertian sources (`etendue_lambertian()`,
  `etendue_2d()`), the concentration limit (`concentration_limit()`), the
  coupling bound min(1, G_target/G_source) (`coupling_limit()`), and two
  estimators from rays: the occupied volume of a phase space histogram
  (`occupied_etendue()`) and the participation étendue
  ``G_eff = P**2/int L**2`` (`effective_etendue()`), which is exact for a
  uniform fill, unbiased against the Monte Carlo noise, and invariant under
  any lossless map, since the map moves the values of L around without
  changing how much volume carries which value;
- simple non-imaging elements traced in closed form: refraction at a plane
  (`refract_plane()`), a diffuser (`diffuse()`) and a straight light pipe
  with square or rectangular cross section (`light_pipe()`, by unfolding
  the reflections at its walls);
- the Sun as a source and the temperature that concentrated sunlight can
  reach (`SUN_ANGULAR_RADIUS`, `concentration_temperature()`);
- nonimaging concentrators: the compound parabolic concentrator (`CPC`,
  `cpc_profile()`), designed by the edge-ray principle, hollow with mirror
  walls or solid with TIR walls, as a 2D trough or a 3D surface of
  revolution, optionally truncated; rays are traced through its wall
  reflections (bracketing and bisection of the wall hit, with a compiled
  kernel if numba is installed), and `tilted_beam()` launches parallel
  rays over an aperture;
- backward ray tracing: `reverse_system()` turns a centred
  `raytrace.System` round, `backward_trace()` finds the radiance arriving
  at a point from given directions and `backward_irradiance()` integrates
  it into the irradiance (brightness theorem: E = integral of L dp dq), so
  the field of a lamp, condenser and diaphragms is mapped without forward
  Monte Carlo through small apertures.
- homogenizers: ideal thin lenses and lenslet arrays for rays
  (`thin_lens()`, `lenslet_array()`), and the scalar wave field of a
  fly's eye homogenizer with one or two arrays in the focal plane of its
  condenser (`fly_eye_field_1d()`), for the interference of coherent light.

Units: lengths in any unit (mm in the notebooks), angles in radians, NA
includes the index of the medium.

References
----------
W. T. Welford and R. Winston, "High Collection Nonimaging Optics",
Academic Press (1989), ch. 2 and appendix A.

R. Winston, J. C. Miñano and P. Benítez, "Nonimaging Optics", Elsevier
(2005), ch. 3 and 16.

J. Chaves, "Introduction to Nonimaging Optics", 2nd ed., CRC Press (2016),
ch. 1 and 16.

M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), sec. 4.8 (the étendue as an invariant of the ray map).

R. Winston, "Principles of solar concentrators of a novel design", Solar
Energy 16, 89-95 (1974).
"""

from collections import namedtuple

import numpy as np

from .raytrace import Rays
from . import _illumination_numba


# Trace CPCs with the compiled kernel if numba is installed
# (pip install pyoptics[fast]); False: always NumPy.
use_numba = True


# mean angular radius of the solar disk (959.6 arcsec at 1 AU)
SUN_ANGULAR_RADIUS = np.deg2rad(959.6/3600)
# effective (black body) temperature of the Sun's photosphere in K
SUN_TEMPERATURE = 5772.0


# sources:
def _rng(rng):
    return np.random.default_rng(rng)


def disk_points(n_rays, radius, rng=None):
    """Points distributed uniformly on a disk.

    Parameters
    ----------
    n_rays : int
        Number of points.
    radius : number
        Radius of the disk.
    rng : None, int or numpy.random.Generator, optional
        Random generator or seed.

    Returns
    -------
    xy : (n_rays, 2) array
    """

    rng = _rng(rng)
    r = radius*np.sqrt(rng.random(n_rays))
    phi = 2*np.pi*rng.random(n_rays)

    return np.column_stack([r*np.cos(phi), r*np.sin(phi)])


def square_points(n_rays, width, height=None, rng=None):
    """Points distributed uniformly on a rectangle centred on the origin.

    Parameters
    ----------
    n_rays : int
        Number of points.
    width : number
        Full width in x.
    height : number, optional
        Full height in y (default: width, a square).
    rng : None, int or numpy.random.Generator, optional
        Random generator or seed.

    Returns
    -------
    xy : (n_rays, 2) array
    """

    rng = _rng(rng)
    height = width if height is None else height

    return (rng.random((n_rays, 2)) - 0.5)*np.array([width, height])


def lambertian_directions(n_rays, na=1.0, n=1.0, rng=None):
    """Unit directions of a Lambertian (constant radiance) emitter.

    A source of constant radiance puts equal power into equal areas of the
    plane of the direction sines (L, M), dP ~ cos(theta) dOmega = dL dM/N*N.
    The directions are therefore drawn uniformly from the disk
    L**2 + M**2 <= (na/n)**2. With na = n this is the full Lambertian
    emitter (intensity ~ cos(theta)); with na = sin(alpha) it is a disk of
    constant brightness and angular radius alpha, such as the Sun.

    Parameters
    ----------
    n_rays : int
        Number of directions.
    na : number, optional
        Numerical aperture n*sin(theta_max) of the emission cone.
    n : number, optional
        Refractive index of the medium.
    rng : None, int or numpy.random.Generator, optional
        Random generator or seed.

    Returns
    -------
    d : (n_rays, 3) array
        Unit directions with d_z > 0.
    """

    s = na/n
    if not 0 <= s <= 1:
        raise ValueError("Need 0 <= na <= n.")
    LM = disk_points(n_rays, s, rng)

    return np.column_stack([LM, np.sqrt(np.clip(1 - np.sum(LM**2, axis=1), 0, None))])


def directions_from_sines(p, q, n=1.0):
    """Unit directions from the optical direction cosines p = n*L, q = n*M.

    Parameters
    ----------
    p, q : arrays
        Optical direction cosines.
    n : number, optional
        Refractive index.

    Returns
    -------
    d : (N, 3) array
        Unit directions with d_z >= 0 (NaN where p**2 + q**2 > n**2).
    """

    L, M = np.broadcast_arrays(np.asarray(p, float)/n, np.asarray(q, float)/n)
    with np.errstate(invalid="ignore"):
        N = np.sqrt(1 - L**2 - M**2)
    d = np.column_stack([L.ravel(), M.ravel(), N.ravel()])
    d[~np.isfinite(d[:, 2])] = np.nan

    return d


class LambertianSource(object):
    """Flat source of constant radiance (Monte Carlo rays).

    Parameters
    ----------
    size : number
        Radius of a disk, or full width of a square (or rectangle).
    na : number, optional
        Numerical aperture of the emission cone (default 1: a full
        Lambertian emitter in air).
    shape : {"disk", "square"}, optional
        Shape of the emitting area.
    height : number, optional
        Full height of a rectangle (shape "square"; default: size).
    n : number, optional
        Refractive index of the medium in front of the source.
    z : number, optional
        Axial position of the source plane.
    center : pair, optional
        Centre (x, y) of the source.
    power : number, optional
        Total emitted power (shared by the rays as their weights).
    """

    def __init__(self, size, na=1.0, shape="disk", height=None, n=1.0, z=0.0, center=(0.0, 0.0),
                 power=1.0):
        if shape not in ("disk", "square"):
            raise ValueError("shape must be 'disk' or 'square'.")
        self.size = size
        self.na = na
        self.shape = shape
        self.height = height
        self.n = n
        self.z = z
        self.center = tuple(center)
        self.power = power

    def __repr__(self):
        return "LambertianSource(size={}, na={}, shape={!r}, n={}, z={})".format(
            self.size, self.na, self.shape, self.n, self.z)

    @property
    def area(self):
        """Emitting area."""
        if self.shape == "disk":
            return np.pi*self.size**2
        return self.size*(self.size if self.height is None else self.height)

    @property
    def etendue(self):
        """Étendue pi*A*NA**2 of the source."""
        return etendue_lambertian(self.area, self.na)

    @property
    def radiance(self):
        """Basic radiance power/étendue (power per phase space volume)."""
        return self.power/self.etendue

    def points(self, n_rays, rng=None):
        """Start points (n_rays, 3) on the source plane."""
        if self.shape == "disk":
            xy = disk_points(n_rays, self.size, rng)
        else:
            xy = square_points(n_rays, self.size, self.height, rng)
        xy = xy + np.array(self.center)

        return np.column_stack([xy, np.full(n_rays, float(self.z))])

    def rays(self, n_rays, rng=None):
        """Monte Carlo rays of the source.

        Parameters
        ----------
        n_rays : int
            Number of rays (each carries the power power/n_rays).
        rng : None, int or numpy.random.Generator, optional
            Random generator or seed.

        Returns
        -------
        Rays
            Points, unit directions and zero optical paths, for
            `raytrace.System.trace()`.
        """

        rng = _rng(rng)
        P = self.points(n_rays, rng)
        d = lambertian_directions(n_rays, self.na, self.n, rng)

        return Rays(P, d, np.zeros(n_rays))


# étendue in closed form:
def etendue_lambertian(area, na):
    """Étendue pi*A*NA**2 of a Lambertian emitter (or acceptor).

    Parameters
    ----------
    area : number
        Emitting (or accepting) area.
    na : number
        Numerical aperture n*sin(theta) of its cone.

    Returns
    -------
    G : number
    """

    return np.pi*area*np.asarray(na, float)**2


def etendue_2d(width, na):
    """Étendue 2*w*NA of a strip of width w into the directions +-NA (2D).

    Parameters
    ----------
    width : number
        Full width.
    na : number
        Numerical aperture n*sin(theta).

    Returns
    -------
    G : number
    """

    return 2*width*np.asarray(na, float)


def concentration_limit(half_angle, n=1.0, dim=3):
    """Maximum concentration of light from a source of angular radius alpha.

    A receiver that accepts light from all directions of a medium of
    index n has the étendue pi*A*n**2 (3D) or 2*w*n (2D); the light from a
    source seen under the half angle alpha has pi*A*sin(alpha)**2, so ::

        C_max = n**2/sin(alpha)**2  (3D),    C_max = n/sin(alpha)  (2D).

    Parameters
    ----------
    half_angle : number or array
        Angular radius alpha of the source (or acceptance half angle).
    n : number, optional
        Refractive index at the receiver.
    dim : {3, 2}, optional
        Three dimensional or trough (2D) concentrator.

    Returns
    -------
    C : number or array
    """

    s = np.sin(np.asarray(half_angle, float))
    if dim == 3:
        return n**2/s**2
    if dim == 2:
        return n/s
    raise ValueError("dim must be 2 or 3.")


def coupling_limit(etendue_source, etendue_target):
    """Upper bound min(1, G_target/G_source) of the power coupled from a
    source of constant radiance into a target (fibre, light pipe, pupil).

    Parameters
    ----------
    etendue_source, etendue_target : numbers or arrays

    Returns
    -------
    eta : number or array
    """

    return np.minimum(1.0, np.asarray(etendue_target, float)/etendue_source)


def concentration_temperature(concentration, absorptance=1.0, emittance=1.0,
                              half_angle=SUN_ANGULAR_RADIUS, T_source=SUN_TEMPERATURE):
    """Stagnation temperature of a receiver under concentrated sunlight.

    A black (or grey) receiver in vacuum, cooled only by its own thermal
    radiation into the hemisphere, absorbs absorptance*C*sigma*T_source**4
    /C_max (the Sun's radiance times the concentrated étendue) and emits
    emittance*sigma*T**4, so ::

        T = T_source*(absorptance*C/(emittance*C_max))**(1/4).

    At C = C_max a black receiver reaches the temperature of the Sun, and no
    optics can do better (the second law of thermodynamics).

    Parameters
    ----------
    concentration : number or array
        Concentration C (flux density over that of direct sunlight).
    absorptance, emittance : numbers, optional
        Of the receiver (1, 1: black body).
    half_angle : number, optional
        Angular radius of the source.
    T_source : number, optional
        Black body temperature of the source in K.

    Returns
    -------
    T : number or array
        Temperature in K.
    """

    C_max = concentration_limit(half_angle)

    return T_source*(absorptance*np.asarray(concentration, float)/(emittance*C_max))**0.25


# phase space:
def phase_space(points, directions, n=1.0):
    """Canonical coordinates (x, y, p, q) = (x, y, n*L, n*M) of rays.

    Parameters
    ----------
    points, directions : (N, 3) arrays
        Points (on a plane z = const) and unit directions.
    n : number, optional
        Refractive index of the medium.

    Returns
    -------
    X : (N, 4) array
    """

    P = np.asarray(points, float).reshape(-1, 3)
    d = np.asarray(directions, float).reshape(-1, 3)

    return np.column_stack([P[:, 0], P[:, 1], n*d[:, 0], n*d[:, 1]])


def skew_invariant(points, directions, n=1.0):
    """Skew invariant n*(x*M - y*L) of rays.

    It is the angular momentum of the ray about the z axis and conserved in
    every rotationally symmetric system (lenses, mirrors, cones, GRIN
    media). It is zero for meridional rays; skew rays carry it through the
    system, which is what limits 3D concentrators of revolution.

    Parameters
    ----------
    points, directions : (N, 3) arrays
        Points (anywhere on the ray) and unit directions.
    n : number, optional
        Refractive index.

    Returns
    -------
    h : (N,) array
    """

    P = np.asarray(points, float).reshape(-1, 3)
    d = np.asarray(directions, float).reshape(-1, 3)

    return n*(P[:, 0]*d[:, 1] - P[:, 1]*d[:, 0])


def system_phase_map(system, X, z_in=None, n_in=None, clip=True):
    """Map canonical ray coordinates through a `raytrace.System`.

    Parameters
    ----------
    system : raytrace.System
        The system; rays start in the plane z = z_in in front of the first
        surface and end in its image plane.
    X : (N, 4) array
        Input coordinates (x, y, p, q).
    z_in : number, optional
        Axial position of the input plane (default: 0, the first vertex,
        valid if that surface is plane or the rays meet it behind z = 0).
    n_in : number, optional
        Index of the input medium (default: system.n_object).
    clip : bool, optional
        Vignette rays at the semi diameters.

    Returns
    -------
    Y : (N, 4) array
        Output coordinates in the image plane (NaN for lost rays).
    """

    X = np.atleast_2d(np.asarray(X, float))
    z_in = 0.0 if z_in is None else z_in
    n_in = system.n_object if n_in is None else n_in
    P = np.column_stack([X[:, 0], X[:, 1], np.full(len(X), float(z_in))])
    d = directions_from_sines(X[:, 2], X[:, 3], n_in)
    bundle = system.trace(Rays(P, d, np.zeros(len(X))), clip=clip)
    Q, e = system.to_image_frame(bundle.points[-1], bundle.directions[-1])

    return phase_space(Q, e, bundle.n[-1])


def phase_space_jacobian(ray_map, X, h=1e-6):
    """Jacobian matrices of a map of canonical ray coordinates.

    A lossless optical system is a symplectic map: its Jacobian
    J = dY/dX obeys J.T @ Omega @ J = Omega with
    Omega = [[0, I], [-I, 0]], so det J = 1 and the étendue of every small
    pencil is conserved.

    Parameters
    ----------
    ray_map : callable
        Function (N, d) -> (N, d) of canonical coordinates, e.g.
        ``lambda X: system_phase_map(system, X)``; d = 2 (x, p) or
        4 (x, y, p, q).
    X : (N, d) array
        Points at which to differentiate.
    h : number or (d,) array, optional
        Steps of the central differences.

    Returns
    -------
    J : (N, d, d) array
    """

    X = np.atleast_2d(np.asarray(X, float))
    N, dim = X.shape
    h = np.broadcast_to(np.asarray(h, float), (dim,))
    J = np.empty((N, dim, dim))
    for j in range(dim):
        e = np.zeros(dim)
        e[j] = h[j]
        J[:, :, j] = (ray_map(X + e) - ray_map(X - e))/(2*h[j])

    return J


def symplectic_defect(J):
    """Largest element of ``|J.T Omega J - Omega|`` for each Jacobian.

    Parameters
    ----------
    J : (..., d, d) array
        Jacobians in the coordinates (x, [y,] p, [q]).

    Returns
    -------
    defect : (...) array
    """

    J = np.asarray(J, float)
    d = J.shape[-1]//2
    Om = np.block([[np.zeros((d, d)), np.eye(d)], [-np.eye(d), np.zeros((d, d))]])
    D = np.swapaxes(J, -1, -2) @ Om @ J - Om

    return np.max(np.abs(D), axis=(-2, -1))


def phase_space_histogram(X, bins, ranges, weights=None):
    """Radiance in phase space: power per bin volume.

    Parameters
    ----------
    X : (N, d) array
        Canonical coordinates of the rays (NaN rows are dropped).
    bins : int or sequence of ints
        Number of bins per coordinate.
    ranges : sequence of (min, max)
        Range of every coordinate.
    weights : (N,) array, optional
        Power of every ray (default: 1/N each, unit total power of the
        rays launched).

    Returns
    -------
    L : array
        Basic radiance (power per phase space volume) in the bins.
    edges : list of arrays
        Bin edges.
    """

    X = np.atleast_2d(np.asarray(X, float))
    w = np.full(len(X), 1/len(X)) if weights is None else np.asarray(weights, float)
    ok = np.all(np.isfinite(X), axis=1)
    H, edges = np.histogramdd(X[ok], bins=bins, range=ranges, weights=w[ok])
    vol = np.prod([e[1] - e[0] for e in edges])

    return H/vol, edges


def effective_etendue(X, bins, ranges, weights=None):
    """Participation étendue G_eff = P**2/int L**2 of a ray set.

    For a beam of constant basic radiance L_0 over the phase space volume G,
    P = L_0*G and int L**2 = L_0**2*G, so G_eff = G. For any other
    distribution G_eff is smaller than the occupied volume and weighs it by
    the radiance. It is invariant under every lossless (symplectic) map,
    which only rearranges the values of L in phase space, and grows under
    diffusers and mixing. The square of the binned radiance is corrected
    for the Monte Carlo noise (sum of w_i w_j over pairs of different rays
    in a bin), so the estimate is unbiased for fine bins with few rays.

    Parameters
    ----------
    X : (N, d) array
        Canonical coordinates (NaN rows count as lost power).
    bins : int or sequence of ints
        Number of bins per coordinate; the bins must be small compared to
        the scale on which L varies.
    ranges : sequence of (min, max)
        Range of every coordinate (must contain all rays).
    weights : (N,) array, optional
        Ray powers (default 1/N).

    Returns
    -------
    G : number
        Effective étendue in the units of the coordinates (area*NA**2).
    """

    X = np.atleast_2d(np.asarray(X, float))
    w = np.full(len(X), 1/len(X)) if weights is None else np.asarray(weights, float)
    ok = np.all(np.isfinite(X), axis=1)
    W, edges = np.histogramdd(X[ok], bins=bins, range=ranges, weights=w[ok])
    W2, _ = np.histogramdd(X[ok], bins=bins, range=ranges, weights=w[ok]**2)
    vol = np.prod([e[1] - e[0] for e in edges])
    P = W.sum()
    pairs = np.sum(W**2 - W2)

    return P**2*vol/pairs


def occupied_etendue(X, bins, ranges, weights=None, threshold=0.0):
    """Phase space volume of the bins that carry power.

    Parameters
    ----------
    X : (N, d) array
        Canonical coordinates.
    bins : int or sequence of ints
        Number of bins per coordinate.
    ranges : sequence of (min, max)
        Range of every coordinate.
    weights : (N,) array, optional
        Ray powers.
    threshold : number, optional
        Count only bins whose radiance exceeds threshold times the mean
        radiance of the occupied bins.

    Returns
    -------
    G : number
        Occupied volume (biased high by bins cut by the rim of the beam and
        low where the ray density is too small to fill the bins).
    """

    L, edges = phase_space_histogram(X, bins, ranges, weights)
    vol = np.prod([e[1] - e[0] for e in edges])
    occupied = L > 0
    if threshold > 0:
        occupied &= L > threshold*L[occupied].mean()

    return occupied.sum()*vol


# simple elements in closed form:
def refract_plane(directions, n1, n2):
    """Refraction of unit directions at a plane z = const.

    The optical direction cosines p, q are conserved (Snell's law), the
    angles shrink by n1/n2 and the basic radiance L/n**2 stays the same.

    Parameters
    ----------
    directions : (N, 3) array
        Unit directions in the medium n1, d_z > 0.
    n1, n2 : numbers
        Refractive indices in front of and behind the plane.

    Returns
    -------
    d2 : (N, 3) array
        Unit directions behind the plane (NaN for total internal reflection).
    """

    d = np.asarray(directions, float).reshape(-1, 3)

    return directions_from_sines(n1*d[:, 0], n1*d[:, 1], n2)


def diffuse(directions, spread, n=1.0, kind="gaussian", rng=None):
    """Thin diffuser: add random direction sines to every ray.

    Parameters
    ----------
    directions : (N, 3) array
        Unit directions in front of the diffuser.
    spread : number
        Width of the scattering in the direction sines: the standard
        deviation per axis (kind "gaussian") or the radius of a disk (kind
        "disk", a top hat diffuser).
    n : number, optional
        Refractive index of the medium.
    kind : {"gaussian", "disk"}, optional
        Scattering distribution.
    rng : None, int or numpy.random.Generator, optional
        Random generator or seed.

    Returns
    -------
    d2 : (N, 3) array
        New unit directions; NaN for rays scattered beyond the horizon
        (evanescent), which are lost.
    """

    rng = _rng(rng)
    d = np.asarray(directions, float).reshape(-1, 3)
    if kind == "gaussian":
        dLM = spread*rng.standard_normal((len(d), 2))
    elif kind == "disk":
        dLM = disk_points(len(d), spread, rng)
    else:
        raise ValueError("kind must be 'gaussian' or 'disk'.")

    return directions_from_sines(d[:, 0] + dLM[:, 0], d[:, 1] + dLM[:, 1], 1.0)


def _fold(u, width):
    """Fold the unfolded coordinate u of a pipe with mirror walls at
    +-width/2 back into the pipe; returns the coordinate, the sign of the
    direction (-1 after an odd number of reflections) and the number of
    reflections."""

    v = u + width/2
    k = np.floor(v/width)
    v = v - k*width
    odd = (k % 2) != 0

    return np.where(odd, width - v, v) - width/2, np.where(odd, -1.0, 1.0), np.abs(k)


def light_pipe(points, directions, width, length, height=None, n_pipe=1.5, n_outside=1.0,
               n_in=1.0, n_out=None, mirror=False):
    """Straight light pipe (integrator rod) with rectangular cross section.

    The entrance face lies in the plane of the points, centred on the axis;
    the rays refract into the pipe (plane face), are reflected at its walls
    and leave through the exit face at z + length. A straight rectangular
    pipe is the unfolded mirror kaleidoscope: every reflection at a wall
    only flips the sign of one direction cosine, so the exit point follows
    from folding the straight ray of the unfolded pipe back into the
    cross section. The set of exit directions is that of the entrance with
    the signs of p and q flipped for each odd number of reflections, and
    the étendue is conserved.

    Parameters
    ----------
    points : (N, 3) array
        Points on the entrance face (x, y relative to the pipe axis).
    directions : (N, 3) array
        Unit directions in front of the entrance face.
    width, length : numbers
        Full width in x and length of the pipe.
    height : number, optional
        Full height in y (default: width, a square pipe).
    n_pipe : number, optional
        Refractive index of the pipe (1 for a hollow mirror pipe).
    n_outside : number, optional
        Index around a solid pipe; rays that would not be totally reflected
        at its walls are lost (unless mirror is True).
    n_in, n_out : numbers, optional
        Indices in front of the entrance and behind the exit face (default
        n_out = n_in).
    mirror : bool, optional
        Mirror walls (no TIR condition).

    Returns
    -------
    P_out : (N, 3) array
        Points on the exit face (NaN for lost rays).
    d_out : (N, 3) array
        Unit directions behind the exit face.
    n_reflections : (N, 2) array
        Reflections at the x and y walls.
    """

    height = width if height is None else height
    n_out = n_in if n_out is None else n_out
    P = np.asarray(points, float).reshape(-1, 3)
    d = refract_plane(directions, n_in, n_pipe)
    tx, ty = d[:, 0]/d[:, 2], d[:, 1]/d[:, 2]
    x, sx, kx = _fold(P[:, 0] + length*tx, width)
    y, sy, ky = _fold(P[:, 1] + length*ty, height)
    inside = (np.abs(P[:, 0]) <= width/2) & (np.abs(P[:, 1]) <= height/2)
    if not mirror:
        # TIR at the x walls needs the direction cosine along the wall
        # normal below sqrt(1 - (n_outside/n_pipe)**2), same for y
        c = np.sqrt(max(0.0, 1 - (n_outside/n_pipe)**2))
        inside &= ((kx == 0) | (np.abs(d[:, 0]) < c)) & ((ky == 0) | (np.abs(d[:, 1]) < c))
    d_in = np.column_stack([sx*d[:, 0], sy*d[:, 1], d[:, 2]])
    d_out = refract_plane(d_in, n_pipe, n_out)
    P_out = np.column_stack([x, y, P[:, 2] + length])
    lost = ~inside | ~np.all(np.isfinite(d_out), axis=1)
    P_out[lost] = np.nan
    d_out[lost] = np.nan

    return P_out, d_out, np.column_stack([kx, ky])


def propagate(points, directions, dz):
    """Propagate rays in a homogeneous medium to the plane dz further on.

    Parameters
    ----------
    points, directions : (N, 3) arrays
    dz : number
        Axial distance.

    Returns
    -------
    points : (N, 3) array
    """

    P = np.asarray(points, float).reshape(-1, 3)
    d = np.asarray(directions, float).reshape(-1, 3)

    return P + (dz/d[:, 2])[:, None]*d


# nonimaging concentrators:
def cpc_profile(acceptance, exit_radius, n_points=4001):
    """Profile of a compound parabolic concentrator (CPC).

    The wall is a parabola whose axis is tilted by the acceptance angle
    theta to the CPC axis and whose focus is the opposite rim of the exit
    aperture (the edge-ray principle: rays entering at theta through the
    entrance rim are reflected onto the opposite exit rim). With the focal
    length f = (1 + sin(theta)) a' and the parameter phi from 2*theta
    (entrance) to theta + pi/2 (exit), ::

        r(phi)  = 2*f*sin(phi - theta)/(1 - cos(phi)) - a',
        z'(phi) = 2*f*cos(phi - theta)/(1 - cos(phi)),

    measured from the exit plane. The entrance radius is a = a'/sin(theta),
    the length L = (a + a')/tan(theta).

    Parameters
    ----------
    acceptance : number
        Acceptance half angle theta (inside the medium of the CPC).
    exit_radius : number
        Radius (half width) a' of the exit aperture.
    n_points : int, optional
        Number of profile points.

    Returns
    -------
    z : (n_points,) array
        Distance from the entrance plane (0 to L), increasing.
    r : (n_points,) array
        Radius of the wall, decreasing from a to a'.
    """

    th, ap = acceptance, exit_radius
    f = ap*(1 + np.sin(th))
    phi = np.linspace(2*th, th + np.pi/2, n_points)
    r = 2*f*np.sin(phi - th)/(1 - np.cos(phi)) - ap
    zp = 2*f*np.cos(phi - th)/(1 - np.cos(phi))
    L = zp[0]
    z = L - zp
    z[0], z[-1] = 0.0, L
    r[0], r[-1] = ap/np.sin(th), ap

    return z, r


class CPC(object):
    """Compound parabolic concentrator: mirror or dielectric (TIR), 2D or 3D.

    The entrance aperture (radius, or half width, a) lies in the plane
    z = z0, the exit aperture (a' = a*sin(theta)) at z0 + L; light enters
    travelling in +z. A "3d" CPC is a surface of revolution about the z
    axis, a "2d" CPC is a trough with walls at x = +-R(z), unbounded in y.
    The ideal concentration is 1/sin(theta) (2D) or 1/sin(theta)**2 (3D),
    the limit of `concentration_limit()`: the 2D CPC is an ideal
    concentrator, the 3D CPC rejects some skew rays inside theta and
    accepts some outside.

    A dielectric CPC (index n) is filled with a solid; the acceptance angle
    is given outside (in the medium n_in in front of it), the profile is
    designed for the refracted angle asin(n_in*sin(theta)/n), and the
    concentration is n**2/sin(theta)**2 (3D) relative to n_in = 1. Its walls
    reflect by total internal reflection (rays that do not meet the TIR
    condition leave and are lost) unless mirror is True.

    Parameters
    ----------
    acceptance : number
        Acceptance half angle theta in front of the entrance (radians).
    exit_radius : number
        Radius (half width in 2D) a' of the exit aperture.
    n : number, optional
        Refractive index inside (1: a hollow mirror CPC).
    kind : {"3d", "2d"}, optional
        Surface of revolution or trough.
    n_in : number, optional
        Index in front of the entrance.
    truncation : number, optional
        Cut the CPC at this entrance radius (between a' and a) to shorten
        it; the cut part of the profile is removed.
    mirror : bool, optional
        Reflecting walls (default: True for n = 1, TIR for n > 1).
    reflectivity : number, optional
        Reflectance of mirror walls (weight per reflection).
    z0 : number, optional
        Position of the entrance plane.
    n_points : int, optional
        Number of profile points (the wall is interpolated linearly
        between them).
    """

    def __init__(self, acceptance, exit_radius, n=1.0, kind="3d", n_in=1.0, truncation=None,
                 mirror=None, reflectivity=1.0, z0=0.0, n_points=20001):
        if kind not in ("3d", "2d"):
            raise ValueError("kind must be '3d' or '2d'.")
        self.acceptance = acceptance
        self.exit_radius = exit_radius
        self.n = n
        self.kind = kind
        self.n_in = n_in
        self.mirror = (n == 1.0) if mirror is None else mirror
        self.reflectivity = reflectivity
        self.z0 = z0
        self.design_angle = np.arcsin(n_in*np.sin(acceptance)/n)
        z, r = cpc_profile(self.design_angle, exit_radius, n_points)
        if truncation is not None:
            if not exit_radius < truncation <= r[0]:
                raise ValueError("truncation must lie between the exit and the entrance radius.")
            i = np.searchsorted(-r, -truncation)
            zc = np.interp(truncation, r[::-1], z[::-1])
            z = np.concatenate([[zc], z[i:]]) - zc
            r = np.concatenate([[truncation], r[i:]])
        self._z, self._r = z, r
        # wall radius and slope on a uniform grid in z (linear interpolation):
        self._zg = np.linspace(0.0, z[-1], n_points)
        self._z_step = self._zg[1]
        self._rg = np.interp(self._zg, z, r)
        self._sg = np.gradient(self._rg, self._zg)

    def __repr__(self):
        return "CPC(acceptance={}, exit_radius={}, n={}, kind={!r}, length={:.4g})".format(
            self.acceptance, self.exit_radius, self.n, self.kind, self.length)

    @property
    def entrance_radius(self):
        """Radius (half width) of the entrance aperture."""
        return self._r[0]

    @property
    def length(self):
        """Length of the CPC."""
        return self._z[-1]

    @property
    def geometric_concentration(self):
        """Entrance over exit area (3D) or width (2D)."""
        ratio = self.entrance_radius/self.exit_radius
        return ratio**2 if self.kind == "3d" else ratio

    @property
    def ideal_concentration(self):
        """Concentration limit for the acceptance angle (relative to n_in)."""
        return concentration_limit(self.acceptance, self.n/self.n_in, 3 if self.kind == "3d" else 2)

    def radius(self, z):
        """Wall radius at the distance z from the entrance."""
        u = np.clip(np.asarray(z, float)/self._z_step, 0, len(self._rg) - 1)
        i = np.minimum(u.astype(int), len(self._rg) - 2)
        f = u - i

        return (1 - f)*self._rg[i] + f*self._rg[i + 1]

    def profile(self):
        """Profile points (z from the entrance, r)."""
        return self._z.copy(), self._r.copy()

    def _rho(self, P):
        if self.kind == "3d":
            return np.hypot(P[..., 0], P[..., 1])
        return np.abs(P[..., 0])

    def _normal(self, Q):
        """Inward unit normal of the wall at the points Q (local z)."""
        s = np.interp(Q[:, 2], self._zg, self._sg)
        if self.kind == "3d":
            rho = np.hypot(Q[:, 0], Q[:, 1])
            ex, ey = Q[:, 0]/rho, Q[:, 1]/rho
        else:
            ex, ey = np.sign(Q[:, 0]), np.zeros(len(Q))
        nv = np.column_stack([-ex, -ey, s])  # inward: towards the axis, and along +z by the slope
        return nv/np.linalg.norm(nv, axis=1)[:, None]

    def trace(self, points, directions, max_reflections=500, n_steps=48, n_out=None, paths=False):
        """Trace rays from the entrance plane through the CPC.

        Parameters
        ----------
        points : (N, 3) array
            Points on the entrance plane (only x, y are used).
        directions : (N, 3) array
            Unit directions in front of the entrance (medium n_in).
        max_reflections : int, optional
            Rays with more reflections are counted as lost.
        n_steps : int, optional
            Samples per ray segment to bracket the wall hit (refined by
            bisection).
        n_out : number, optional
            Index behind the exit face (default: n, the absorber is in
            optical contact); rays totally reflected at the exit face are
            lost.
        paths : bool, optional
            Also return the vertices of every ray (for drawing).

        Returns
        -------
        CPCTrace
            Exit points and directions (NaN unless transmitted), the
            transmitted flag, the ray weights (the reflectance of mirror
            walls), the number of wall reflections and the fate of every
            ray: 0 transmitted, 1 turned back out of the entrance, 2 escaped
            through the wall (TIR failed), 3 too many reflections, 4 missed
            the entrance, 5 TIR at the exit face.
        vertices : (M, N, 3) array
            Only if paths is True: the entrance point, the wall hits and the
            end point of every ray (repeated after it has ended), in the
            frame of the CPC (entrance at z = 0).
        """

        P0 = np.asarray(points, float).reshape(-1, 3)
        N = len(P0)
        z_end = self.length
        fate = np.full(N, -1)
        P = np.column_stack([P0[:, 0], P0[:, 1], np.zeros(N)])
        d = refract_plane(directions, self.n_in, self.n)
        fate[self._rho(P) > self.entrance_radius*(1 + 1e-12)] = 4
        fate[~np.all(np.isfinite(d), axis=1)] = 4
        k = np.zeros(N, dtype=int)
        w = np.ones(N)
        c_tir = np.sqrt(max(0.0, 1 - (1.0/self.n)**2)) if not self.mirror else None
        active = fate < 0
        vertices = [P.copy()]
        if use_numba and _illumination_numba.available and not paths:
            _illumination_numba.trace(P, d, fate, k, w, self._z_step, self._rg, self._sg, z_end,
                                      self.kind == "3d", self.mirror, -1.0 if c_tir is None else c_tir,
                                      self.reflectivity, max_reflections, n_steps,
                                      4*(self.length + 2*self.entrance_radius))
            active = fate < 0
        while np.any(active):
            idx = np.nonzero(active)[0]
            Pa, da = P[idx], d[idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                t_end = np.where(da[:, 2] > 0, (z_end - Pa[:, 2])/da[:, 2],
                                 np.where(da[:, 2] < 0, -Pa[:, 2]/da[:, 2], np.inf))
            # cap the search for rays (nearly) parallel to the planes:
            t_end = np.minimum(t_end, 4*(self.length + 2*self.entrance_radius))
            ts = np.linspace(0, 1, n_steps + 1)[None, 1:]*t_end[:, None]
            Q = Pa[:, None, :] + ts[..., None]*da[:, None, :]
            g = self.radius(np.clip(Q[..., 2], 0, z_end)) - self._rho(Q)
            out = g < 0
            hit = np.any(out, axis=1)
            j = np.argmax(out, axis=1)
            # rays that reach a plane without hitting the wall:
            done = idx[~hit]
            dz = da[~hit, 2]
            Pend = Pa[~hit] + t_end[~hit, None]*da[~hit]
            P[done] = Pend
            fate[done] = np.where(dz > 0, 0, 1)
            # refine the hits by bisection between the last inside and the first outside sample:
            hi_idx = idx[hit]
            jh = j[hit]
            th = ts[hit]
            lo = np.where(jh > 0, th[np.arange(len(jh)), np.maximum(jh - 1, 0)], 0.0)
            hi = th[np.arange(len(jh)), jh]
            Ph, dh = Pa[hit], da[hit]
            for _ in range(50):
                mid = 0.5*(lo + hi)
                Qm = Ph + mid[:, None]*dh
                inside = self.radius(np.clip(Qm[:, 2], 0, z_end)) - self._rho(Qm) >= 0
                lo = np.where(inside, mid, lo)
                hi = np.where(inside, hi, mid)
            Qh = Ph + lo[:, None]*dh
            nv = self._normal(Qh)
            cos_i = -np.sum(dh*nv, axis=1)  # nv points inwards, rays hit from inside: d.nv < 0
            if c_tir is not None:
                # TIR needs sin(i) > 1/n, i.e. |cos i| < sqrt(1 - 1/n**2)
                escaped = np.abs(cos_i) >= c_tir
            else:
                escaped = np.zeros(len(hi_idx), dtype=bool)
                w[hi_idx] *= self.reflectivity
            d_new = dh + 2*cos_i[:, None]*nv
            d_new /= np.linalg.norm(d_new, axis=1)[:, None]
            P[hi_idx] = Qh
            d[hi_idx] = d_new
            k[hi_idx] += 1
            fate[hi_idx[escaped]] = 2
            fate[hi_idx[(k[hi_idx] > max_reflections) & ~escaped]] = 3
            active = fate < 0
            if paths:
                vertices.append(P.copy())
        n_out = self.n if n_out is None else n_out
        trans = fate == 0
        d_out = np.full((N, 3), np.nan)
        P_out = np.full((N, 3), np.nan)
        if np.any(trans):
            d_t = refract_plane(d[trans], self.n, n_out)
            tir = ~np.all(np.isfinite(d_t), axis=1)
            ti = np.nonzero(trans)[0]
            fate[ti[tir]] = 5
            d_out[ti[~tir]] = d_t[~tir]
            P_out[ti[~tir]] = P[ti[~tir]]
        trans = fate == 0
        P_out[:, 2] += self.z0
        w[~trans] = 0.0

        result = CPCTrace(P_out, d_out, trans, w, k, fate)
        if paths:
            return result, np.array(vertices)

        return result


CPCTrace = namedtuple("CPCTrace", ["points", "directions", "transmitted", "weights",
                                    "n_reflections", "fate"])


def tilted_beam(n_rays, half_width, angle, azimuth=0.0, kind="3d", rng=None):
    """Parallel rays at the polar angle `angle` over an entrance aperture.

    Parameters
    ----------
    n_rays : int
        Number of rays.
    half_width : number
        Radius of the disk (kind "3d") or half width of the strip in x
        (kind "2d", y = 0) that the rays fill uniformly.
    angle : number
        Angle of the rays with the z axis.
    azimuth : number, optional
        Azimuth of the direction (0: in the x-z plane).
    kind : {"3d", "2d"}, optional
        Disk or strip.
    rng : None, int or numpy.random.Generator, optional
        Random generator or seed.

    Returns
    -------
    points, directions : (n_rays, 3) arrays
    """

    rng = _rng(rng)
    if kind == "3d":
        xy = disk_points(n_rays, half_width, rng)
    else:
        xy = np.column_stack([(2*rng.random(n_rays) - 1)*half_width, np.zeros(n_rays)])
    P = np.column_stack([xy, np.zeros(n_rays)])
    d = np.tile([np.sin(angle)*np.cos(azimuth), np.sin(angle)*np.sin(azimuth), np.cos(angle)],
                (n_rays, 1))

    return P, d


def reverse_system(system):
    """The same centred system traced backwards, from its image plane to its
    object plane.

    Ray paths are reversible, so the light that reaches a point of the
    image plane from a given direction can be found by tracing the reversed
    ray back to the source. With the brightness theorem this gives the
    irradiance of an illumination system without forward Monte Carlo:
    E(x) = L*integral over the directions (p, q) whose reversed rays hit
    the source (backward ray tracing). The reversed system is the mirror
    image z -> -z: surfaces in reverse order with opposite curvatures, conic
    and aspheric terms of opposite sign, the gaps in reverse order and the
    media of the forward pass in front of each surface. A forward ray
    arriving at (x, y) with the direction (L, M, N) is reversed by starting
    at (x, y) with the direction (-L, -M, N).

    Parameters
    ----------
    system : raytrace.System
        Centred system of plain `raytrace.Surface` objects (no tilts,
        decentres, ideal lenses or coordinate breaks) with a finite object
        distance.

    Returns
    -------
    raytrace.System
        Its object plane is the image plane of the original, its image plane
        the original object plane.
    """

    import copy
    from .raytrace import Surface, System

    surfaces = system.surfaces
    if np.isinf(system.object_distance):
        raise ValueError("The object distance must be finite.")
    for s in surfaces:
        if type(s) is not Surface or any(s.decenter) or any(s.tilt):
            raise ValueError("Only centred plain Surfaces can be reversed.")
    n = system.n  # indices in front of each surface and in image space
    out = []
    for i in reversed(range(len(surfaces))):
        s = copy.copy(surfaces[i])
        s.R = -s.R
        s.aspheric = tuple(-a for a in s.aspheric)
        s.n_after = n[i]
        s.material = None
        s.thickness = surfaces[i - 1].thickness if i > 0 else system.object_distance
        out.append(s)

    return System(out, len(surfaces) - 1 - system.stop_index, system.wavelength,
                  object_distance=surfaces[-1].thickness, object_height=1.0, n_object=n[-1],
                  units=system.units)


def backward_trace(system, points, p, q, source, reversed_system=None):
    """Radiance arriving at image plane points from the directions (p, q),
    found by tracing the rays backwards to the source.

    Parameters
    ----------
    system : raytrace.System
        The forward (illumination) system, from the source in its object
        plane to its image plane.
    points : (N, 2) array
        Points (x, y) in the image plane.
    p, q : (N,) arrays
        Optical direction cosines of the light arriving there (image space).
    source : callable
        Basic radiance L(x, y) of the source in its plane, e.g. 1 on a lamp
        filament and 0 elsewhere (booleans work).
    reversed_system : raytrace.System, optional
        `reverse_system(system)`, to reuse it between calls.

    Returns
    -------
    L : (N,) array
        Radiance carried by each ray (0 for rays that are vignetted or miss
        the source).
    """

    R = reverse_system(system) if reversed_system is None else reversed_system
    xy = np.atleast_2d(np.asarray(points, float))
    n_img = R.n_object
    P = np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), -R.object_distance)])
    d = directions_from_sines(-np.asarray(p, float), -np.asarray(q, float), n_img)
    B = R.trace(Rays(P, d, np.zeros(len(xy))))
    Q = R.to_image_frame(B.points[-1])
    ok = np.isfinite(Q[:, 0])
    L = np.zeros(len(xy))
    L[ok] = np.asarray(source(Q[ok, 0], Q[ok, 1]), float)

    return L


def backward_irradiance(system, points, na, source, n_directions=2000, rng=None,
                        reversed_system=None):
    """Irradiance at image plane points by backward ray tracing.

    By the brightness theorem the irradiance at a point is
    E = integral L(p, q) dp dq over the directions of the arriving light
    (in a medium of index 1; generally divided by n**2), with L the basic
    radiance of the source point that the reversed ray hits. The integral
    is sampled with n_directions random directions per point, uniform in
    the disk p**2 + q**2 <= na**2 (which must contain all the light).

    Parameters
    ----------
    system : raytrace.System
        Forward illumination system.
    points : (N, 2) array
        Points (x, y) in the image plane.
    na : number
        Largest NA of the arriving light.
    source : callable
        Basic radiance L(x, y) of the source.
    n_directions : int, optional
        Directions per point.
    rng : None, int or numpy.random.Generator, optional
        Random generator or seed.
    reversed_system : raytrace.System, optional
        `reverse_system(system)`.

    Returns
    -------
    E : (N,) array
        Irradiance in units of the source radiance (E = pi*na**2 for a
        point that sees the source in every direction up to na).
    """

    rng = _rng(rng)
    R = reverse_system(system) if reversed_system is None else reversed_system
    xy = np.atleast_2d(np.asarray(points, float))
    X = np.repeat(xy, n_directions, axis=0)
    pq = disk_points(len(X), na, rng)
    L = backward_trace(system, X, pq[:, 0], pq[:, 1], source, reversed_system=R)

    return np.pi*na**2*L.reshape(len(xy), n_directions).mean(axis=1)


# homogenizers: lenslet arrays (fly's eye)

def thin_lens(points, directions, focal_length, center=(0.0, 0.0)):
    """Refract rays at an ideal thin lens in the plane of the points.

    The lens bends every ray towards the point where the parallel ray
    through its centre meets the focal plane: the tangents of the ray
    change by -(x - x_c)/f and -(y - y_c)/f. This is the paraxial lens of
    lens design programs (`raytrace.IdealLens`); it images without
    aberrations but does not conserve étendue at large angles (notebook
    83, section 3), which does not matter for the small lenslet apertures
    of a homogenizer.

    Parameters
    ----------
    points, directions : (N, 3) arrays
        Rays at the lens plane.
    focal_length : number or (N,) array
        Focal length in the medium of the rays.
    center : (2,) or (N, 2) array, optional
        Centre of the lens, per ray if needed.

    Returns
    -------
    directions : (N, 3) array
        Unit directions behind the lens.
    """

    P = np.asarray(points, float).reshape(-1, 3)
    d = np.asarray(directions, float).reshape(-1, 3)
    c = np.broadcast_to(np.asarray(center, float), (len(P), 2))
    f = np.asarray(focal_length, float)
    tx = d[:, 0]/d[:, 2] - (P[:, 0] - c[:, 0])/f
    ty = d[:, 1]/d[:, 2] - (P[:, 1] - c[:, 1])/f
    norm = np.sqrt(1 + tx**2 + ty**2)
    return np.column_stack([tx/norm, ty/norm, 1/norm])


def lenslet_array(points, directions, pitch, focal_length, n_lenslets=None, offset=(0.0, 0.0)):
    """Refract rays at a square array of ideal thin lenslets.

    Each ray is refracted by the lenslet it meets (thin_lens() about the
    lenslet centre). This is the element of a fly's eye homogenizer: two
    such arrays one focal length apart, followed by a condenser, image every
    lenslet aperture of the first array onto the same field.

    Parameters
    ----------
    points, directions : (N, 3) arrays
        Rays at the array plane.
    pitch : number or (2,) sequence
        Lenslet pitch (x, y). Lenslets are square or rectangular and fill
        the plane.
    focal_length : number
    n_lenslets : int or (2,) sequence, optional
        Number of lenslets along x and y, centred on the axis (plus
        offset); rays outside the array are lost (NaN). Default: unlimited.
    offset : (2,) sequence, optional
        Shift of the array (x, y).

    Returns
    -------
    directions : (N, 3) array
        Unit directions behind the array (NaN for lost rays).
    index : (N, 2) int array
        Lenslet indices (i, j) of the rays, counted from the axis (0 is the
        lenslet on the axis for an odd number, or the one right of it).
    """

    P = np.asarray(points, float).reshape(-1, 3)
    p = np.broadcast_to(np.asarray(pitch, float), (2,))
    o = np.broadcast_to(np.asarray(offset, float), (2,))
    n = None if n_lenslets is None else np.broadcast_to(np.asarray(n_lenslets), (2,))
    # odd counts: lenslet centred on the axis; even counts: an edge on the axis
    shift = np.zeros(2) if n is None else np.where(n % 2 == 0, 0.5, 0.0)
    u = (P[:, :2] - o)/p - shift
    idx = np.round(np.nan_to_num(u)).astype(int)
    centers = (idx + shift)*p + o
    d = thin_lens(P, directions, focal_length, centers)
    if n is not None:
        half = n/2
        with np.errstate(invalid="ignore"):
            outside = np.any(np.abs(u + shift) > half, axis=1)
        d[outside] = np.nan
    return d, idx


def fly_eye_field_1d(field, x, wavelength, pitch, f_1, f_2=None, gap=None, focal_length=100.0,
                     n_lenslets=None, n=1.0):
    """Field in the focal plane of the condenser of a fly's eye homogenizer
    (scalar wave optics, one transverse dimension).

    The input field passes the first lenslet array (focal length f_1),
    propagates over the gap (angular spectrum, exact), passes the second
    array (f_2; None for a single array, a non-imaging homogenizer) and an
    ideal condenser of focal length F, in whose back focal plane the field
    is the Fourier transform u = wavelength*F*nu.

    Parameters
    ----------
    field : 1d array
        Complex input field on the equidistant grid x.
    x : 1d array
    wavelength : number
        Vacuum wavelength.
    pitch : number
        Lenslet pitch.
    f_1 : number
        Focal length of the first array.
    f_2 : number, optional
        Focal length of the second array (imaging homogenizer).
    gap : number, optional
        Distance between the arrays, default f_2 (the second array in the
        focal plane of the first, as in an imaging homogenizer with f_1 =
        f_2).
    focal_length : number, optional
        Focal length F of the condenser.
    n_lenslets : int, optional
        Number of lenslets (centred); the field outside is blocked.
    n : number, optional
        Refractive index between the arrays.

    Returns
    -------
    u : 1d array
        Coordinates in the focal plane of the condenser (fftshifted).
    U : 1d array
        Complex field there, normalized so that ``sum |U|**2 du`` equals
        ``sum |field|**2 dx`` of the light the array passes.
    """

    x = np.asarray(x, float)
    E = np.asarray(field, complex)
    dx = x[1] - x[0]
    k = 2*np.pi/wavelength

    def screen(f):
        shift = 0.0 if n_lenslets is None or n_lenslets % 2 else 0.5
        c = (np.round(x/pitch - shift) + shift)*pitch
        t = np.exp(-1j*k*(x - c)**2/(2*f))
        if n_lenslets is not None:
            t[np.abs(x) > n_lenslets*pitch/2] = 0.0
        return t

    E = E*screen(f_1)
    if f_2 is not None:
        g = f_2 if gap is None else gap
        nu = np.fft.fftfreq(len(x), dx)
        kz = np.sqrt((n*k)**2 - (2*np.pi*nu)**2 + 0j)
        E = np.fft.ifft(np.fft.fft(E)*np.exp(1j*kz*g))
        E = E*screen(f_2)
    elif gap is not None:
        raise ValueError("gap needs f_2")
    nu = np.fft.fftshift(np.fft.fftfreq(len(x), dx))
    U = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E)))*dx
    u = wavelength*focal_length*nu
    du = u[1] - u[0]
    U = U/np.sqrt(np.sum(np.abs(U)**2)*du/(np.sum(np.abs(E)**2)*dx))
    return u, U
