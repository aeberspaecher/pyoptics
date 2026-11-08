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
  reach (`SUN_ANGULAR_RADIUS`, `concentration_temperature()`).

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
"""

import numpy as np

from .raytrace import Rays


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
