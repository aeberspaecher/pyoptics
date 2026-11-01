#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Conversion between ray ensembles and sampled wave fields.

Geometrical optics describes a wave field by its rays: the eikonal S (the
optical path, OPL) gives the phase k_0*S, the rays are the normals of its
wavefronts, and the transport equation div(I grad S) = 0 says that the power
in a tube of rays is conserved. This module goes back and forth between the
two descriptions.

Rays to waves
-------------

Ray field summation (rays_to_field()). Rays launched on a structured grid
(e.g. a grid of pupil coordinates) are traced to a plane; the grid of their
intersections is cut into triangles, each a small tube of rays. A triangle
carries the power of its launch triangle, so its irradiance in the plane is
the launch irradiance times the ratio of the launch and target areas (the
inverse Jacobian of the ray map, Born and Wolf ch. 3.1). Inside the
triangle the phase is interpolated from the eikonal S and its gradient
n*d_perp at the three rays (the mean of the linear interpolation and of the
tangent planes, which is exact for quadratic wavefronts), the field of the
triangle is ::

    u = sqrt(I_launch*|A_launch/A_target|)*exp(i*k_0*S)*exp(-i*pi*m/2),

and the fields of all triangles that cover a sample are added. Where the ray
map folds (behind a caustic), several sheets of triangles cover the same
point and interfere; m, the Maslov (KMAH) index, counts the caustics that the
tube has crossed: the number of negative eigenvalues of the 2 x 2 affine map
from the launch triangle to the target triangle (1 at a line focus, 2 at a
point focus: the Gouy phase -pi). This is the classical asymptotic field. It
is exact far from caustics, where the field is locally a sum of plane waves
with slowly varying amplitude, and it diverges on the caustic itself
(A_target -> 0), where the rays touch and geometrical optics breaks down; in
the shadow of a caustic it gives zero instead of the evanescent Airy tail
(Kravtsov and Orlov 1993).

Gaussian beamlets (GaussianBeamlets). The standard remedy is to give every
ray a finite width: the field in a launch plane is written as a sum of
Gaussian beamlets centred on the rays, each with the local tilt and
curvature of the wavefront (Červený, Popov and Pšenčík 1982; Arnaud 1985;
the Gaussian beam decomposition of lens design programs). A frame of
Gaussians exp(-r**2/w**2) on a grid of spacing Delta adds up to
pi*w**2/Delta**2 (to better than 1e-8 for w >= 1.5*Delta), so the beamlet of
a sample with complex amplitude A and the area Delta**2 has the amplitude
A*Delta**2/(pi*w**2) and ::

    u_j(x) = a_j*exp(i*(phi_j + k_j.xi + xi.Gamma_j.xi/2)),   xi = x - x_j,
    Gamma_j = H_j + 2i/w_j**2,

with the transverse wave vector k_j and the Hessian H_j of the phase. In a
homogeneous medium the angular spectrum of every beamlet is propagated in
closed form with k_z expanded to second order about k_j (k_z ~ k_z0 + g.dk +
dk.G.dk/2, any tilt): ::

    u_j(x, z) = a_j/sqrt(det(I - z*Gamma_j*G))
                *exp(i*(phi_j + z*k_z0 + k_j.xi + eta.Gamma_z.eta/2)),
    eta = xi + z*g = xi - z*k_j/k_z0,   Gamma_z = Gamma_j*(I - z*G*Gamma_j)^-1.

The centre moves along the ray, the width and curvature follow the complex
q parameter, and the square root (taken on the continuous branch: the
eigenvalues of Gamma*G have negative imaginary parts) gives the Gouy phase.
The beamlets stay finite at foci and caustics, so their sum is the field
everywhere, as long as the beamlets are narrow compared to the scale on
which the wavefront curvature changes and wide compared to the wavelength.
Beamlets can be launched from rays (GaussianBeamlets.from_rays(), curvature
from neighbouring rays) or from a sampled field
(GaussianBeamlets.from_field(), also from a QuadraticPhaseField of
pyoptics.analytic_phase, whose analytic phase supplies tilt and curvature).

GaussianBeamlets.trace() carries the beamlets through the surfaces of a
raytrace.System (Gaussian beam decomposition through real optics). Every
beamlet is followed by its base ray and eight parabasal rays (centre and
direction shifted in x and y), traced exactly. Their central differences in
the output plane give the 4 x 4 ray transfer matrix M = [[A, B], [C, D]] of
the neighbourhood of the base ray in the canonical coordinates (x, k_perp),
which is symplectic, and the beamlet follows the complex ABCD law (a
Gaussian beamlet is the family of complex rays k_perp = Gamma*xi, Arnaud
1985) ::

    Gamma' = (C + D Gamma)(A + B Gamma)^-1,   a' = a/sqrt(det(A + B Gamma)),

with the phase k_0*OPL of the base ray. det(A + B Gamma) never vanishes,
but its argument grows by pi through every line focus: the square root is
taken on the branch that follows the beamlet (its Gouy phase), accumulated
in closed form along every straight segment of the base ray. Beamlets
whose rays are vignetted are dropped, so an aperture inside a system is
resolved to the beamlet size; for a sharp cut, sum the beamlets in the
aperture plane and decompose the field again with from_field().

Waves to rays
-------------

Phase gradient (field_to_rays()). The time averaged energy flux of a scalar
field, the Poynting vector of the paraxial vector field, is proportional to
``Im(conj(u)*grad u) = |u|**2 grad(phase)``: the local ray direction is the
phase gradient, k_perp = grad(phase), completed with
``k_z = sqrt(n**2 k_0**2 - |k_perp|**2)``, each sample becomes a ray with the
power ``|u|**2 dx dy`` and the optical path phase/k_0. The gradient is computed
from the phase differences of neighbouring samples (no unwrapping needed for
the directions). The result can be traced with raytrace.System.trace(). It
is the right picture where the field is locally a single plane wave; inside
a caustic, where several sheets overlap, the local phase gradient is an
average direction weighted with the interference pattern and need not
describe any ray.

Local spectra and Wigner functions (local_spectrum_rays(), wigner()). Phase
space distributions assign several directions to one point. The Wigner
function ::

    W(x, k) = int u(x + s/2) conj(u(x - s/2)) exp(-i*k*s) ds

(here in 1D, for a section of a field) has the intensity and the spectrum
as marginals, is concentrated on the Lagrangian manifold of the ray family,
k = dS/dx, which folds at caustics, and oscillates (negative values)
between the sheets (Bastiaans 1979; Alonso 2011). Its smoothed, positive
version is the spectrogram (Husimi function): the spectrum of the field in a
Gaussian window. local_spectrum_rays() samples it in 2D: every window
centre and every spectral component above a threshold is a ray, with the
power of that component. It resolves crossing ray sheets at the price of the
uncertainty relation (position to the window size, direction to
2*pi/window), and the rays carry no phase (an incoherent ray set).

Helpers: straight_rays() moves rays to another plane of a homogeneous
medium, ray_irradiance() bins ray powers into an irradiance map.

Conventions as in pyoptics.propagators and pyoptics.raytrace: fields vary as
exp(i(k.r - omega*t)), the phase of the field is +k_0*OPL (vacuum
wavenumber times the optical path, which includes the refractive index),
the light travels in +z, ray directions are unit vectors. Fields are
arrays of shape (Ny, Nx) on linear coordinates x, y. ``|u|**2`` is the
irradiance on the plane (power per area of the plane, not per area normal to
the ray).

References
----------
M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999), ch. 3.

Yu. A. Kravtsov and Yu. I. Orlov, "Caustics, Catastrophes and Wave Fields",
Springer (1993).

V. P. Maslov and M. V. Fedoriuk, "Semi-Classical Approximation in Quantum
Mechanics", Reidel (1981).

V. Červený, M. M. Popov and I. Pšenčík, "Computation of wave fields in
inhomogeneous media - Gaussian beam approach", Geophys. J. R. Astr. Soc. 70,
109-128 (1982).

J. A. Arnaud, "Representation of Gaussian beams by complex rays", Appl.
Opt. 24, 538-543 (1985).

A. W. Greynolds, "Propagation of generally astigmatic Gaussian beams along
skew ray paths", Proc. SPIE 560, 33-50 (1985).

M. J. Bastiaans, "Wigner distribution function and its application to
first-order optics", J. Opt. Soc. Am. 69, 1710-1716 (1979).

M. A. Alonso, "Wigner functions in optics: describing beams as ray bundles
and pulses as particle ensembles", Adv. Opt. Photon. 3, 272-365 (2011).

G. W. Forbes and M. A. Alonso, "Using rays better. I. Theory for smoothly
varying media", J. Opt. Soc. Am. A 18, 1132-1145 (2001).
"""

from collections import namedtuple

import numpy as np

from .utils import wavenumber


__all__ = ["FieldRays", "straight_rays", "ray_irradiance", "rays_to_field", "GaussianBeamlets",
           "field_to_rays", "local_spectrum_rays", "wigner"]


FieldRays = namedtuple("FieldRays", ["points", "directions", "opl", "power"])
FieldRays.__doc__ = """Rays from a field: start points and unit directions,
shape (N, 3), optical path (N,) (NaN if the rays carry no phase) and power
(N,). Has the attributes of raytrace.Rays, so it can be passed to
System.trace() directly."""


def _grid(x, name):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("{} must be a 1D array of at least two coordinates".format(name))
    d = np.diff(x)
    if not np.allclose(d, d[0], rtol=1e-6, atol=0) or d[0] <= 0:
        raise ValueError("{} must be linear, ascending and equidistant".format(name))
    return x


def straight_rays(points, directions, opl, z, n=1.0):
    """Move rays along straight lines to the plane z of a homogeneous medium.

    Parameters
    ----------
    points, directions : arrays, shape (..., 3)
        Positions and unit directions.
    opl : array, shape (...)
        Optical path at the positions.
    z : number
        z of the target plane.
    n : number, optional
        Refractive index of the medium.

    Returns
    -------
    points, opl : arrays
        Intersections with the plane and the optical path there (the path is
        negative for rays that travel backwards to reach the plane).
    """

    points = np.asarray(points, dtype=float)
    directions = np.asarray(directions, dtype=float)
    t = (z - points[..., 2])/directions[..., 2]
    return points + t[..., None]*directions, np.asarray(opl, dtype=float) + n*t


def ray_irradiance(points, power, x_edges, y_edges):
    """Irradiance of a ray ensemble: the powers binned on a grid, per area.

    Parameters
    ----------
    points : array, shape (..., 2) or (..., 3)
        Ray positions in the plane (x, y; z is ignored).
    power : array or number
        Power of each ray.
    x_edges, y_edges : arrays
        Bin edges.

    Returns
    -------
    E : array, shape (len(y_edges) - 1, len(x_edges) - 1)
        Power per area of the bins (the geometrical intensity).
    """

    P = np.asarray(points, dtype=float).reshape(-1, np.shape(points)[-1])
    w = np.broadcast_to(np.asarray(power, dtype=float), np.shape(points)[:-1]).reshape(-1)
    ok = np.isfinite(P[:, 0]) & np.isfinite(P[:, 1]) & np.isfinite(w)
    H = np.histogram2d(P[ok, 1], P[ok, 0], [y_edges, x_edges], weights=w[ok])[0]
    return H/np.outer(np.diff(y_edges), np.diff(x_edges))


# rays to field:

def _triangles(shape):
    """Vertex indices (T, 3) of the two triangles of every cell of a structured grid."""
    Nv, Nu = shape
    i = np.arange(Nv - 1)[:, None]*Nu + np.arange(Nu - 1)[None, :]
    i = i.ravel()
    a, b, c, d = i, i + 1, i + Nu + 1, i + Nu  # (v, u), (v, u+1), (v+1, u+1), (v+1, u)
    return np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def _maslov(M):
    """Number of negative eigenvalues of real 2 x 2 matrices M (..., 2, 2):
    1 for det < 0, 2 for det > 0 and a negative trace (a complex pair with
    negative real part, a rotated point focus, also counts 2)."""
    det = M[..., 0, 0]*M[..., 1, 1] - M[..., 0, 1]*M[..., 1, 0]
    tr = M[..., 0, 0] + M[..., 1, 1]
    return np.where(det < 0, 1, np.where(tr < 0, 2, 0))


def rays_to_field(x, y, points, directions, opl, wavelength, launch=None, irradiance=1.0, n=1.0,
                  gouy=True, max_pairs=4_000_000):
    """Sampled field from rays on a structured grid (ray field summation).

    Every cell of the ray grid is split into two triangles (tubes of rays).
    A triangle carries the power of its launch triangle, its irradiance in
    the target plane is the launch irradiance times the area ratio, its phase
    is interpolated from the eikonal and the ray directions of its corners,
    and the fields of all triangles covering a sample are added (several
    sheets behind a caustic). See the module docstring.

    Parameters
    ----------
    x, y : arrays
        Linear, equidistant coordinates of the output grid.
    points : array, shape (Nv, Nu, 2) or (Nv, Nu, 3)
        Ray intersections with the target plane, on the structured launch
        grid (neighbouring rays are neighbours in the array). NaN for lost
        rays; their triangles are skipped.
    directions : array, shape (Nv, Nu, 3)
        Unit ray directions at the target plane.
    opl : array, shape (Nv, Nu)
        Optical paths at the target plane.
    wavelength : number
        Vacuum wavelength.
    launch : array, shape (Nv, Nu, 2), optional
        Positions of the rays in the launch plane (e.g. entrance pupil
        coordinates in length units, or the start points). Default: the
        array indices (u, v), so that irradiance is the power per grid cell.
    irradiance : number or array (Nv, Nu), optional
        Irradiance at the launch positions (power per launch area).
    n : number, optional
        Refractive index at the target plane.
    gouy : bool, optional
        Include the phase -pi/2 per crossed caustic (Maslov index).
    max_pairs : int, optional
        Triangle-sample pairs evaluated at once (memory).

    Returns
    -------
    u : array, shape (len(y), len(x))
        Complex field with ``|u|**2`` the geometrical irradiance and phase
        k_0*OPL.
    """

    x, y = _grid(x, "x"), _grid(y, "y")
    P = np.asarray(points, dtype=float)
    if P.ndim != 3 or P.shape[-1] not in (2, 3):
        raise ValueError("points must have the shape (Nv, Nu, 2) or (Nv, Nu, 3)")
    shape = P.shape[:2]
    P = P[..., :2].reshape(-1, 2)
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    S = np.asarray(opl, dtype=float).reshape(-1)
    if launch is None:
        V, U = np.mgrid[0:shape[0], 0:shape[1]]
        Q = np.stack([U, V], -1).reshape(-1, 2).astype(float)
    else:
        Q = np.asarray(launch, dtype=float)[..., :2].reshape(-1, 2)
    I0 = np.broadcast_to(np.asarray(irradiance, dtype=float), shape).reshape(-1)
    k0 = 2*np.pi/wavelength

    tri = _triangles(shape)
    finite = np.isfinite(P).all(1) & np.isfinite(d).all(1) & np.isfinite(S) & np.isfinite(Q).all(1)
    tri = tri[finite[tri].all(1)]
    # affine map launch -> target per triangle
    Ep = np.stack([P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]]], -1)  # columns
    Eq = np.stack([Q[tri[:, 1]] - Q[tri[:, 0]], Q[tri[:, 2]] - Q[tri[:, 0]]], -1)
    det_q = Eq[:, 0, 0]*Eq[:, 1, 1] - Eq[:, 0, 1]*Eq[:, 1, 0]
    det_p = Ep[:, 0, 0]*Ep[:, 1, 1] - Ep[:, 0, 1]*Ep[:, 1, 0]
    keep = (det_q != 0) & (det_p != 0)
    tri, Ep, Eq, det_q, det_p = tri[keep], Ep[keep], Eq[keep], det_q[keep], det_p[keep]
    Eq_inv = np.stack([np.stack([Eq[:, 1, 1], -Eq[:, 0, 1]], -1),
                       np.stack([-Eq[:, 1, 0], Eq[:, 0, 0]], -1)], -2)/det_q[:, None, None]
    M = Ep @ Eq_inv
    amp = np.sqrt(I0[tri].mean(1)*np.abs(det_q/det_p))
    if gouy:
        amp = amp*np.exp(-0.5j*np.pi*_maslov(M))

    # orient the triangles counterclockwise for the fill rule
    flip = det_p < 0
    tri = tri.copy()
    tri[flip, 1], tri[flip, 2] = tri[flip, 2], tri[flip, 1].copy()

    dx, dy = x[1] - x[0], y[1] - y[0]
    Nx, Ny = len(x), len(y)
    Vx, Vy = P[tri, 0], P[tri, 1]  # (T, 3)
    ix0 = np.clip(np.ceil((Vx.min(1) - x[0])/dx - 1e-9), 0, Nx).astype(int)
    ix1 = np.clip(np.floor((Vx.max(1) - x[0])/dx + 1e-9), -1, Nx - 1).astype(int)
    iy0 = np.clip(np.ceil((Vy.min(1) - y[0])/dy - 1e-9), 0, Ny).astype(int)
    iy1 = np.clip(np.floor((Vy.max(1) - y[0])/dy + 1e-9), -1, Ny - 1).astype(int)
    nx = np.maximum(ix1 - ix0 + 1, 0)
    ny = np.maximum(iy1 - iy0 + 1, 0)
    counts = nx*ny
    # edges opposite to the vertices (counterclockwise) and their ownership
    # (top-left rule: samples on a shared edge are counted once)
    ex = np.stack([Vx[:, 2] - Vx[:, 1], Vx[:, 0] - Vx[:, 2], Vx[:, 1] - Vx[:, 0]], 1)
    ey = np.stack([Vy[:, 2] - Vy[:, 1], Vy[:, 0] - Vy[:, 2], Vy[:, 1] - Vy[:, 0]], 1)
    own = (ey < 0) | ((ey == 0) & (ex < 0))
    start = np.stack([np.roll(Vx, -1, 1), np.roll(Vy, -1, 1)], -1)  # vertex i+1 starts edge i
    area2 = np.abs(det_p)

    out = np.zeros(Nx*Ny, dtype=complex)
    order = np.cumsum(counts)
    first = 0
    while first < len(tri):
        base = order[first - 1] if first > 0 else 0
        last = np.searchsorted(order, base + max_pairs, side="right")
        last = max(last, first + 1)
        sl = slice(first, last)
        c = counts[sl]
        if c.sum() == 0:
            first = last
            continue
        t = np.repeat(np.arange(first, last), c)
        off = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        jx = ix0[t] + off % nx[t]
        jy = iy0[t] + off//nx[t]
        px, py = x[jx], y[jy]
        # edge functions: twice the signed areas of the sub triangles
        e = (ex[t]*(py[:, None] - start[t, :, 1]) - ey[t]*(px[:, None] - start[t, :, 0]))
        inside = np.all((e > 0) | ((e == 0) & own[t]), axis=1)
        t, px, py, e = t[inside], px[inside], py[inside], e[inside]
        lam = e/area2[t, None]
        v = tri[t]
        # eikonal: mean of the linear interpolation and of the corners' tangent
        # planes (n*d_perp = grad S), exact for quadratic wavefronts
        Sv = S[v] + 0.5*n*(d[v, 0]*(px[:, None] - P[v, 0]) + d[v, 1]*(py[:, None] - P[v, 1]))
        val = amp[t]*np.exp(1j*k0*np.sum(lam*Sv, 1))
        idx = jy[inside]*Nx + jx[inside]
        out += np.bincount(idx, val.real, Nx*Ny) + 1j*np.bincount(idx, val.imag, Nx*Ny)
        first = last
    return out.reshape(Ny, Nx)


# Gaussian beamlets:

def _inv2(A):
    det = A[..., 0, 0]*A[..., 1, 1] - A[..., 0, 1]*A[..., 1, 0]
    inv = np.empty_like(A)
    inv[..., 0, 0] = A[..., 1, 1]
    inv[..., 1, 1] = A[..., 0, 0]
    inv[..., 0, 1] = -A[..., 0, 1]
    inv[..., 1, 0] = -A[..., 1, 0]
    return inv/det[..., None, None]


def _nan_gradient(a):
    """Gradient along both axes of a 2D array with unit spacing: central
    differences, one sided next to NaN samples (lost rays) and at the edges."""
    out = []
    for axis in (0, 1):
        b = np.moveaxis(np.asarray(a, dtype=float), axis, 0)
        nxt = np.full_like(b, np.nan)
        prv = np.full_like(b, np.nan)
        nxt[:-1], prv[1:] = b[1:], b[:-1]
        g = 0.5*(nxt - prv)
        g = np.where(np.isfinite(g), g, nxt - b)
        g = np.where(np.isfinite(g), g, b - prv)
        out.append(np.moveaxis(g, 0, axis))
    return out


class GaussianBeamlets(object):
    """A field written as a sum of Gaussian beamlets, propagated in closed form.

    Beamlet j in the launch plane z0 ::

        u_j = a_j*exp(i*(phase_j + k_j.xi + xi.Gamma_j.xi/2)),  xi = (x, y) - centre_j,

    with the transverse wave vector k_j and the complex symmetric matrix
    Gamma_j = H_j + 2i/w_j**2 (H: Hessian of the wavefront phase, w: 1/e
    amplitude radius). field() evaluates the sum in any plane z of the
    homogeneous medium (see the module docstring). Usually built with
    from_rays() or from_field().

    Parameters
    ----------
    centres : array, shape (N, 2)
        Positions (x, y) in the launch plane.
    k_perp : array, shape (N, 2)
        Transverse wave vectors (k_x, k_y), ``|k_perp| < n*k_0``.
    phase : array, shape (N,)
        Phase at the centres.
    amplitude : array, shape (N,)
        Complex amplitudes a_j.
    Gamma : array, shape (N, 2, 2)
        Complex curvature matrices (positive definite imaginary part).
    wavelength : number
        Vacuum wavelength.
    n : number, optional
        Refractive index of the medium.
    z0 : number, optional
        z of the launch plane.
    """

    def __init__(self, centres, k_perp, phase, amplitude, Gamma, wavelength, n=1.0, z0=0.0):
        self.centres = np.asarray(centres, dtype=float).reshape(-1, 2)
        self.k_perp = np.asarray(k_perp, dtype=float).reshape(-1, 2)
        self.phase = np.asarray(phase, dtype=float).reshape(-1)
        self.amplitude = np.asarray(amplitude, dtype=complex).reshape(-1)
        self.Gamma = np.asarray(Gamma, dtype=complex).reshape(-1, 2, 2)
        self.wavelength = float(wavelength)
        self.n = n
        self.z0 = float(z0)
        k = self.k
        if np.any(np.sum(self.k_perp**2, 1) >= k**2):
            raise ValueError("GaussianBeamlets: |k_perp| must be smaller than n*k_0")

    @property
    def k(self):
        """Wavenumber in the medium."""
        return wavenumber(self.wavelength, self.n)

    def __len__(self):
        return len(self.amplitude)

    def __repr__(self):
        return "GaussianBeamlets({} beamlets, wavelength={:g}, n={}, z0={:g})".format(
            len(self), self.wavelength, self.n, self.z0)

    @classmethod
    def from_field(cls, field, x, y, wavelength, n=1.0, step=4, overlap=1.5, phase=None,
                   threshold=1e-4, z0=0.0):
        """Beamlets on a sub grid of a sampled field.

        Parameters
        ----------
        field : array (Ny, Nx) or analytic_phase.QuadraticPhaseField
            The field. For a QuadraticPhaseField, tilt and curvature of the
            analytic phase are taken exactly and only the smooth amplitude
            is differentiated (x, y and wavelength are then taken from it and
            the arguments ignored).
        x, y : arrays
            Linear coordinates.
        wavelength : number
            Vacuum wavelength.
        n : number, optional
            Refractive index.
        step : int, optional
            Beamlet spacing in samples.
        overlap : number, optional
            Beamlet radius w over the spacing (>= 1.5 for a flat frame).
        phase : array, optional
            Unwrapped phase of the field; default: unwrapped by integrating
            the wrapped phase differences.
        threshold : number, optional
            Drop beamlets with ``|u|**2`` below threshold times the maximum.
        z0 : number, optional
            z of the field.
        """

        A, X, Y, gx, gy, hxx, hxy, hyy, ph, wl, n = _field_derivatives(field, x, y, wavelength, n,
                                                                         phase)
        sl = (slice(step//2, None, step),)*2
        I = np.abs(A)**2
        keep = I[sl] > threshold*I.max()
        dx, dy = X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0]
        area = step**2*dx*dy
        w = overlap*np.sqrt(area)
        pick = lambda a: a[sl][keep]
        H = np.stack([np.stack([pick(hxx), pick(hxy)], -1), np.stack([pick(hxy), pick(hyy)], -1)], -2)
        Gamma = H + 2j/w**2*np.eye(2)
        a = np.abs(pick(A))*area/(np.pi*w**2)
        return cls(np.column_stack([pick(X), pick(Y)]), np.column_stack([pick(gx), pick(gy)]),
                   pick(ph), a, Gamma, wl, n, z0)

    @classmethod
    def from_rays(cls, points, directions, opl, wavelength, launch=None, irradiance=1.0, n=1.0,
                  overlap=1.5):
        """One beamlet per ray of a structured ray grid in a plane.

        The wavefront curvature at every ray comes from the directions of its
        neighbours (H = dk_perp/dx, finite differences on the grid), the
        beamlet radius from the local ray spacing, w = overlap*sqrt(area),
        the amplitude from the irradiance, carried from the launch plane like
        in rays_to_field().

        Parameters
        ----------
        points : array, shape (Nv, Nu, 3)
            Ray intersections with the launch plane of the beamlets (all with
            the same z).
        directions : array, shape (Nv, Nu, 3)
            Unit directions.
        opl : array, shape (Nv, Nu)
            Optical paths.
        wavelength : number
            Vacuum wavelength.
        launch, irradiance : optional
            As in rays_to_field(): positions and irradiance of the rays
            where they were launched (default: the irradiance in the beamlet
            plane is the power per grid cell over the cell area, i.e. the
            given irradiance is the power per cell).
        n : number, optional
            Refractive index of the medium.
        overlap : number, optional
            Beamlet radius over the ray spacing.
        """

        P = np.asarray(points, dtype=float)
        d = np.asarray(directions, dtype=float)
        S = np.asarray(opl, dtype=float)
        shape = P.shape[:2]
        k0 = 2*np.pi/wavelength
        K = n*k0*d[..., :2]
        # Jacobians with respect to the grid indices (u along axis 1, v along axis 0)
        xv, xu = _nan_gradient(P[..., 0])
        yv, yu = _nan_gradient(P[..., 1])
        kxv, kxu = _nan_gradient(K[..., 0])
        kyv, kyu = _nan_gradient(K[..., 1])
        J = np.stack([np.stack([xu, xv], -1), np.stack([yu, yv], -1)], -2)
        Kj = np.stack([np.stack([kxu, kxv], -1), np.stack([kyu, kyv], -1)], -2)
        H = Kj @ _inv2(J)
        H = 0.5*(H + np.swapaxes(H, -1, -2))
        area = np.abs(J[..., 0, 0]*J[..., 1, 1] - J[..., 0, 1]*J[..., 1, 0])
        if launch is None:
            area_launch = np.ones(shape)
        else:
            L = np.asarray(launch, dtype=float)
            lv, lu = _nan_gradient(L[..., 0])
            mv, mu = _nan_gradient(L[..., 1])
            area_launch = np.abs(lu*mv - lv*mu)
        I = np.broadcast_to(np.asarray(irradiance, dtype=float), shape)*area_launch/area
        ok = (np.isfinite(P).all(-1) & np.isfinite(d).all(-1) & np.isfinite(S) & np.isfinite(I)
              & np.isfinite(H).all((-1, -2)) & (I > 0))
        w = overlap*np.sqrt(area[ok])
        Gamma = H[ok] + 2j/w[:, None, None]**2*np.eye(2)
        a = np.sqrt(I[ok])*area[ok]/(np.pi*w**2)
        z0 = np.nanmean(P[..., 2][ok])
        return cls(P[ok][:, :2], K[ok], k0*S[ok], a, Gamma, wavelength, n, z0)

    def propagated(self, z):
        """Centres, widths and curvatures in the plane z (see field()).

        Returns
        -------
        centres : array (N, 2)
            Where the beamlet axes cross the plane.
        Gamma_z : array (N, 2, 2)
            Complex curvature matrices there.
        factor : array (N,)
            Complex amplitude factor a_j/sqrt(det(I - (z - z0) Gamma G))
            times the phase exp(i*(phase_j + (z - z0)*k**2/k_z)) on the axis.
        """

        dz = z - self.z0
        k = self.k
        kx, ky = self.k_perp[:, 0], self.k_perp[:, 1]
        kz = np.sqrt(k**2 - kx**2 - ky**2)
        # Hessian of k_z(k_x, k_y)
        G = np.empty((len(self), 2, 2))
        G[:, 0, 0] = -1/kz - kx**2/kz**3
        G[:, 1, 1] = -1/kz - ky**2/kz**3
        G[:, 0, 1] = G[:, 1, 0] = -kx*ky/kz**3
        GG = self.Gamma @ G
        tr = GG[:, 0, 0] + GG[:, 1, 1]
        det = GG[:, 0, 0]*GG[:, 1, 1] - GG[:, 0, 1]*GG[:, 1, 0]
        disc = np.sqrt(tr**2/4 - det)
        nu1, nu2 = tr/2 + disc, tr/2 - disc
        # continuous branch: the eigenvalues have Im < 0, 1 - z*nu never crosses the cut
        root = np.sqrt(1 - dz*nu1)*np.sqrt(1 - dz*nu2)
        Gz = self.Gamma @ _inv2(np.eye(2) - dz*np.swapaxes(GG, -1, -2))
        Gz = 0.5*(Gz + np.swapaxes(Gz, -1, -2))
        centres = self.centres + dz*self.k_perp/kz[:, None]
        factor = self.amplitude/root*np.exp(1j*(self.phase + dz*k**2/kz))
        return centres, Gz, factor

    def field(self, x, y, z=None, max_pairs=20_000_000):
        """The sum of the beamlets at the points (x, y) of the plane z.

        Parameters
        ----------
        x, y : arrays
            Coordinates, broadcast against each other (e.g. a meshgrid, or a
            line).
        z : number, optional
            Plane (default: the launch plane z0).
        max_pairs : int, optional
            Beamlet-point pairs evaluated at once (memory).

        Returns
        -------
        u : complex array of the broadcast shape of x and y
        """

        z = self.z0 if z is None else z
        X, Y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        shape = X.shape
        X, Y = X.ravel(), Y.ravel()
        centres, Gz, factor = self.propagated(z)
        out = np.zeros(len(X), dtype=complex)
        chunk = max(1, int(max_pairs//max(len(X), 1)))
        for j in range(0, len(self), chunk):
            s = slice(j, j + chunk)
            ex = X[None, :] - centres[s, 0, None]
            ey = Y[None, :] - centres[s, 1, None]
            g = Gz[s]
            q = g[:, 0, 0, None]*ex**2 + 2*g[:, 0, 1, None]*ex*ey + g[:, 1, 1, None]*ey**2
            lin = self.k_perp[s, 0, None]*ex + self.k_perp[s, 1, None]*ey
            out += np.einsum("j,jm->m", factor[s], np.exp(1j*(lin + 0.5*q)))
        return out.reshape(shape)

    def trace(self, system, z=None, clip=True, delta=1e-3, details=False):
        """Carry the beamlets through the surfaces of a raytrace.System.

        Every beamlet is followed by its base ray and eight parabasal rays
        (its centre shifted by +-delta*w in x and y, its transverse wave
        vector by +-delta*2/w in k_x and k_y), all traced exactly through the
        system. In the output plane, central differences of the parabasal
        rays give the 4 x 4 ray transfer matrix M = [[A, B], [C, D]] of the
        neighbourhood of the base ray, in the canonical coordinates
        (x, y, k_x, k_y) of the launch plane and of the output plane, and the
        beamlet follows the complex ABCD law (Arnaud 1985) ::

            Gamma' = (C + D Gamma)(A + B Gamma)^-1,
            a' = a/sqrt(det(A + B Gamma)),  phase' = k_0*OPL of the base ray.

        The square root is taken on the branch that follows the beamlet
        through the system (its Gouy phase): the argument of det(A + B Gamma)
        is accumulated in planes normal to the base ray, in closed form along
        every straight segment and by the (small) change of the projection at
        every surface. A beamlet is dropped if one of its rays is lost
        (vignetting with clip=True, total internal reflection, missed
        surfaces). Mirrors fold the light out of +z; the output plane must lie
        in a medium where the light travels in +z.

        Parameters
        ----------
        system : raytrace.System
            The system, with the same wavelength; the beamlets lie in its
            object space (index system.n[0]), in front of the first surface.
        z : number, optional
            Output plane (global z) behind the last surface; default: the
            image plane of the system.
        clip : bool, optional
            Drop beamlets that hit a semi diameter.
        delta : number, optional
            Size of the parabasal offsets relative to the beamlet.
        details : bool, optional
            Also return the transfer matrices and Gouy phases.

        Returns
        -------
        beamlets : GaussianBeamlets
            In the plane z, in the medium behind the last surface.
        M : array (N, 4, 4)
            Only with details=True: transfer matrices of the kept beamlets.
        gouy : array (N,)
            Only with details=True: -arg(det(A + B Gamma))/2, unwrapped (for
            a round beamlet through a focus: -pi/2 per line focus).
        keep : array (N,) of bool
            Only with details=True: which input beamlets were kept.
        """

        from .raytrace import Rays  # raytrace imports are heavy, keep them local

        if not np.isclose(system.wavelength, self.wavelength, rtol=1e-12, atol=0.0):
            raise ValueError("GaussianBeamlets.trace: the system has another wavelength")
        n_sys = system.n
        if not np.isclose(n_sys[0], self.n):
            raise ValueError("GaussianBeamlets.trace: the beamlets must lie in the object space "
                             "of the system (n = {})".format(n_sys[0]))
        N = len(self)
        k0 = 2*np.pi/self.wavelength
        k = self.k
        z = system.image_z if z is None else float(z)

        # base ray and eight parabasal rays (+-e_j in x, y, k_x, k_y):
        w = np.sqrt(2/np.real(np.trace(self.Gamma.imag, axis1=1, axis2=2)/2))
        eps = np.stack([delta*w, delta*w, delta*2/w, delta*2/w], -1)  # (N, 4)
        offsets = np.zeros((9, N, 4))
        for j in range(4):
            offsets[1 + 2*j, :, j] = eps[:, j]
            offsets[2 + 2*j, :, j] = -eps[:, j]
        state = np.concatenate([self.centres, self.k_perp], -1)[None] + offsets  # (9, N, 4)
        kz = np.sqrt(k**2 - state[..., 2]**2 - state[..., 3]**2)
        P0 = np.stack([state[..., 0], state[..., 1], np.full((9, N), self.z0)], -1)
        D0 = np.stack([state[..., 2], state[..., 3], kz], -1)/k
        opl0 = np.broadcast_to(self.phase/k0, (9, N))
        bundle = system.trace(Rays(P0.reshape(-1, 3), D0.reshape(-1, 3), opl0.reshape(-1)),
                              clip=clip, last=len(system.surfaces) - 1)
        S = len(system.surfaces)
        pts = bundle.points.reshape(S + 1, 9, N, 3)
        dirs = bundle.directions.reshape(S + 1, 9, N, 3)
        opls = bundle.opl.reshape(S + 1, 9, N)
        keep = np.all(np.isfinite(pts), axis=(0, 1, 3)) & np.all(np.isfinite(dirs), axis=(0, 1, 3))
        if np.any(keep & (dirs[-1, 0, :, 2] <= 0)):
            raise ValueError("GaussianBeamlets.trace: the light must travel in +z behind the "
                             "last surface")
        pts, dirs, opls = pts[:, :, keep], dirs[:, :, keep], opls[:, :, keep]
        eps, Gamma = eps[keep], self.Gamma[keep]
        n_out = abs(n_sys[-1])

        def matrix(Q, d, origin, normal, e1, e2, n):
            """Transfer matrix from the launch plane to the plane through origin
            with the given normal, coordinates along e1, e2; the rays are the
            lines Q + s*d."""
            s = np.sum((origin - Q)*normal, -1)/np.sum(d*normal, -1)
            X = Q + s[..., None]*d - origin
            u = np.stack([np.sum(X*e1, -1), np.sum(X*e2, -1),
                          n*k0*np.sum(d*e1, -1), n*k0*np.sum(d*e2, -1)], -1)  # (9, N, 4)
            M = np.empty(u.shape[1:2] + (4, 4))
            for j in range(4):
                M[:, :, j] = (u[1 + 2*j] - u[2 + 2*j])/(2*eps[:, j, None])
            return M

        def XP(M):
            X = M[:, :2, :2] + M[:, :2, 2:] @ Gamma
            P = M[:, 2:, :2] + M[:, 2:, 2:] @ Gamma
            return X, P

        def det2(X):
            return X[:, 0, 0]*X[:, 1, 1] - X[:, 0, 1]*X[:, 1, 0]

        def normal_frame(t):
            """Right handed (e1, e2, t) normal to the unit vectors t."""
            a = np.where(np.abs(t[:, 0:1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
            e1 = np.cross(a, t)
            e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
            return e1, np.cross(t, e1)

        ex, ey, ez = np.eye(3)
        # unwrapped arg det(A + B Gamma): launch plane (identity) -> normal planes along
        # the base ray -> output plane
        theta = np.zeros(len(eps))
        prev = np.ones(len(eps), dtype=complex)  # det X in the previous plane

        def step(detX, predicted=None):
            """Add the change of arg(det X) to theta; predicted: the analytic
            (continuous) increment, which fixes the multiple of 2*pi."""
            nonlocal prev, theta
            change = np.angle(detX/prev)
            if predicted is not None:
                change = change + 2*np.pi*np.round((predicted - change)/(2*np.pi))
            theta = theta + change
            prev = detX

        # S + 1 straight segments: object space, behind each surface; the
        # segment i starts at pts[i] with direction dirs[i] and ends at pts[i + 1]
        z_end = np.broadcast_to(np.array([0.0, 0.0, z]), pts[0].shape)
        for i in range(S + 1):
            Q, d, n = pts[i], dirs[i], abs(n_sys[i])
            t = d[0]
            e1, e2 = normal_frame(t)
            start = Q[0]
            M = matrix(Q, d, start, t, e1, e2, n)
            X, P = XP(M)
            step(det2(X))  # change of the plane at a surface (or launch): no focus
            end = pts[i + 1][0] if i < S else start + (z - start[:, 2:3])/t[:, 2:3]*t
            L = np.sum((end - start)*t, -1)
            nu = np.linalg.eigvals(P @ _inv2(X))  # Im > 0
            predicted = np.sum(np.angle(1 + L[:, None]*nu/(n*k0)), -1)
            M = matrix(Q, d, end, t, e1, e2, n)
            X, P = XP(M)
            step(det2(X), predicted)
        # output plane z
        M = matrix(pts[-1], dirs[-1], z_end, ez, ex, ey, n_out)
        X, P = XP(M)
        step(det2(X))
        Gamma_out = P @ _inv2(X)
        Gamma_out = 0.5*(Gamma_out + np.swapaxes(Gamma_out, -1, -2))
        detX = det2(X)
        factor = np.exp(-0.5j*theta)/np.sqrt(np.abs(detX))

        base_d = dirs[-1, 0]
        s = (z - pts[-1, 0, :, 2])/base_d[:, 2]
        centre = pts[-1, 0, :, :2] + s[:, None]*base_d[:, :2]
        opl = opls[-1, 0] + n_out*s
        out = GaussianBeamlets(centre, n_out*k0*base_d[:, :2], k0*opl, self.amplitude[keep]*factor,
                               Gamma_out, self.wavelength, n_out, z)
        if details:
            return out, M, -0.5*theta, keep
        return out


# field to rays:

def _wrapped_derivatives(u, dx, dy):
    """Phase gradient from the wrapped phase differences of neighbours (the
    mean of the forward and backward differences: valid up to pi per
    sample)."""
    def along(u, d):
        step = np.angle(u[1:]*np.conj(u[:-1]))/d
        g = np.empty(u.shape)
        g[1:-1] = 0.5*(step[1:] + step[:-1])
        g[0], g[-1] = step[0], step[-1]
        return g
    return along(u.T, dx).T, along(u, dy)


def _unwrap(u, I, threshold=1e-6):
    """Phase unwrapped along rows from the column of the brightest sample, the
    rows aligned along that column (adequate for smooth, single valued
    fields)."""
    ph = np.angle(u)
    r, c = np.unravel_index(np.argmax(I), I.shape)
    col = np.unwrap(ph[:, c])
    rows = np.unwrap(ph, axis=1)
    return rows - rows[:, c:c + 1] + col[:, None]


def _field_derivatives(field, x, y, wavelength, n, phase):
    """Amplitude, coordinates, phase gradient, Hessian and unwrapped phase of a
    sampled field or a QuadraticPhaseField."""
    from .analytic_phase import QuadraticPhaseField
    if isinstance(field, QuadraticPhaseField):
        if field.domain != "space":
            raise ValueError("a spectrum cannot be converted to rays")
        x, y, wavelength, n = field.x, field.y, field.wavelength, field.n
        A = field.amplitude
        X, Y = np.meshgrid(x, y)
        dx, dy = x[1] - x[0], y[1] - y[0]
        I = np.abs(A)**2
        res = _unwrap(A, I) if phase is None else np.asarray(phase) - field.phase()
        rgy, rgx = np.gradient(res, dy, dx)
        hyy, hyx = np.gradient(rgy, dy, dx)
        hxy, hxx = np.gradient(rgx, dy, dx)
        ax, ay = field.alpha
        bx, by = field.beta
        gx = rgx + 2*ax*X + bx
        gy = rgy + 2*ay*Y + by
        return (A, X, Y, gx, gy, hxx + 2*ax, 0.5*(hxy + hyx), hyy + 2*ay, res + field.phase(),
                wavelength, n)
    x, y = _grid(x, "x"), _grid(y, "y")
    u = np.asarray(field, dtype=complex)
    X, Y = np.meshgrid(x, y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    I = np.abs(u)**2
    ph = _unwrap(u, I) if phase is None else np.asarray(phase, dtype=float)
    if phase is None:
        gx, gy = _wrapped_derivatives(u, dx, dy)
    else:
        gy, gx = np.gradient(ph, dy, dx)
    hyy, hyx = np.gradient(gy, dy, dx)
    hxy, hxx = np.gradient(gx, dy, dx)
    return u, X, Y, gx, gy, hxx, 0.5*(hxy + hyx), hyy, ph, wavelength, n


def _rays_from(X, Y, kx, ky, power, opl, k, z):
    kz2 = k**2 - kx**2 - ky**2
    ok = kz2 > 0
    kz = np.sqrt(np.where(ok, kz2, 0.0))
    D = np.stack([kx, ky, kz], -1)/k
    P = np.stack([X, Y, np.full_like(X, float(z))], -1)
    return FieldRays(P[ok], D[ok], opl[ok], power[ok])


def field_to_rays(field, x=None, y=None, wavelength=None, n=1.0, z=0.0, step=1, threshold=1e-3,
                  phase=None, structured=False):
    """Rays along the local phase gradient (energy flux) of a field.

    Every sample (of a sub grid with the given step) with ``|u|**2`` above
    threshold times the maximum (and so are its four neighbours) becomes a ray at (x, y, z) with the
    direction (k_x, k_y, k_z)/(n*k_0), k_perp = grad(phase), the power
    ``|u|**2`` times the area per ray and the optical path phase/k_0.

    Parameters
    ----------
    field : array (Ny, Nx) or analytic_phase.QuadraticPhaseField
        Field. For a QuadraticPhaseField the analytic phase is
        differentiated exactly (x, y, wavelength and n are taken from it).
    x, y : arrays
        Linear coordinates.
    wavelength : number
        Vacuum wavelength.
    n : number, optional
        Refractive index.
    z : number, optional
        z of the plane (for the start points).
    step : int, optional
        Take every step-th sample along x and y.
    threshold : number, optional
        Relative intensity below which no rays are made.
    phase : array, optional
        Unwrapped phase; default: unwrapped from the field (for the optical
        path; the directions never need unwrapping).
    structured : bool, optional
        If True, keep the grid: the arrays have the shape (Ny', Nx', ...)
        of the (sub sampled) grid, with NaN for the dropped samples, so that
        the rays (also after System.trace() and a reshape) can be turned
        into a field again with rays_to_field() (launch: the start points,
        irradiance: ``|u|**2``).

    Returns
    -------
    FieldRays
        Can be traced with raytrace.System.trace(). Samples whose gradient
        exceeds n*k_0 (evanescent) are dropped.
    """

    A, X, Y, gx, gy, _, _, _, ph, wl, n = _field_derivatives(field, x, y, wavelength, n, phase)
    k = wavenumber(wl, n)
    dx, dy = X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0]
    sl = (slice(step//2, None, step),)*2
    I = np.abs(A)**2
    bright = I > threshold*I.max()
    # the phase differences need bright neighbours (no rays from the dark edge samples)
    lit = bright.copy()
    lit[1:] &= bright[:-1]
    lit[:-1] &= bright[1:]
    lit[:, 1:] &= bright[:, :-1]
    lit[:, :-1] &= bright[:, 1:]
    keep = lit[sl]
    power = I[sl]*dx*dy*step**2
    if structured:
        kx, ky = gx[sl], gy[sl]
        kz2 = k**2 - kx**2 - ky**2
        keep = keep & (kz2 > 0)
        D = np.stack([kx, ky, np.sqrt(np.where(keep, kz2, 0.0))], -1)/k
        P = np.stack([X[sl], Y[sl], np.full(X[sl].shape, float(z))], -1)
        P[~keep], D[~keep] = np.nan, np.nan
        return FieldRays(P, D, np.where(keep, ph[sl]*n/k, np.nan), np.where(keep, power, 0.0))
    rays = _rays_from(X[sl][keep], Y[sl][keep], gx[sl][keep], gy[sl][keep], power[keep],
                      ph[sl][keep]*n/k, k, z)
    return rays


def local_spectrum_rays(field, x, y, wavelength, n=1.0, z=0.0, window=32, step=None,
                        threshold=1e-2, pad=2, jitter=None):
    """Rays from the local spectra of a field (sampled spectrogram / Husimi
    function).

    The field is cut into Gaussian windowed patches (1/e**2 intensity radius
    window/4 samples) centred every step samples. Every component of the
    (zero padded) patch spectrum with a power above threshold times the
    largest one becomes a ray at the patch centre with the direction of its
    wave vector and the power of that component, scaled such that all rays
    together carry the power of the field. Several ray directions per point:
    crossing sheets of a caustic are resolved. The rays carry no phase (opl
    NaN).

    Parameters
    ----------
    field : array (Ny, Nx)
        Sampled field.
    x, y : arrays
        Linear coordinates.
    wavelength : number
        Vacuum wavelength.
    n : number, optional
        Refractive index.
    z : number, optional
        z of the plane.
    window : int, optional
        Patch size in samples (even).
    step : int, optional
        Distance of the patch centres in samples (default window//4).
    threshold : number, optional
        Relative power below which spectral components are dropped.
    pad : int, optional
        Zero padding factor of the patch FFT (finer direction sampling).
    jitter : numpy.random.Generator or int, optional
        If given (a generator or a seed), every ray is moved to a random
        point of its phase space cell (uniformly within the step in x and y
        and the spectral bin in k_x and k_y). Without, the rays sit on a
        lattice of positions and directions, which shows as moire patterns
        when the rays are binned in other planes.

    Returns
    -------
    FieldRays
        The power is normalized such that its sum equals ``sum(|u|**2)*dx*dy``.
    """

    x, y = _grid(x, "x"), _grid(y, "y")
    u = np.asarray(field, dtype=complex)
    step = window//4 if step is None else step
    dx, dy = x[1] - x[0], y[1] - y[0]
    s = np.arange(window) - window/2 + 0.5
    g = np.exp(-(s/(window/4))**2)  # amplitude window
    G = g[:, None]*g[None, :]
    cy = np.arange(window//2, len(y) - window//2 + 1, step)
    cx = np.arange(window//2, len(x) - window//2 + 1, step)
    Np = pad*window
    kx = 2*np.pi*np.fft.fftfreq(Np, dx)
    ky = 2*np.pi*np.fft.fftfreq(Np, dy)
    KX, KY = np.meshgrid(kx, ky)
    k = wavenumber(wavelength, n)
    pts, dirs, pw = [], [], []
    spectra = []
    for j in cy:
        rows = u[j - window//2:j + window//2]
        patches = np.stack([rows[:, i - window//2:i + window//2] for i in cx])*G
        spectra.append(np.abs(np.fft.fft2(patches, (Np, Np)))**2)
    spectra = np.array(spectra)  # (Py, Px, Np, Np)
    smax = spectra.max()
    for a, j in enumerate(cy):
        for b, i in enumerate(cx):
            Sp = spectra[a, b]
            m = (Sp > threshold*smax) & (KX**2 + KY**2 < k**2)
            if not np.any(m):
                continue
            kxs, kys = KX[m], KY[m]
            kz = np.sqrt(k**2 - kxs**2 - kys**2)
            dirs.append(np.column_stack([kxs, kys, kz])/k)
            # patch centre: between the samples i - 1 and i (even window)
            pts.append(np.tile([x[i] - 0.5*dx, y[j] - 0.5*dy, z], (len(kxs), 1)))
            pw.append(Sp[m])
    if not pts:
        return FieldRays(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0))
    P = np.concatenate(pts)
    D = np.concatenate(dirs)
    w = np.concatenate(pw)
    if jitter is not None:
        rng = np.random.default_rng(jitter)
        P[:, 0] += (rng.random(len(P)) - 0.5)*step*dx
        P[:, 1] += (rng.random(len(P)) - 0.5)*step*dy
        kxy = D[:, :2]*k + (rng.random((len(P), 2)) - 0.5)*[kx[1] - kx[0], ky[1] - ky[0]]
        kz2 = k**2 - np.sum(kxy**2, 1)
        ok = kz2 > 0
        D = np.column_stack([kxy, np.sqrt(np.maximum(kz2, 0))])/k
        P, D, w = P[ok], D[ok], w[ok]
    w = w*np.sum(np.abs(u)**2)*dx*dy/w.sum()
    return FieldRays(P, D, np.full(len(w), np.nan), w)


def wigner(u, x, upsample=2):
    """Wigner distribution of a 1D field (e.g. a section of a 2D field).

    W(x, k) = int u(x + s/2) conj(u(x - s/2)) exp(-i*k*s) ds, evaluated on the
    sample positions x and the wavenumbers k (the field is interpolated by
    Fourier zero padding first, by the factor upsample, so that the
    wavenumbers cover the full band of the input sampling without aliasing).

    Its marginals are the intensity, ``int W dk/(2*pi) = |u(x)|**2``, and the
    power spectrum; it is real and can be negative (interference between ray
    sheets).

    Parameters
    ----------
    u : array (N,)
        Field.
    x : array (N,)
        Linear coordinates.
    upsample : int, optional
        Interpolation factor (>= 2).

    Returns
    -------
    k : array (N*upsample,)
        Wavenumbers, ascending, from -pi/dx to pi/dx.
    W : array (len(k), N)
        Wigner distribution, W[i, j] at (x[j], k[i]).
    """

    x = _grid(x, "x")
    u = np.asarray(u, dtype=complex)
    N = len(u)
    dx = x[1] - x[0]
    # Fourier interpolation onto the spacing dx/upsample
    M = N*upsample
    U = np.fft.fftshift(np.fft.fft(u))
    Up = np.zeros(M, dtype=complex)
    Up[(M - N)//2:(M - N)//2 + N] = U
    v = np.fft.ifft(np.fft.ifftshift(Up))*upsample
    h = dx/upsample
    # lags s = 2*m*h, m = -L .. L - 1 around every original sample
    L = M//2
    m = np.arange(-L, L)
    centre = np.arange(N)*upsample
    ip = centre[:, None] + m[None, :]
    im = centre[:, None] - m[None, :]
    valid = (ip >= 0) & (ip < M) & (im >= 0) & (im < M)
    r = np.where(valid, v[np.clip(ip, 0, M - 1)]*np.conj(v[np.clip(im, 0, M - 1)]), 0)
    # W(k) = sum_m r(m) exp(-i k 2 m h) 2h; FFT over m
    R = np.fft.fft(np.fft.ifftshift(r, axes=1), axis=1)*2*h
    k = 2*np.pi*np.fft.fftfreq(2*L, 2*h)
    order = np.argsort(k)
    return k[order], np.real(R[:, order]).T
