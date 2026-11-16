#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Ray dynamics in 2D billiards: the bounce map, Poincaré surfaces of
section and the open dielectric billiard of a microcavity.

A ray inside a closed curve runs straight and is reflected specularly at the
boundary. Between two bounces it is fully described by the point of the
bounce and the direction of the ray, in the **Birkhoff coordinates** ::

    s = arc length along the boundary (counter-clockwise from the point at
        polar angle 0), 0 <= s < L,
    p = sin(chi) = tangential component of the unit direction, -1 <= p <= 1,

chi the angle of incidence from the normal, positive for rays that travel
counter-clockwise. The bounce map (s, p) -> (s', p') preserves the area
ds dp (it is the Poincaré map of the billiard flow), so a uniform ray
ensemble stays uniform: Liouville's theorem of notebook 83 (étendue) on the surface of
section. The boundary is the same `PolarShape` (`fdtd.Limacon`,
`fdtd.Stadium`, a circle) that the FDTD and FDFD solvers draw, so the ray
picture and the modes of `pyoptics.fdfd` belong to the same cavity.

- In the circle every ray keeps its p (angular momentum conservation): the
  surface of section is a stack of horizontal lines, the billiard is
  integrable.
- The limaçon of Pascal (Robnik 1983) has a mixed phase space: regular
  islands (stable periodic orbits and their KAM tori) in a chaotic sea; for
  small deformations whispering gallery rays near p = +-1 stay regular.
- The Bunimovich stadium is fully chaotic: one orbit fills the whole
  surface of section (apart from the marginally stable bouncing ball
  orbits between the straight sides).

**Open billiard.** In a dielectric cavity of index n in air a ray is only
partly reflected: with the Fresnel reflectance R(chi) (polarization s for
E along z, the Ez modes of `pyoptics.fdfd`, which the microcavity
literature calls TM), total internal reflection above the critical lines
abs(p) = 1/n. Each ray carries an intensity that is multiplied by R at every
bounce; the part (1 - R) leaves by refraction (Snell, n p = sin(chi_out)).
After a transient, the intensity that remains in the cavity is distributed
over the surface of section in a fixed pattern, the **steady probability
distribution**, which decays as a whole (Lee et al. 2004; Shinohara and
Harayama 2007). In a chaotic cavity it follows the unstable manifold of
the chaotic saddle and decides where and in which directions the light
leaves; its long-lived resonances have Husimi functions
(`pyoptics.fdfd.husimi_birkhoff()`) concentrated on the same structure.

Functions and classes
---------------------
- `Billiard(shape)`: the boundary (arc length, tangents, normals), the
  bounce map `bounce()`, `orbit()` for many starting points at once and
  `surface_of_section()` for a Poincaré plot.
- `steady_distribution()`, `escape_directions()`: the open, dielectric
  billiard with Fresnel weights; the surface of section and the far field
  of the leaking rays.
- `boundary_husimi()`: the Husimi functions of a wave in Birkhoff
  coordinates from its values and normal derivatives on the boundary
  (Hentschel, Schomerus and Schubert 2003), the wave counterpart of the
  surface of section; `pyoptics.fdfd.husimi_birkhoff()` takes them from an
  FDFD mode.
- `circle_wgm_p()`: the p of the whispering gallery mode m of a disk,
  m/(n k R).

Conventions as in `pyoptics.fdtd`: 2D, lengths in any unit, the boundary
counter-clockwise, normals outward.

References
----------
G. D. Birkhoff, "Dynamical Systems", AMS (1927), chapter VI.

M. Robnik, "Classical dynamics of a family of billiards with analytic
boundaries", J. Phys. A 16, 3971-3986 (1983).

L. A. Bunimovich, "On the ergodic properties of nowhere dispersing
billiards", Commun. Math. Phys. 65, 295-312 (1979).

S.-Y. Lee, S. Rim, J.-W. Ryu, T.-Y. Kwon, M. Choi and C.-M. Kim,
"Quasiscarred resonances in a spiral-shaped microcavity", Phys. Rev. Lett.
93, 164102 (2004).

S. Shinohara and T. Harayama, "Signature of ray chaos in quasi-bound wave
functions for a stadium-shaped dielectric cavity", Phys. Rev. E 75, 036216
(2007).

J. Wiersig and M. Hentschel, "Combining directional light output and
ultralow loss in deformed microdisks", Phys. Rev. Lett. 100, 033901 (2008).

M. Hentschel, H. Schomerus and R. Schubert, "Husimi functions at
dielectric interfaces: Inside-outside duality for optical systems and
beyond", Europhys. Lett. 62, 636-642 (2003).

H. Cao and J. Wiersig, "Dielectric microcavities: Model systems for wave
chaos and non-Hermitian physics", Rev. Mod. Phys. 87, 61-111 (2015).
"""

import numpy as np
from scipy.interpolate import CubicSpline

from .interfaces import R_s, R_p


__all__ = ["Billiard", "steady_distribution", "escape_directions", "fresnel_reflectance", "circle_wgm_p",
           "boundary_husimi"]


def circle_wgm_p(m, n, k, R):
    """p = sin(chi) of the rays of the whispering gallery mode with the
    azimuthal number m of a disk of radius R and index n at the vacuum
    wavenumber k: m/(n k R) (the angular momentum of the mode, which the
    rays of the circle conserve)."""

    return m/(n*k*R)


def fresnel_reflectance(p, n, polarization="s"):
    """Reflectance of a ray inside a medium of index n (in air) hitting the
    boundary with p = sin(chi); 1 for total internal reflection
    (abs(p) >= 1/n). polarization "s" (E perpendicular to the plane of
    incidence: E along z in 2D) or "p"."""

    chi = np.arcsin(np.clip(np.abs(p), 0, 1))
    R = {"s": R_s, "p": R_p}[polarization](chi, n, 1.0)
    return np.where(np.abs(p)*n >= 1, 1.0, R)


class Billiard(object):
    """Billiard inside a closed, convex, star-shaped curve.

    Parameters
    ----------
    shape : fdtd.PolarShape
        The boundary r(phi) about shape.center (`fdtd.Limacon`,
        `fdtd.Stadium`, ...). It has to be convex: the bounce map finds the
        one other intersection of a chord with the curve.
    n_table : int, optional
        Points of the table of the arc length s(phi).

    Attributes
    ----------
    length : float
        Perimeter L.
    """

    def __init__(self, shape, n_table=16384):
        self.shape = shape
        self.center = np.asarray(shape.center, dtype=float)
        self._r = shape.radius
        phi = 2*np.pi*np.arange(n_table)/n_table
        speed = np.hypot(self._r(phi), self._dr(phi))
        # spectral antiderivative of the periodic speed |dX/dphi|
        F = np.fft.fft(speed)
        kk = np.fft.fftfreq(n_table, 1.0/n_table)
        G = np.zeros_like(F)
        G[kk != 0] = F[kk != 0]/(1j*kk[kk != 0])
        periodic = np.real(np.fft.ifft(G))
        self.length = float(F[0].real/n_table*2*np.pi)
        periodic = periodic - periodic[0]
        phi_c = np.append(phi, 2*np.pi)
        self._s_per = CubicSpline(phi_c, np.append(periodic, periodic[0]), bc_type="periodic")

    # geometry ------------------------------------------------------------

    def _dr(self, phi, h=1e-3):
        dr = getattr(self.shape, "dradius", None)
        if dr is not None:
            return dr(phi)
        f = self._r
        return (8*(f(phi + h) - f(phi - h)) - (f(phi + 2*h) - f(phi - 2*h)))/(12*h)

    def point(self, phi):
        """Boundary points (..., 2) at the polar angles phi."""

        r = self._r(phi)
        return np.stack([self.center[0] + r*np.cos(phi), self.center[1] + r*np.sin(phi)], -1)

    def _dX(self, phi):
        r, dr = self._r(phi), self._dr(phi)
        c, s = np.cos(phi), np.sin(phi)
        return np.stack([dr*c - r*s, dr*s + r*c], -1)

    def tangent(self, phi):
        """Unit tangents (..., 2), counter-clockwise."""

        d = self._dX(phi)
        return d/np.linalg.norm(d, axis=-1, keepdims=True)

    def normal(self, phi):
        """Unit outward normals (..., 2)."""

        t = self.tangent(phi)
        return np.stack([t[..., 1], -t[..., 0]], -1)

    def arc_length(self, phi):
        """Arc length s in [0, L) of the boundary point at the polar angle phi."""

        phi = np.mod(phi, 2*np.pi)
        return np.mod(self.length*phi/(2*np.pi) + self._s_per(phi), self.length)

    def polar_angle(self, s):
        """Polar angle phi in [0, 2 pi) of the boundary point at the arc length s."""

        s = np.mod(np.asarray(s, dtype=float), self.length)
        phi = 2*np.pi*s/self.length
        for _ in range(6):  # Newton on s(phi) = s from the uniform guess
            ds = self.length*phi/(2*np.pi) + self._s_per(np.mod(phi, 2*np.pi)) - s
            phi = phi - ds/np.hypot(self._r(phi), self._dr(phi))
        return np.mod(phi, 2*np.pi)

    def boundary(self, num=1024):
        """Equidistant points along the boundary.

        Returns
        -------
        s : array (num,)
            Arc lengths k L/num.
        points, tangents, normals : arrays (num, 2)
        """

        s = self.length*np.arange(num)/num
        phi = self.polar_angle(s)
        return s, self.point(phi), self.tangent(phi), self.normal(phi)

    # dynamics -------------------------------------------------------------

    def _direction(self, phi, p):
        t, n = self.tangent(phi), self.normal(phi)
        c = np.sqrt(np.maximum(1 - p**2, 0))[..., None]
        return p[..., None]*t - c*n

    def _hit(self, phi0, d, iterations=34):
        """Polar angle of the next boundary point along the direction d
        (pointing inward) from the boundary point at phi0."""

        P = self.point(phi0)

        def g(phi):
            X = self.point(phi) - P
            return d[..., 0]*X[..., 1] - d[..., 1]*X[..., 0]

        # g < 0 just after phi0 and > 0 just before phi0 + 2 pi (convex curve)
        lo, hi = np.zeros_like(phi0), np.full_like(phi0, 2*np.pi)
        for _ in range(iterations):
            mid = 0.5*(lo + hi)
            neg = g(phi0 + mid) < 0
            lo = np.where(neg, mid, lo)
            hi = np.where(neg, hi, mid)
        phi = phi0 + 0.5*(lo + hi)
        for _ in range(2):  # Newton polish
            dX = self._dX(phi)
            dg = d[..., 0]*dX[..., 1] - d[..., 1]*dX[..., 0]
            phi = phi - g(phi)/np.where(dg != 0, dg, 1.0)
        return np.mod(phi, 2*np.pi)

    def _bounce_phi(self, phi, p):
        d = self._direction(phi, p)
        phi1 = self._hit(phi, d)
        p1 = np.sum(d*self.tangent(phi1), -1)
        return phi1, np.clip(p1, -1, 1)

    def bounce(self, s, p):
        """The bounce map: Birkhoff coordinates of the next reflection.

        Parameters
        ----------
        s, p : arrays
            Arc length and sin(chi) of the current reflection.

        Returns
        -------
        s, p : arrays
            Those of the next reflection.
        """

        s, p = np.broadcast_arrays(np.asarray(s, float), np.asarray(p, float))
        phi1, p1 = self._bounce_phi(self.polar_angle(s), p)
        return self.arc_length(phi1), p1

    def orbit(self, s0, p0, n_bounces):
        """Iterate the bounce map for many rays at once.

        Parameters
        ----------
        s0, p0 : arrays (N,)
            Starting points in Birkhoff coordinates.
        n_bounces : int
            Number of bounces.

        Returns
        -------
        S, P : arrays (n_bounces + 1, N)
            Birkhoff coordinates of the start and every reflection.
        """

        s0, p0 = np.broadcast_arrays(np.atleast_1d(np.asarray(s0, float)), np.atleast_1d(np.asarray(p0, float)))
        phi, p = self.polar_angle(s0), p0.copy()
        S = np.empty((n_bounces + 1,) + s0.shape)
        P = np.empty_like(S)
        S[0], P[0] = np.mod(s0, self.length), p0
        for j in range(n_bounces):
            phi, p = self._bounce_phi(phi, p)
            S[j + 1], P[j + 1] = self.arc_length(phi), p
        return S, P

    def surface_of_section(self, n_rays=60, n_bounces=400, seed=0, s0=None, p0=None):
        """Poincaré surface of section: orbits of rays started at random
        (or given) points.

        Parameters
        ----------
        n_rays : int, optional
            Number of orbits (if s0, p0 are not given).
        n_bounces : int, optional
            Bounces per orbit.
        seed : int, optional
            Of the random starting points.
        s0, p0 : arrays, optional
            Starting points.

        Returns
        -------
        S, P : arrays (n_bounces + 1, n_rays)
            One column per orbit.
        """

        if s0 is None:
            rng = np.random.default_rng(seed)
            s0 = rng.uniform(0, self.length, n_rays)
            p0 = rng.uniform(-1, 1, n_rays)
        return self.orbit(s0, p0, n_bounces)

    def escape_angle(self, s, p, n):
        """Far-field direction (polar angle of the refracted ray, radians)
        of a ray that leaves the cavity of index n at (s, p); NaN for total
        internal reflection."""

        phi = self.polar_angle(s)
        t, nrm = self.tangent(phi), self.normal(phi)
        st = n*np.asarray(p, float)
        ct = np.sqrt(np.where(np.abs(st) < 1, 1 - st**2, np.nan))
        d = st[..., None]*t + ct[..., None]*nrm
        return np.arctan2(d[..., 1], d[..., 0])


def _weights(P, n, polarization):
    """Intensity of every ray when it arrives at each bounce (before the
    reflection there) and the reflectance at that bounce."""

    R = fresnel_reflectance(P, n, polarization)
    w = np.ones_like(P)
    w[1:] = np.cumprod(R[:-1], axis=0)
    return w, R


def steady_distribution(S, P, n, length, bins=(200, 100), skip=None, polarization="s"):
    """Steady probability distribution of an open dielectric billiard.

    The rays of an orbit ensemble (from `Billiard.orbit()`, started
    uniformly in phase space) lose the fraction 1 - R(p) of their intensity
    at every bounce. After skip bounces the intensity left in the cavity
    has settled into a fixed distribution over the surface of section,
    which decays as a whole. This histogram of the intensity arriving at
    the boundary, each bounce normalized by the intensity still in the
    cavity, averaged over the bounces after skip.

    Parameters
    ----------
    S, P : arrays (n_bounces + 1, N)
        Orbits.
    n : number
        Refractive index of the cavity (in air).
    length : number
        Perimeter of the billiard (range of s).
    bins : (int, int), optional
        Bins in s and p.
    skip : int, optional
        Transient bounces left out; default a third of them.
    polarization : "s" or "p", optional

    Returns
    -------
    s_edges, p_edges : arrays
    D : array (bins[1], bins[0])
        Distribution, D[i, j] at (s_j, p_i), mean 1.
    decay : array (n_bounces + 1,)
        Total intensity of the rays arriving at each bounce (1 at the
        first): its logarithmic slope is the ray-model loss per bounce.
    """

    skip = S.shape[0]//3 if skip is None else skip
    w, R = _weights(P, n, polarization)
    total = w.sum(1)
    wn = w[skip:]/total[skip:, None]
    s_edges = np.linspace(0, length, bins[0] + 1)
    p_edges = np.linspace(-1, 1, bins[1] + 1)
    D, _, _ = np.histogram2d(P[skip:].ravel(), S[skip:].ravel(), bins=(p_edges, s_edges), weights=wn.ravel())
    D = D/D.mean()
    decay = total/total[0]
    return s_edges, p_edges, D, decay


def escape_directions(billiard, S, P, n, bins=360, skip=None, polarization="s"):
    """Far-field pattern of the rays that leave an open dielectric billiard.

    At every bounce after skip the fraction 1 - R of the ray's intensity (in
    the normalization of `steady_distribution()`) leaves in the direction of
    the refracted ray.

    Parameters
    ----------
    billiard : Billiard
    S, P : arrays (n_bounces + 1, N)
        Orbits.
    n : number
        Refractive index of the cavity.
    bins : int, optional
        Angular bins over (-pi, pi].
    skip : int, optional
        Transient bounces left out; default a third of them.
    polarization : "s" or "p", optional

    Returns
    -------
    phi : array (bins,)
        Bin centres (radians, from +x).
    I : array (bins,)
        Intensity, mean 1 (as `fdfd.FarField.intensity`).
    """

    skip = S.shape[0]//3 if skip is None else skip
    w, R = _weights(P, n, polarization)
    total = w.sum(1)
    out = (w*(1 - R))[skip:]/total[skip:, None]
    leaving = np.abs(P[skip:])*n < 1
    ang = billiard.escape_angle(S[skip:][leaving], P[skip:][leaving], n)
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    I, _ = np.histogram(ang, bins=edges, weights=out[leaving])
    return 0.5*(edges[1:] + edges[:-1]), I/I.mean()


def boundary_husimi(s, psi, dpsi, length, n, k, q=None, p=None, sigma=None, cos_min=0.05):
    """Husimi functions of a 2D wave on the boundary of a billiard, in the
    Birkhoff coordinates (q, p) of the ray picture.

    Near a boundary point the wave in a medium of index n is a sum of plane
    waves; the one with the tangential wavenumber n k p and the normal one
    n k cos(chi) travels along the outward normal nu (towards the
    boundary on the inside, away from the cavity on the outside), its
    partner the opposite way. With psi and dpsi = d psi/d nu on the
    boundary, the two amplitudes are (psi +- dpsi/(i n k cos chi))/2.
    Projected on coherent states along the boundary,

        xi_(q,p)(s) = (pi sigma**2)**(-1/4) sum_l exp(-(s - q + l L)**2/(2 sigma**2) + i n k p (s - q + l L)),

    and weighted with the flux factor cos(chi), they give (Hentschel,
    Schomerus and Schubert 2003)

        H_out(q, p) = n k cos(chi)/(2 pi) abs(h + h'/(i n k cos chi))**2/4,
        H_in(q, p)  = n k cos(chi)/(2 pi) abs(h - h'/(i n k cos chi))**2/4,

    h, h' the projections of psi and dpsi. Inside the cavity H_out is the
    wave that hits the boundary (the "incident" Husimi function, comparable
    with the rays arriving at the surface of section) and H_in the reflected
    one; outside the cavity, H_out is the emerging wave.

    Parameters
    ----------
    s : array (N,)
        Equidistant arc lengths over [0, length) (`Billiard.boundary()`).
    psi, dpsi : arrays (N,)
        Wave and its derivative along the outward normal at s.
    length : number
        Perimeter L.
    n : number
        Index of the medium on the side where psi and dpsi are taken.
    k : number
        Vacuum wavenumber (real part, for a resonance).
    q : array, optional
        Positions along the boundary (default 256 points over [0, L)).
    p : array, optional
        sin(chi) (default 201 points over [-1, 1]).
    sigma : number, optional
        Width of the coherent states along s; default sqrt(L/(2 n k)),
        which gives the same relative resolution in q (sigma/L) and p
        (1/(n k sigma) out of 2).
    cos_min : number, optional
        Lower bound of cos(chi) in the denominator (grazing incidence).

    Returns
    -------
    q, p : arrays
    H_out, H_in : arrays (len(p), len(q))
        H[i, j] at (q[j], p[i]).
    """

    s = np.asarray(s, float)
    psi, dpsi = np.asarray(psi, complex), np.asarray(dpsi, complex)
    nk = n*k
    q = length*np.arange(256)/256 if q is None else np.asarray(q, float)
    p = np.linspace(-1, 1, 201) if p is None else np.asarray(p, float)
    sigma = np.sqrt(length/(2*nk)) if sigma is None else sigma
    ds = length/len(s)
    D = np.mod(s[None, :] - q[:, None] + length/2, length) - length/2  # (Nq, Ns), periodic
    G = (np.pi*sigma**2)**-0.25*np.exp(-D**2/(2*sigma**2))
    a, b = psi, dpsi
    h = np.empty((len(p), len(q)), complex)
    hd = np.empty_like(h)
    for i, pi_ in enumerate(p):
        E = G*np.exp(-1j*nk*pi_*D)*ds
        h[i] = E @ a
        hd[i] = E @ b
    c = np.sqrt(np.clip(1 - p**2, 0, 1))[:, None]
    cc = np.maximum(c, cos_min)
    pref = nk*c/(2*np.pi)/4
    H_out = pref*np.abs(h + hd/(1j*nk*cc))**2
    H_in = pref*np.abs(h - hd/(1j*nk*cc))**2
    return q, p, H_out, H_in
