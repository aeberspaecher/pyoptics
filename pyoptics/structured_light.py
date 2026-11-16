#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Triangulation and structured light: laser triangulation sensors, fringe
projection with phase shifting and temporal phase unwrapping, and the dot
projector of active stereo depth cameras.

Pinhole cameras and projectors
------------------------------
`Pinhole` is an ideal camera (or projector, which is a camera run
backwards) with its centre, orientation (`look_at()`), focal length in
pixels and principal point. `Pinhole.project()` maps world points to pixel
coordinates, `Pinhole.rays()` pixels to rays. Scenes are depth maps
Z = depth(X, Y) in world coordinates; `intersect_depth()` finds where rays
meet them (fixed point iteration, valid for slopes below about 1 seen
along the rays; no occlusion).

Laser triangulation
-------------------
A laser at the baseline b from a camera lens of focal length f sends a
beam at the angle theta towards the camera axis. The spot appears at
x = f*(b/Z - tan(theta)) on the sensor, so ::

    Z = f*b/(x + f*tan(theta)),   dZ/dx = -Z**2/(f*b)

(`triangulation_depth()`, `triangulation_sensitivity()`): the depth
resolution falls with the square of the distance. On a rough surface the
spot is a speckle pattern whose centroid wanders; Dorsch, Häusler and
Herrmann's limit is ::

    delta_Z = C*lambda/(2*pi*sin(u_obs)*sin(theta))

(`speckle_depth_uncertainty()`), with the observation aperture sin(u_obs)
and the speckle contrast C, independent of the pixel size.

Fringe projection
-----------------
A projector shows sinusoidal fringes along its column coordinate u_p,
``I_k = a + b*cos(2*pi*u_p/p - delta_k)`` with N phase steps
(`fringe_patterns()`, with an optional gamma that distorts them). The
N-step algorithm (`interferometry.psi_n_step()`) recovers the wrapped
phase 2*pi*u_p/p for each camera pixel; `temporal_unwrap()` unwraps it
with a sequence of coarser periods, the coarsest covering the whole
projector. Each camera pixel then knows which projector column lit it,
and `triangulate_column()` intersects the camera ray with the plane of
light of that column.

Dot projector and active stereo
-------------------------------
A depth camera of the kind in smartphones projects tens of thousands of
near infrared dots: a VCSEL array in the focal plane of a collimator
gives a few hundred beams, and a diffractive fan-out grating replicates
them into a grid of orders. `fanout_ifta()` designs the phase of one
period of the grating with an iterative Fourier transform algorithm
(signal orders constrained, the rest free, weights adapted for equal
orders, optionally quantized to a number of levels); `order_efficiencies()`
evaluates a phase. `dot_directions()` combines emitters and orders into
the directions of all dots, `render_dots()` images them onto a camera,
`block_match()` finds the shift of each window of the image against an
image of a reference plane (zero mean normalized cross correlation,
sub-pixel peak), and `depth_from_disparity()` turns the disparity d into
depth, ``1/Z = 1/Z_ref + d/(f*b)``.

References
----------
- R. G. Dorsch, G. Häusler and J. M. Herrmann, "Laser triangulation:
  fundamental uncertainty in distance measurement", Appl. Opt. 33, 1306
  (1994).
- S. Zhang, "High-speed 3D shape measurement with structured light
  methods: a review", Opt. Lasers Eng. 106, 119 (2018).
- C. Zuo, L. Huang, M. Zhang, Q. Chen and A. Asundi, "Temporal phase
  unwrapping algorithms for fringe projection profilometry: a comparative
  review", Opt. Lasers Eng. 85, 84 (2016).
- M. Takeda and K. Mutoh, "Fourier transform profilometry for the
  automatic measurement of 3-D object shapes", Appl. Opt. 22, 3977 (1983).
- B. Freedman, A. Shpunt, M. Machline and Y. Arieli, "Depth mapping using
  projected patterns", US patent 8,150,142 (2012).
- F. Wyrowski, "Diffractive optical elements: iterative calculation of
  quantized, blazed phase structures", J. Opt. Soc. Am. A 7, 961 (1990).
"""

import numpy as np
from scipy.ndimage import uniform_filter

from .unwrap import unwrap_with_reference


__all__ = ["look_at", "Pinhole", "intersect_depth", "triangulation_depth",
           "triangulation_sensitivity", "speckle_depth_uncertainty", "fringe_patterns",
           "temporal_unwrap", "triangulate_column", "fanout_ifta", "order_efficiencies",
           "dot_directions", "render_dots", "block_match", "depth_from_disparity"]


TWO_PI = 2*np.pi


# --------------------------------------------------------------------------
# pinhole cameras and scenes
# --------------------------------------------------------------------------

def look_at(position, target, up=(0.0, 1.0, 0.0)):
    """Rotation matrix of a camera at `position` looking at `target`.

    Parameters
    ----------
    position, target : sequences of 3 numbers
    up : sequence of 3 numbers, optional
        World direction that becomes (close to) the camera's +y axis.

    Returns
    -------
    R : (3, 3) array
        Rows: the camera's x, y and z (viewing) axes in world coordinates.
    """

    z = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, dtype=float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


class Pinhole(object):
    """An ideal pinhole camera or projector.

    Parameters
    ----------
    position : sequence of 3 numbers
        Centre of projection.
    rotation : (3, 3) array
        Rows: camera x, y, z axes in world coordinates (`look_at()`).
    focal : number
        Focal length in pixels.
    size : (int, int)
        Width and height in pixels.
    center : (number, number), optional
        Principal point (u, v); defaults to the middle of the image.
    """

    def __init__(self, position, rotation, focal, size, center=None):
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        self.focal = float(focal)
        self.size = (int(size[0]), int(size[1]))
        if center is None:
            center = ((self.size[0] - 1)/2, (self.size[1] - 1)/2)
        self.center = (float(center[0]), float(center[1]))

    def project(self, points):
        """Pixel coordinates of world points.

        Parameters
        ----------
        points : array, shape (..., 3)

        Returns
        -------
        u, v : arrays
            Pixel coordinates (NaN behind the camera).
        """

        P = (np.asarray(points, dtype=float) - self.position) @ self.rotation.T
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(P[..., 2] > 0, P[..., 2], np.nan)
            u = self.focal*P[..., 0]/Z + self.center[0]
            v = self.focal*P[..., 1]/Z + self.center[1]
        return u, v

    def rays(self, u, v):
        """Unit directions (world) of the rays through pixel coordinates.

        Parameters
        ----------
        u, v : numbers or arrays

        Returns
        -------
        d : array, shape (..., 3)
        """

        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        c = np.stack([(u - self.center[0])/self.focal, (v - self.center[1])/self.focal,
                      np.ones(u.shape)], axis=-1)
        d = c @ self.rotation
        return d/np.linalg.norm(d, axis=-1, keepdims=True)

    def pixel_grid(self):
        """Pixel coordinates of all pixels.

        Returns
        -------
        u, v : arrays of shape (height, width)
        """

        return np.meshgrid(np.arange(self.size[0], dtype=float),
                           np.arange(self.size[1], dtype=float))


def intersect_depth(origin, directions, depth, n_iter=40):
    """Points where rays meet a depth map surface Z = depth(X, Y).

    Parameters
    ----------
    origin : sequence of 3 numbers
        Common origin of the rays.
    directions : array, shape (..., 3)
        Ray directions with a positive z component.
    depth : callable
        depth(X, Y) -> Z for arrays X, Y.
    n_iter : int, optional
        Fixed point iterations.

    Returns
    -------
    P : array, shape (..., 3)
    """

    o = np.asarray(origin, dtype=float)
    d = np.asarray(directions, dtype=float)
    s = (depth(o[0] + 0*d[..., 0], o[1] + 0*d[..., 1]) - o[2])/d[..., 2]
    for _ in range(n_iter):
        X = o[0] + s*d[..., 0]
        Y = o[1] + s*d[..., 1]
        s = (depth(X, Y) - o[2])/d[..., 2]
    return o + s[..., None]*d


# --------------------------------------------------------------------------
# laser triangulation
# --------------------------------------------------------------------------

def triangulation_depth(x_image, baseline, focal, laser_angle=0.0):
    """Depth of a laser spot from its image position.

    The camera looks along +Z from the origin; the laser sits at X = b and
    points its beam at the angle theta towards the camera axis. The spot
    images to x = f*(b/Z - tan(theta)).

    Parameters
    ----------
    x_image : number or array
        Position of the spot on the sensor (same unit as focal).
    baseline : number
    focal : number
        Image distance of the camera lens.
    laser_angle : number, optional
        rad.

    Returns
    -------
    Z : number or array
        ``f*b/(x + f*tan(theta))``.
    """

    return focal*baseline/(np.asarray(x_image, dtype=float) + focal*np.tan(laser_angle))


def triangulation_sensitivity(Z, baseline, focal):
    """Depth change per unit shift of the spot on the sensor, Z**2/(f*b).

    Parameters
    ----------
    Z : number or array
    baseline, focal : numbers

    Returns
    -------
    number or array
    """

    return np.asarray(Z, dtype=float)**2/(focal*baseline)


def speckle_depth_uncertainty(wavelength, sin_u, triangulation_angle, contrast=1.0):
    """Speckle limit of laser triangulation (Dorsch, Häusler and Herrmann
    1994), ``C*lambda/(2*pi*sin(u_obs)*sin(theta))``.

    Parameters
    ----------
    wavelength : number
    sin_u : number or array
        Numerical aperture of the observation.
    triangulation_angle : number or array
        Angle between illumination and observation (rad).
    contrast : number, optional
        Speckle contrast C (1 for fully developed speckle).

    Returns
    -------
    number or array
        Standard deviation of the depth.
    """

    return contrast*wavelength/(TWO_PI*np.asarray(sin_u)*np.sin(triangulation_angle))


# --------------------------------------------------------------------------
# fringe projection
# --------------------------------------------------------------------------

def fringe_patterns(u, period, n_steps, gamma=1.0, offset=0.5, amplitude=0.5):
    """Phase shifted fringe patterns along the projector column u.

    Parameters
    ----------
    u : array
        Projector column coordinates (pixels).
    period : number
        Fringe period (pixels).
    n_steps : int
        Number of phase steps delta_k = 2*pi*k/N.
    gamma : number, optional
        The projector shows I**gamma for the commanded I (gamma = 1 is a
        linear projector).
    offset, amplitude : numbers, optional

    Returns
    -------
    I : array, shape u.shape + (n_steps,)
        ``(a + b*cos(2*pi*u/p - delta_k))**gamma``; psi_n_step() returns the
        phase 2*pi*u/p (wrapped).
    """

    d = TWO_PI*np.arange(n_steps)/n_steps
    I = offset + amplitude*np.cos(TWO_PI*np.asarray(u, dtype=float)[..., None]/period - d)
    return np.clip(I, 0, None)**gamma


def temporal_unwrap(phases, periods):
    """Projector coordinate from wrapped phases of several fringe periods.

    Parameters
    ----------
    phases : sequence of arrays
        Wrapped phases 2*pi*u/p_k (as from psi_n_step()), coarsest first.
    periods : sequence of numbers
        The periods p_k (pixels), decreasing. The first must cover the
        whole projector (p_0 >= its width) so that its phase is
        unambiguous; each next one is unwrapped with the coordinate from
        the previous one, which works while its error stays below p_k/2.

    Returns
    -------
    u : array
        Projector coordinate from the finest period.
    """

    phases = [np.asarray(p, dtype=float) for p in phases]
    u = periods[0]*np.mod(phases[0], TWO_PI)/TWO_PI
    for psi, p in zip(phases[1:], periods[1:]):
        phi = unwrap_with_reference(psi, TWO_PI*u/p)
        u = p*phi/TWO_PI
    return u


def triangulate_column(camera, projector, u_cam, v_cam, u_proj):
    """3D points from camera pixels and the projector columns that lit them.

    Parameters
    ----------
    camera, projector : Pinhole
    u_cam, v_cam : arrays
        Camera pixel coordinates.
    u_proj : array
        Projector column coordinate seen at each pixel.

    Returns
    -------
    P : array, shape (..., 3)
        Intersection of each camera ray with the plane through the
        projector centre that contains projector column u_proj.
    """

    d = camera.rays(u_cam, v_cam)
    vc = projector.center[1]
    r1 = projector.rays(u_proj, vc)
    r2 = projector.rays(u_proj, vc + 1.0)
    n = np.cross(r1, r2)
    s = np.sum(n*(projector.position - camera.position), axis=-1)/np.sum(n*d, axis=-1)
    return camera.position + s[..., None]*d


# --------------------------------------------------------------------------
# dot projector
# --------------------------------------------------------------------------

def _order_index(orders, n):
    o = np.asarray(orders, dtype=int)
    return o[:, 1] % n, o[:, 0] % n


def order_efficiencies(phase, orders):
    """Diffraction efficiencies of orders of one period of a phase grating.

    Parameters
    ----------
    phase : (n, n) array
        Phase (rad) of one period, sampled with n pixels per period.
    orders : array, shape (m, 2)
        Orders (m_x, m_y).

    Returns
    -------
    eta : array, shape (m,)
        Fraction of the incident power in each order (thin element, scalar
        theory).
    """

    n = phase.shape[0]
    F = np.fft.fft2(np.exp(1j*phase))/phase.size
    iy, ix = _order_index(orders, n)
    return np.abs(F[iy, ix])**2


def fanout_ifta(n, orders, n_iter=200, levels=None, rng=None, quantize_from=0.5,
                refine_sweeps=2):
    """Phase of a fan-out grating that splits a beam into equal orders.

    Iterative Fourier transform algorithm on one period of n x n pixels:
    the amplitudes of the signal orders are set to their target, all other
    orders are left free (they collect the noise), the phase of the
    element is kept and its amplitude set to 1. The targets are adapted to
    even out the orders, and from the iteration n_iter*quantize_from on the
    phase is pulled step by step to `levels` equally spaced levels, after
    which a pixel by pixel search polishes the quantized element.

    Parameters
    ----------
    n : int
        Pixels per period (along both axes).
    orders : array, shape (m, 2)
        Signal orders (m_x, m_y), |m| < n/2.
    n_iter : int, optional
    levels : int, optional
        Number of phase levels of the fabricated element (None: continuous).
    rng : numpy Generator or seed, optional
        For the random start phase.
    quantize_from : number, optional
        Fraction of the iterations after which quantization starts.
    refine_sweeps : int, optional
        With levels: sweeps of a pixel by pixel search over the levels
        after the IFTA, which keeps changes that raise the efficiency
        times the uniformity (the IFTA alone leaves 10 to 20 %
        nonuniformity for few levels).

    Returns
    -------
    phase : (n, n) array
        Phase of one period (rad), in [0, 2*pi).
    eta : array
        Efficiencies of the signal orders (`order_efficiencies()`).
    """

    rng = np.random.default_rng(rng)
    orders = np.asarray(orders, dtype=int)
    iy, ix = _order_index(orders, n)
    phase = TWO_PI*rng.random((n, n))
    w = np.ones(len(orders))
    q0 = int(n_iter*quantize_from)
    for it in range(n_iter):
        F = np.fft.fft2(np.exp(1j*phase))
        a = np.abs(F[iy, ix])
        if it > 5:
            w *= np.sqrt(a.mean()/np.maximum(a, 1e-12))**0.5
        G = F.copy()
        target = w*np.sqrt(np.sum(np.abs(F)**2)/np.sum(w**2))
        G[iy, ix] = target*np.exp(1j*np.angle(F[iy, ix]))
        phase = np.mod(np.angle(np.fft.ifft2(G)), TWO_PI)
        if levels is not None and it >= q0:
            step = TWO_PI/levels
            frac = min(1.0, (it - q0 + 1)/max(1, n_iter - q0 - 10))
            qd = step*np.round(phase/step)
            phase = np.mod(phase + frac*(qd - phase), TWO_PI)
    if levels is not None:
        step = TWO_PI/levels
        phase = np.mod(step*np.round(phase/step), TWO_PI)
        phase = _refine_levels(phase, orders, levels, refine_sweeps, rng)
    return phase, order_efficiencies(phase, orders)


def _uniform_cost(eta):
    return -eta.sum()*(1 - (eta.max() - eta.min())/(eta.max() + eta.min()))


def _refine_levels(phase, orders, levels, sweeps, rng):
    """Pixel by pixel search on a quantized phase: try the other levels of
    each pixel and keep a change if it raises the efficiency times the
    uniformity of the signal orders (updated in O(m) per trial)."""

    n = phase.shape[0]
    iy, ix = _order_index(orders, n)
    t = np.exp(1j*phase)
    F = np.fft.fft2(t)[iy, ix]/phase.size
    cost = _uniform_cost(np.abs(F)**2)
    vals = np.exp(1j*TWO_PI*np.arange(levels)/levels)
    for _ in range(sweeps):
        for p in rng.permutation(phase.size):
            r, c = divmod(p, n)
            ker = np.exp(-2j*np.pi*(iy*r + ix*c)/n)/phase.size
            for v in vals:
                if v == t[r, c]:
                    continue
                F_new = F + (v - t[r, c])*ker
                cost_new = _uniform_cost(np.abs(F_new)**2)
                if cost_new < cost:
                    F, cost, t[r, c] = F_new, cost_new, v
    return np.mod(np.angle(t), TWO_PI)


def dot_directions(emitters, focal_length, orders, period, wavelength):
    """Directions of the dots of a VCSEL array, collimator and fan-out
    grating.

    Parameters
    ----------
    emitters : array, shape (k, 2)
        Positions of the emitters in the focal plane of the collimator.
    focal_length : number
        Same unit as the emitter positions.
    orders : array, shape (m, 2)
        Orders of the fan-out grating.
    period : number
        Grating period (same unit as the wavelength).
    wavelength : number

    Returns
    -------
    k : array, shape (m*k, 2)
        Direction cosines (k_x, k_y) of all dots: the emitter at e leaves
        the collimator with -e/sqrt(|e|**2 + f**2), and the grating adds
        m*lambda/period. Directions with |k| >= 1 are evanescent and
        dropped.
    """

    e = np.asarray(emitters, dtype=float)
    k_e = -e/np.sqrt(np.sum(e**2, axis=1, keepdims=True) + focal_length**2)
    k_o = np.asarray(orders, dtype=float)*wavelength/period
    k = (k_o[:, None, :] + k_e[None, :, :]).reshape(-1, 2)
    return k[np.sum(k**2, axis=1) < 1]


def render_dots(camera, source, directions, depth, sigma=0.8, weights=None, n_iter=40):
    """Camera image of dots projected onto a depth map surface.

    Parameters
    ----------
    camera : Pinhole
    source : sequence of 3 numbers
        Position of the dot projector; it looks along +Z.
    directions : array, shape (m, 2)
        Direction cosines (k_x, k_y) of the dots.
    depth : callable
        depth(X, Y) -> Z.
    sigma : number, optional
        rms radius of a dot image (pixels).
    weights : array, optional
        Relative powers of the dots.
    n_iter : int, optional

    Returns
    -------
    image : (height, width) array
    uv : array, shape (m, 2)
        Image positions of the dots.
    """

    k = np.asarray(directions, dtype=float)
    d = np.c_[k, np.sqrt(1 - np.sum(k**2, axis=1))]
    P = intersect_depth(source, d, depth, n_iter)
    u, v = camera.project(P)
    w = np.ones(len(u)) if weights is None else np.asarray(weights, dtype=float)
    W, Hh = camera.size
    img = np.zeros((Hh, W))
    r = int(np.ceil(4*sigma))
    ok = np.isfinite(u) & (u > -r) & (u < W + r) & (v > -r) & (v < Hh + r)
    offs = np.arange(-r, r + 1)
    for uu, vv, ww in zip(u[ok], v[ok], w[ok]):
        i0, j0 = int(round(vv)), int(round(uu))
        ii = i0 + offs
        jj = j0 + offs
        gi = np.exp(-(ii - vv)**2/(2*sigma**2))
        gj = np.exp(-(jj - uu)**2/(2*sigma**2))
        mi = (ii >= 0) & (ii < Hh)
        mj = (jj >= 0) & (jj < W)
        img[np.ix_(ii[mi], jj[mj])] += ww*np.outer(gi[mi], gj[mj])
    return img, np.c_[u, v]


def block_match(image, reference, window=9, max_disparity=32, min_disparity=None):
    """Horizontal disparity of an image against a reference image.

    For every pixel, the window around it is compared with the windows of
    the reference shifted by d = min_disparity ... max_disparity pixels,
    image(x) ~ reference(x - d), by the zero mean normalized cross
    correlation; the best integer d is refined with a parabola through the
    neighbouring scores.

    Parameters
    ----------
    image, reference : 2D arrays
    window : int, optional
        Side of the square window (odd).
    max_disparity : int, optional
    min_disparity : int, optional
        Defaults to -max_disparity.

    Returns
    -------
    d : array
        Disparity (pixels).
    score : array
        Correlation at the best disparity (1: perfect match), a confidence.
    """

    if min_disparity is None:
        min_disparity = -max_disparity
    I = np.asarray(image, dtype=float)
    R = np.asarray(reference, dtype=float)

    def stats(A):
        m = uniform_filter(A, window, mode="nearest")
        v = uniform_filter(A*A, window, mode="nearest") - m**2
        return m, np.sqrt(np.maximum(v, 1e-12))

    mI, sI = stats(I)
    ds = np.arange(min_disparity, max_disparity + 1)
    scores = np.empty((len(ds),) + I.shape)
    for k, d in enumerate(ds):
        Rs = np.roll(R, d, axis=1)
        mR, sR = stats(Rs)
        cov = uniform_filter(I*Rs, window, mode="nearest") - mI*mR
        scores[k] = cov/(sI*sR)
    best = np.argmax(scores, axis=0)
    score = np.take_along_axis(scores, best[None], 0)[0]
    km = np.clip(best - 1, 0, len(ds) - 1)
    kp = np.clip(best + 1, 0, len(ds) - 1)
    s_m = np.take_along_axis(scores, km[None], 0)[0]
    s_p = np.take_along_axis(scores, kp[None], 0)[0]
    den = s_m - 2*score + s_p
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where((den < 0) & (km != best) & (kp != best), 0.5*(s_m - s_p)/den, 0.0)
    return ds[best] + frac, score


def depth_from_disparity(disparity, baseline, focal, z_ref):
    """Depth from the disparity against a reference plane,
    ``1/Z = 1/Z_ref + d/(f*b)``.

    Parameters
    ----------
    disparity : number or array
        Shift d of the pattern (pixels) against the reference image,
        positive for points closer than the reference (projector at +b
        along the camera's x axis).
    baseline : number
    focal : number
        Focal length in pixels.
    z_ref : number
        Depth of the reference plane.

    Returns
    -------
    Z : number or array
    """

    return 1/(1/z_ref + np.asarray(disparity, dtype=float)/(focal*baseline))
