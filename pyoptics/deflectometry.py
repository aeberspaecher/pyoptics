#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Phase measuring deflectometry (PMD): the shape of specular surfaces
from the distorted reflection of fringes on a screen.

A camera looks at a mirror-like surface and sees, reflected in it, a
screen that shows phase-shifted sinusoidal fringes along both of its
axes. Phase shifting and temporal unwrapping (as in fringe projection,
`structured_light`, notebook 133, triangulation and structured light)
tell for every camera pixel which screen point it sees. The camera pixel
fixes the direction of the line of sight, the screen point the direction
of the reflected ray towards the screen; the law of reflection makes the
surface normal the bisector of the two, ::

    n = (c + s)/abs(c + s),  c = (C - P)/abs(C - P),  s = (S - P)/abs(S - P),

with the camera centre C, the surface point P and the screen point S. The
slopes are integrated to the height (`freeform.integrate_gradient()`).
Deflectometry measures slopes, not heights: it reaches nanometres on
mid-spatial ripples of a few millimetres that an interferometer barely
sees, measures freeform and strongly curved surfaces without null optics
and needs nothing but a monitor and a camera.

The height-slope ambiguity
--------------------------
The surface point P lies somewhere on the line of sight, and a different
P gives a different normal for the same screen point: one measurement
cannot separate height and slope. The usual way out, followed here, is a
known height at one point (a distance sensor, a mechanical stop) and an
iteration: assume a surface, compute P and the slopes, integrate,
update the surface, repeat (`reconstruct()`). Each iteration shrinks the
height error by about the ratio of the integration length to the
distances of camera and screen. Stereo deflectometry (two cameras)
removes the need for the known point.

Sensitivity
-----------
A screen point found with the uncertainty delta_s at the distance d from
the surface gives the slope uncertainty delta_s/(2 d)
(`slope_resolution()`); with a phase noise of 2 pi/1000 of fringes 5 mm
apart on a screen 500 mm away that is 10 nm/mm (10 urad). Integrated
over a ripple of the period Lambda the height uncertainty is about
slope_error Lambda/(2 pi). The camera is focused on the surface, so the
fringes on the screen are blurred by the defocus; sinusoidal fringes stay
sinusoidal, only their contrast drops (`defocus_contrast()`).

Geometry
--------
World coordinates in mm: the surface under test is the sag Z = f(X, Y)
near Z = 0, seen from above (Z > 0). The camera is a
`structured_light.Pinhole` (positions in mm, focal length in pixels), the
screen a `Screen`. Grids of heights and slopes are indexed [x, y]
(``np.meshgrid(x, y, indexing="ij")``), as `freeform.integrate_gradient()`
expects.

References
----------
M. C. Knauer, J. Kaminski and G. Häusler, "Phase measuring
deflectometry: a new approach to measure specular free-form surfaces",
Proc. SPIE 5457, 366-376 (2004).

P. Su, R. E. Parks, L. Wang, R. P. Angel and J. H. Burge, "Software
configurable optical test system: a computerized reverse Hartmann test",
Appl. Opt. 49, 4404-4412 (2010).

L. Huang, M. Idir, C. Zuo and A. Asundi, "Review of phase measuring
deflectometry", Opt. Lasers Eng. 107, 247-257 (2018).

E. Olesch, C. Faber and G. Häusler, "Deflectometric self-calibration for
arbitrary specular surfaces", Proc. DGaO 112, A3 (2011).

R. T. Frankot and R. Chellappa, "A method for enforcing integrability in
shape from shading algorithms", IEEE Trans. Pattern Anal. Mach. Intell.
10, 439-451 (1988).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

from .freeform import integrate_gradient
from .structured_light import intersect_depth


class Screen(object):
    """A flat screen (monitor) that shows the fringe patterns.

    Parameters
    ----------
    center : sequence of 3 numbers
        Centre of the screen (mm).
    rotation : (3, 3) array
        Rows: the screen's u axis, v axis and normal in world coordinates
        (e.g. `structured_light.look_at()` from the screen towards the
        surface).
    pitch : number
        Pixel pitch (mm).
    size : (int, int)
        Width and height in pixels.
    """

    def __init__(self, center, rotation, pitch, size):
        self.center = np.asarray(center, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        self.pitch = float(pitch)
        self.size = (int(size[0]), int(size[1]))

    def to_world(self, u, v):
        """World points of screen pixel coordinates.

        Parameters
        ----------
        u, v : arrays
            Pixel coordinates (0 at the first pixel).

        Returns
        -------
        array, shape (..., 3)
        """

        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        a = (u - (self.size[0] - 1)/2)*self.pitch
        b = (v - (self.size[1] - 1)/2)*self.pitch
        return self.center + a[..., None]*self.rotation[0] + b[..., None]*self.rotation[1]

    def intersect(self, origins, directions):
        """Pixel coordinates where rays hit the screen plane.

        Parameters
        ----------
        origins, directions : arrays, shape (..., 3)

        Returns
        -------
        u, v : arrays
            Pixel coordinates (NaN for rays parallel to the screen or
            pointing away from it); points outside the screen are not
            masked.
        """

        o = np.asarray(origins, dtype=float)
        d = np.asarray(directions, dtype=float)
        n = self.rotation[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.sum((self.center - o)*n, axis=-1)/np.sum(d*n, axis=-1)
            s = np.where(s > 0, s, np.nan)
        Q = o + s[..., None]*d - self.center
        u = Q @ self.rotation[0]/self.pitch + (self.size[0] - 1)/2
        v = Q @ self.rotation[1]/self.pitch + (self.size[1] - 1)/2
        return u, v

    def contains(self, u, v):
        """True for pixel coordinates on the screen.

        Parameters
        ----------
        u, v : arrays

        Returns
        -------
        array of bool
        """

        return (u >= 0) & (u <= self.size[0] - 1) & (v >= 0) & (v <= self.size[1] - 1)


def reflect(d, n):
    """Mirror directions d at surfaces with the unit normals n.

    Parameters
    ----------
    d, n : arrays, shape (..., 3)

    Returns
    -------
    array, shape (..., 3)
    """

    d = np.asarray(d, dtype=float)
    n = np.asarray(n, dtype=float)
    return d - 2*np.sum(d*n, axis=-1, keepdims=True)*n


def normals_from_slopes(gx, gy):
    """Unit normals (pointing up, +Z) of a surface Z = f(X, Y) with the
    slopes gx = df/dX and gy = df/dY.

    Parameters
    ----------
    gx, gy : arrays

    Returns
    -------
    array, shape (..., 3)
    """

    gx, gy = np.broadcast_arrays(np.asarray(gx, dtype=float), np.asarray(gy, dtype=float))
    n = np.stack([-gx, -gy, np.ones(gx.shape)], axis=-1)
    return n/np.linalg.norm(n, axis=-1, keepdims=True)


def slopes_from_normals(n):
    """Slopes (df/dX, df/dY) of the surface with the unit normals n.

    Parameters
    ----------
    n : array, shape (..., 3)

    Returns
    -------
    gx, gy : arrays
    """

    n = np.asarray(n, dtype=float)
    return -n[..., 0]/n[..., 2], -n[..., 1]/n[..., 2]


def trace(camera, screen, surface, slopes, n_iter=30):
    """Forward model: the screen point that each camera pixel sees in a
    specular surface.

    Parameters
    ----------
    camera : structured_light.Pinhole
        Looking down at the surface.
    screen : Screen
    surface : callable
        surface(X, Y) -> Z.
    slopes : callable
        slopes(X, Y) -> (df/dX, df/dY).
    n_iter : int, optional
        Fixed point iterations of the ray-surface intersection.

    Returns
    -------
    P : array, shape (height, width, 3)
        Surface points seen by the pixels.
    u_s, v_s : arrays, shape (height, width)
        Screen pixel coordinates seen by the pixels; NaN where the
        reflected ray misses the screen.
    """

    uu, vv = camera.pixel_grid()
    d = camera.rays(uu, vv)
    P = intersect_depth(camera.position, d, surface, n_iter=n_iter)
    gx, gy = slopes(P[..., 0], P[..., 1])
    r = reflect(d, normals_from_slopes(gx, gy))
    u_s, v_s = screen.intersect(P, r)
    off = ~screen.contains(u_s, v_s)
    u_s = np.where(off, np.nan, u_s)
    v_s = np.where(off, np.nan, v_s)
    return P, u_s, v_s


def slopes_from_points(camera_position, P, S):
    """Surface slopes from the law of reflection: the normal at P bisects
    the directions towards the camera centre and the screen point S.

    Parameters
    ----------
    camera_position : sequence of 3 numbers
    P : array, shape (..., 3)
        Surface points (assumed or known).
    S : array, shape (..., 3)
        Screen points seen at P.

    Returns
    -------
    gx, gy : arrays
        df/dX and df/dY.
    """

    P = np.asarray(P, dtype=float)
    c = np.asarray(camera_position, dtype=float) - P
    s = np.asarray(S, dtype=float) - P
    n = c/np.linalg.norm(c, axis=-1, keepdims=True) + s/np.linalg.norm(s, axis=-1, keepdims=True)
    return slopes_from_normals(n/np.linalg.norm(n, axis=-1, keepdims=True))


def _surface_function(x, y, Z):
    f = RegularGridInterpolator((x, y), Z, bounds_error=False, fill_value=None)

    def surface(X, Y):
        X, Y = np.broadcast_arrays(X, Y)
        return f(np.stack([X.ravel(), Y.ravel()], axis=-1)).reshape(X.shape)
    return surface


def reconstruct(camera, screen, u_s, v_s, x, y, anchor=(0.0, 0.0, 0.0), z_start=0.0, n_iter=6,
                mask=None):
    """Surface heights from measured screen coordinates.

    Starting from the flat surface Z = z_start, repeat: intersect the lines
    of sight with the current surface, compute the slopes from the law of
    reflection (`slopes_from_points()`), resample them onto the grid (x,
    y), integrate them by least squares and shift the result to pass
    through the known point `anchor`.

    Parameters
    ----------
    camera : structured_light.Pinhole
    screen : Screen
    u_s, v_s : arrays, shape (height, width)
        Measured screen pixel coordinates of all camera pixels (NaN where
        invalid).
    x, y : 1d arrays
        Regular grid of the reconstruction (mm).
    anchor : (x0, y0, z0), optional
        A point of the surface with known height.
    z_start : number, optional
        Height of the initial flat surface.
    n_iter : int, optional
        Iterations.
    mask : array of bool, optional
        Camera pixels to use (default: all with valid screen coordinates).

    Returns
    -------
    Z : array, shape (len(x), len(y))
        Heights on the grid ([x, y] indexing), NaN outside the measured
        area (the convex hull of the surface points; inside the
        integration the slopes are extended to the whole grid with their
        nearest values).
    gx, gy : arrays, shape (len(x), len(y))
        Slopes of the last iteration on the grid.
    history : list of arrays
        Z after each iteration.
    """

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="ij")
    valid = np.isfinite(u_s) & np.isfinite(v_s)
    if mask is not None:
        valid &= mask
    uu, vv = camera.pixel_grid()
    d = camera.rays(uu[valid], vv[valid])
    S = screen.to_world(u_s[valid], v_s[valid])
    Z = np.full(X.shape, float(z_start))
    history = []
    for _ in range(n_iter):
        surface = _surface_function(x, y, Z)
        P = intersect_depth(camera.position, d, surface, n_iter=20)
        sx, sy = slopes_from_points(camera.position, P, S)
        pts = P[:, :2]
        gx = griddata(pts, sx, (X, Y), method="linear")
        gy = griddata(pts, sy, (X, Y), method="linear")
        # outside the measured area: nearest values (integrability only)
        bad = ~np.isfinite(gx) | ~np.isfinite(gy)
        if np.any(bad):
            gx[bad] = griddata(pts, sx, (X[bad], Y[bad]), method="nearest")
            gy[bad] = griddata(pts, sy, (X[bad], Y[bad]), method="nearest")
        Z = integrate_gradient(gx, gy, x[1] - x[0], y[1] - y[0])
        Z = Z - _surface_function(x, y, Z)(np.array(anchor[0]), np.array(anchor[1])) + anchor[2]
        history.append(np.where(bad, np.nan, Z))
    gx[bad], gy[bad] = np.nan, np.nan
    return history[-1], gx, gy, history


def slope_resolution(screen_uncertainty, distance):
    """Slope uncertainty (rad) from the uncertainty of the decoded screen
    position, delta_s/(2 d).

    Parameters
    ----------
    screen_uncertainty : number or array
        Uncertainty of the screen point (same unit as distance).
    distance : number or array
        Distance between surface and screen.

    Returns
    -------
    number or array
    """

    return np.asarray(screen_uncertainty)/(2*np.asarray(distance))


def screen_uncertainty(period, phase_noise):
    """Uncertainty of a screen coordinate decoded from fringes of the
    given period with the phase noise (rad): period phase_noise/(2 pi).

    Parameters
    ----------
    period : number or array
    phase_noise : number or array

    Returns
    -------
    number or array
    """

    return np.asarray(period)*np.asarray(phase_noise)/(2*np.pi)


def defocus_contrast(period, blur_diameter):
    """Contrast of sinusoidal fringes of the given period blurred by a
    uniform disc of the given diameter (geometric defocus): the transfer
    function 2 J1(pi D/p)/(pi D/p). The fringes stay sinusoidal, so the
    phase is unchanged where the contrast is positive.

    Parameters
    ----------
    period : number or array
    blur_diameter : number or array

    Returns
    -------
    number or array
    """

    from scipy.special import j1
    a = np.pi*np.asarray(blur_diameter, dtype=float)/np.asarray(period, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > 1e-9, 2*j1(a)/a, 1.0)


def blur_diameter(aperture, focus_distance, screen_distance):
    """Diameter of the blur of a screen point at the virtual distance
    focus_distance + screen_distance seen by a camera of the given entrance
    pupil diameter focused at focus_distance, measured on the screen
    (flat mirror, thin lens): aperture screen_distance/focus_distance.

    Parameters
    ----------
    aperture : number
        Entrance pupil diameter of the camera.
    focus_distance : number
        Distance camera to surface.
    screen_distance : number
        Distance surface to screen.

    Returns
    -------
    number
    """

    return aperture*screen_distance/focus_distance
