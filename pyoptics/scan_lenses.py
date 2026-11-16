#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Scan lenses for laser scanning and laser processing.

A galvanometer scanner turns a collimated laser beam about a point (or,
with two mirrors, about two points), and a scan lens behind it focuses the
beam onto a flat workpiece. The scanner sits in the entrance pupil of the
lens, in front of it; the object is at infinity and the field is the scan
angle theta.

- An ordinary lens maps the angle to the height h = f*tan(theta). An
  **F-theta lens** has a barrel distortion designed in so that
  h = f*theta: the spot moves at a constant speed for a constant angular
  speed of the mirror, and the scanner control is linear in the angle.
  `ftheta_error()` is the deviation of the real chief ray from f*theta,
  `lens_optimization.ftheta()` the operand that designs it in.
- A **telecentric scan lens** puts its entrance pupil (the scanner) in its
  front focal plane, so that all chief rays leave it parallel to the axis:
  the beam hits the workpiece at normal incidence everywhere (the same
  hole shape and the same spot size over the field, no lateral shift of
  the spot with the focus), at the price of a last lens larger than the
  field. `image_telecentricity()` gives the chief ray angles in image
  space.

Two galvo mirrors (`lidar.galvo_beam()`) are separated along the beam: the
lens can only have its pupil at one point, usually between them. The beam
from the first mirror reaches the lens from a point that moves with the
angle of the second one, and the lines of constant y mirror angle bow on
the workpiece (pincushion in x, the "line bow"). `galvo_spots()` traces
the beam of the two mirrors through a scan lens; `invert_scan()` finds the
mirror angles for given spot positions (Newton's method), and
`CorrectionTable` stores them on a grid and interpolates, the lookup table
of scanner controllers that removes the bow and the residual distortion
of the lens.

The lenses are `raytrace.System` objects with the object at infinity and
the stop (the scanner) as the first surface; lengths in the unit of the
system (mm by default), angles in radians. Notebook 149 (scan lenses)
designs both lenses with `lens_optimization`.

References
----------
W. J. Smith, "Modern Optical Engineering", 4th ed., McGraw-Hill (2008),
section on scanner lenses.

G. F. Marshall and G. E. Stutz (eds.), "Handbook of Optical and Laser
Scanning", 2nd ed., CRC Press (2011).

J. Montagu, "Scanners: galvanometric and resonant", in Marshall and Stutz.

L. Beiser, "Unified Optical Scanning Technology", Wiley (2003).
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

from .lidar import galvo_beam
from .raytrace import Rays

__all__ = ["chief_ray_heights", "ftheta_error", "ftheta_distortion", "image_telecentricity",
           "galvo_spots", "invert_scan", "CorrectionTable"]


def _chief_hits(system, fields):
    """Image plane points and directions of the chief rays at the y field
    angles (aimed at the centre of the stop)."""

    fields = np.atleast_1d(np.asarray(fields, float))
    rays = [system.rays(float(f), 0.0, 0.0) for f in fields]
    joined = Rays(np.concatenate([np.reshape(r.points, (-1, 3)) for r in rays]),
                  np.concatenate([np.reshape(r.directions, (-1, 3)) for r in rays]),
                  np.concatenate([np.reshape(r.opl, -1) for r in rays]))

    return system._image_hits(system.trace(joined, clip=False))


def chief_ray_heights(system, fields):
    """Image heights of the chief rays.

    Parameters
    ----------
    system : raytrace.System
        Scan lens with the object at infinity.
    fields : number or array
        Scan (field) angles in radians.

    Returns
    -------
    array
        y image heights in the image plane.
    """

    return _chief_hits(system, fields)[0][:, 1]


def ftheta_error(system, fields, efl=None):
    """Deviation of the chief ray image height from f*theta.

    Parameters
    ----------
    system : raytrace.System
        Scan lens with the object at infinity.
    fields : number or array
        Scan angles in radians.
    efl : number, optional
        Focal length f of the F-theta law. Default: the paraxial focal
        length of the system.

    Returns
    -------
    array
        h(theta) - f*theta in the length unit.
    """

    f = system.paraxial().efl if efl is None else efl
    fields = np.atleast_1d(np.asarray(fields, float))

    return chief_ray_heights(system, fields) - f*fields


def ftheta_distortion(system, fields, efl=None):
    """F-theta distortion in per cent, (h - f*theta)/(f*theta).

    Parameters
    ----------
    system : raytrace.System
    fields : number or array
        Scan angles in radians (nonzero).
    efl : number, optional
        Default: the paraxial focal length.

    Returns
    -------
    array
    """

    f = system.paraxial().efl if efl is None else efl
    fields = np.atleast_1d(np.asarray(fields, float))

    return 100*ftheta_error(system, fields, f)/(f*fields)


def image_telecentricity(system, fields):
    """Angle of the chief rays to the axis in image space.

    Parameters
    ----------
    system : raytrace.System
    fields : number or array
        Scan angles in radians.

    Returns
    -------
    array
        Angles in radians (positive: the chief ray goes away from the axis).
    """

    D = _chief_hits(system, fields)[1]

    return np.arctan2(D[:, 1], D[:, 2])


def galvo_spots(system, theta_x, theta_y, separation, pupil=0.5):
    """Trace the beam of a two-mirror galvo scanner through a scan lens.

    The scanner of `lidar.galvo_beam()` (x mirror first, y mirror second)
    in front of the lens: unfolded along the beam, the stop plane of the
    lens (its first surface, z = 0) lies `pupil*separation` behind the x
    mirror and `(1 - pupil)*separation` in front of the y mirror (a
    virtual position: the y mirror is behind the stop plane when the
    mirrors straddle it). At rest the beam runs along the axis of the
    lens. Each beam is traced as its central ray.

    Parameters
    ----------
    system : raytrace.System
        Scan lens, object at infinity, stop plane at its first surface.
    theta_x, theta_y : numbers or arrays
        Mechanical mirror angles (rad), broadcast against each other; the
        optical scan angles are about twice these.
    separation : number
        Distance of the mirrors along the beam (length unit of the system).
    pupil : number, optional
        Position of the stop plane between the mirrors: 0 at the x
        mirror, 1 at the y mirror, 0.5 (default) half way.

    Returns
    -------
    x, y : arrays
        Spot positions in the image plane, NaN where the ray fails.
    directions : array, shape (..., 3)
        Directions of the rays at the image plane (telecentricity).
    """

    P, D = galvo_beam(theta_x, theta_y, separation)
    shape = P.shape[:-1]
    P = P.reshape(-1, 3).copy()
    D = D.reshape(-1, 3)
    # galvo frame -> lens frame: the y mirror at (0, separation, 0) is on
    # the lens axis at z = (1 - pupil)*separation
    P[:, 1] -= separation
    P[:, 2] += (1 - pupil)*separation
    scale = max(np.sum(np.abs([s.thickness for s in system.surfaces])), 1.0)
    z0 = -0.1*scale                       # start in front of the stop plane
    P = P + ((z0 - P[:, 2])/D[:, 2])[:, None]*D
    bundle = system.trace(Rays(P, D, np.zeros(len(P))), clip=False)
    I, Dout = system._image_hits(bundle)

    return I[:, 0].reshape(shape), I[:, 1].reshape(shape), Dout.reshape(shape + (3,))


def invert_scan(spots, x, y, theta0=None, iterations=20, step=1e-6, tol=1e-9):
    """Mirror angles that put the spot on given positions (Newton's method).

    Parameters
    ----------
    spots : callable
        spots(theta_x, theta_y) -> (x, y), for arrays of angles (e.g. a
        lambda around `galvo_spots()` or `lidar.galvo_xy()`).
    x, y : arrays
        Target spot positions (broadcast against each other).
    theta0 : pair of arrays, optional
        Start angles. Default: from the derivative of spots at rest (the
        linear scan law).
    iterations : int, optional
        Maximum number of Newton steps.
    step : number, optional
        Angle step of the finite difference Jacobian (rad).
    tol : number, optional
        Stop when all position errors are below this (length unit).

    Returns
    -------
    theta_x, theta_y : arrays
        Mechanical angles.
    error : array
        Remaining distance from the targets.
    """

    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if theta0 is None:
        x0, y0 = spots(np.array([0.0, step, 0.0]), np.array([0.0, 0.0, step]))
        sx, sy = (x0[1] - x0[0])/step, (y0[2] - y0[0])/step
        tx, ty = x/sx, y/sy
    else:
        tx, ty = (np.array(np.broadcast_to(t, x.shape), float) for t in theta0)
    for _ in range(iterations):
        X, Y = spots(np.stack([tx, tx + step, tx]), np.stack([ty, ty, ty + step]))
        ex, ey = X[0] - x, Y[0] - y
        err = np.hypot(ex, ey)
        if np.all(err < tol):
            break
        a, b = (X[1] - X[0])/step, (X[2] - X[0])/step
        c, d = (Y[1] - Y[0])/step, (Y[2] - Y[0])/step
        det = a*d - b*c
        tx = tx - (d*ex - b*ey)/det
        ty = ty - (-c*ex + a*ey)/det
    X, Y = spots(tx, ty)

    return tx, ty, np.hypot(X - x, Y - y)


class CorrectionTable:
    """Lookup table of mirror angles over a grid of spot positions.

    Scanner controllers store the mirror angles for a grid of positions
    on the workpiece (measured, or computed from the design) and
    interpolate between the nodes.

    Parameters
    ----------
    spots : callable
        spots(theta_x, theta_y) -> (x, y), as for `invert_scan()`.
    x_nodes, y_nodes : arrays
        Grid of target positions (increasing).
    method : str, optional
        "linear" (bilinear, default, as most controllers do) or "cubic"
        (an interpolating bicubic spline, `scipy.interpolate.RectBivariateSpline`).
    """

    def __init__(self, spots, x_nodes, y_nodes, method="linear"):
        self.x_nodes = np.asarray(x_nodes, float)
        self.y_nodes = np.asarray(y_nodes, float)
        X, Y = np.meshgrid(self.x_nodes, self.y_nodes, indexing="ij")
        self.theta_x, self.theta_y, self.error = invert_scan(spots, X, Y)
        grid = (self.x_nodes, self.y_nodes)
        if method == "linear":
            self._ix = RegularGridInterpolator(grid, self.theta_x)
            self._iy = RegularGridInterpolator(grid, self.theta_y)
        elif method == "cubic":
            ix = RectBivariateSpline(*grid, self.theta_x, kx=3, ky=3, s=0)
            iy = RectBivariateSpline(*grid, self.theta_y, kx=3, ky=3, s=0)
            self._ix = lambda q: ix(q[:, 0], q[:, 1], grid=False)
            self._iy = lambda q: iy(q[:, 0], q[:, 1], grid=False)
        else:
            raise ValueError("method must be 'linear' or 'cubic'")

    def angles(self, x, y):
        """Interpolated mirror angles for spot positions.

        Parameters
        ----------
        x, y : arrays
            Target positions inside the grid (broadcast).

        Returns
        -------
        theta_x, theta_y : arrays
        """

        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        q = np.stack([x.ravel(), y.ravel()], axis=-1)

        return self._ix(q).reshape(x.shape), self._iy(q).reshape(x.shape)

    def __call__(self, x, y):
        """Same as `angles()`.

        Parameters
        ----------
        x, y : arrays
            Target positions.

        Returns
        -------
        theta_x, theta_y : arrays
        """

        return self.angles(x, y)
