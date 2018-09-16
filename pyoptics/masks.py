#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Masks useful for optical applications.
"""

import numpy as np
import matplotlib.path as mpl_path

from .utils import ensure_meshgrid


def circular_mask(x, y, R, x0=0.0, y0=0.0):
    """Create circular disk.

    Parameters
    ----------
    x, y : arrays
    R : number
        Radius of disk.
    x0, y0 : numbers, optional
        Center coordinates.

    Returns
    -------
    mask : array, int
    """

    XX, YY = ensure_meshrid(x, y)

    mask = np.zeros(np.shape(XX), dtype=np.int)
    mask[(XX -x0)**2 + (YY-y0)**2 <= R**2] = 1

    return mask


def annular_mask(x, y, R_outer, R_inner, x0=0.0, y0=0.0):
    """Create annular disk.

    Parameters
    ----------
    x, y : arrays
    R_outer, R_inner : numbers
        Outer and inner radius of annulus.
    x0, y0 : numbers, optional
        Center coordinates.

    Returns
    -------
    mask : array
    """

    mask = circular_mask(x, y, R_outer, x0, y0)
    mask_inner = circular_mask(x, y, R_inner, x0, y0)
    mask[mask_inner > 0] = 0

    return mask


def rectangluar_mask(x, y, a, b, x0=0.0, y0=0.0, include_1d_output=False):
    """Create rectangluar mask.

    Parameters
    ----------
    x, y : arrays
    a, b : numbers
        Half side lengths of rectangle.
    x0, y0 : numbers, optional
        Center coordinates.
    include_1d_output : boolean, optional
        If True, return signature is (mask, mask_x, mask_y) with mask being the
        outer product of mask_y and mask_x.

    Returns
    -------
    mask : array, int
    """

    mask_x = np.zeros_like(x, dtype=np.int)
    mask_x[((x - x0) >= -a) & ((x - x0) <= +a)] = 1
    mask_y = np.zeros_like(y, dtype=np.int)
    mask_y[((y - y0) >= -b) & ((y - y0) <= +b)] = 1
    mask = np.outer(mask_y, mask_x)

    # TODO: variable signature - is this needed?
    if include_1d_output:
        return mask, mask_x, mask_y
    else:
        return mask


def elliptical_mask(x, y, a, b, x0=0.0, y0=0.0):
    """Create elliptical mask.

    Parameters
    ----------
    x, y : arrays
    a, b : numbers
        Semi-major axis of ellipse.
    x0, y0 : numbers, optional
        Center coordinates.

    Returns
    -------
    mask : array
    """

    XX, YY = ensure_meshrid(x, y)

    mask = np.zeros(np.shape(XX), dtype=np.int)
    mask[(XX -x0)**2/a**2 + (YY-y0)**2/b**2 <= 1] = 1

    return mask


def to_bool(mask):
    """Return a boolean mask from an integer one.

    Parameters
    ----------
    mask : array, integer

    Returns
    -------
    mask_bool : array, boolean
    """

    return np.array(mask, dtype=np.bool)


def to_int(mask):
    """Return integer mask from an integer one.

    Parameters
    ----------
    mask : array, bool

    Returns
    -------
    mask_int : array, integer
    """

    return np.array(mask, dtype=np.int)


def mask_from_polygon(x, y, x_poly, y_poly):
    """Create masks from interior of a convex or concave polygon.

    Parameters
    ----------
    x, y : array
        Coordinates of to create mask of.
    x_poly, y_poly : arrays
        Linear arrays of coordinates on polygonal boundaries.

    Returns
    -------
    mask : array
        Integer mask, 1 for all interior points, 0 for all points outisde given
        polygon.
    """

    X, Y = ensure_meshgrid(x, y)

    path = mpl_path.Path(np.vstack([x_poly, y_poly]).T)
    mask = path.contains_points(np.vstack([X.ravel(), Y.ravel()]).T)
    mask = to_int(np.reshape(mask, X.shape))

    return mask


# TODO: "soft masks" that are not really binary (i.e. anti-aliased masks)?
