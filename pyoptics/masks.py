#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Masks useful for optical applications.
"""

import numpy as np


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

    XX, YY = np.meshgrid(x, y)

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

    XX, YY = np.meshgrid(x, y)

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


# TODO: "soft masks" that are not really binary (i.e. anti-aliased masks)?
