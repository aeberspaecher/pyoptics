#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Image quality metrics for lithography.

Threshold (constant threshold resist) model: a positive resist exposed with
the dose D (relative to the dose-to-clear of the open frame) is removed
where D*I >= threshold, I the image normalized to the open frame. A dark
feature (line on a clear field mask) remains where D*I < threshold, a bright
feature (contact hole, trench) is opened where D*I >= threshold. Its CD is
the distance of the two contour crossings around the feature center.

Metrics:

- contrast(): Michelson contrast,
- image_log_slope(), nils(): (normalized) image log slope at an edge,
- feature_edges(), threshold_cd(): printed edges and CD,
- nils_at_threshold(): NILS at the printed edges,
- dose_to_size(): dose that prints a target CD,
- fem_cd(): focus-exposure matrix of CDs (Bossung curves),
- process_window(), el_dof(): in-spec doses per focus, exposure latitude
  vs. depth of focus,
- gaussian_blur(): a simple resist diffusion model.

The functions work on 1D cuts u, I with u increasing and finely sampled
(linear interpolation between samples).
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from ..vectorial_imaging import contrast, nils


def image_log_slope(x, I, x_edge):
    """``|d ln(I)/dx|`` at x_edge."""

    return np.abs(np.interp(x_edge, x, np.gradient(np.log(I), x)))


def _crossing(u, f, i, j):
    """Linear interpolation of the zero of f between the samples i and j."""

    return u[i] + (u[j] - u[i])*f[i]/(f[i] - f[j])


def feature_edges(u, I, threshold, center=0.0, bright=False):
    """Printed edges of the feature at center.

    Parameters
    ----------
    u, I : arrays
        Cut through the feature (u increasing).
    threshold : number
        Contour level (dose already applied to I).
    center : number, optional
        Position of the feature center.
    bright : bool, optional
        True for a bright feature (prints where I >= threshold).

    Returns
    -------
    left, right : numbers
        Edge positions; nan if the feature does not print (the center is not
        inside the feature region) or the region reaches the end of the cut.
    """

    u, I = np.asarray(u, dtype=float), np.asarray(I, dtype=float)
    f = I - threshold if bright else threshold - I  # > 0 inside the feature
    i_c = int(np.argmin(np.abs(u - center)))
    if f[i_c] <= 0:
        return np.nan, np.nan

    outside = np.nonzero(f <= 0)[0]
    left_out, right_out = outside[outside < i_c], outside[outside > i_c]
    if len(left_out) == 0 or len(right_out) == 0:
        return np.nan, np.nan
    i_l, i_r = left_out[-1], right_out[0]

    return _crossing(u, f, i_l, i_l + 1), _crossing(u, f, i_r - 1, i_r)


def threshold_cd(u, I, threshold, center=0.0, bright=False):
    """Printed CD of the feature at center (nan if it does not print)."""

    left, right = feature_edges(u, I, threshold, center, bright)

    return right - left


def nils_at_threshold(u, I, threshold, center=0.0, bright=False, cd=None):
    """NILS at the printed edges: ``cd*|d ln(I)/du|``, averaged over both edges;
    cd defaults to the printed CD."""

    left, right = feature_edges(u, I, threshold, center, bright)
    if np.isnan(left):
        return np.nan
    w = right - left if cd is None else cd

    return w*np.mean([image_log_slope(u, I, e) for e in (left, right)])


def dose_to_size(u, I, cd_target, threshold=0.3, center=0.0):
    """Dose D that puts the printed edges at center -+ cd_target/2 (mean
    intensity of both edges), D = threshold/I_edge."""

    I_edge = np.mean(np.interp([center - cd_target/2, center + cd_target/2], u, I))

    return threshold/I_edge


def fem_cd(u, images, doses, threshold=0.3, center=0.0, bright=False):
    """Focus-exposure matrix.

    Parameters
    ----------
    u : array
        Cut positions.
    images : sequence of arrays
        Cuts I(u) for each focus.
    doses : array
    threshold, center, bright
        As for threshold_cd().

    Returns
    -------
    cd : array
        Shape (len(doses), len(images)); nan where the feature does not
        print.
    """

    return np.array([[threshold_cd(u, D*np.asarray(I), threshold, center, bright) for I in images]
                     for D in doses])


def process_window(foci, doses, cd, cd_target, tolerance=0.1, n_fine=2001):
    """Range of doses that print within cd_target*(1 -+ tolerance) for each
    focus.

    Parameters
    ----------
    foci, doses : arrays
        Axes of the focus-exposure matrix (doses increasing).
    cd : array
        Shape (len(doses), len(foci)), e.g. from fem_cd().
    cd_target : number
    tolerance : number, optional
        Relative CD tolerance (default 10 %).
    n_fine : int, optional
        Dose samples for the interpolation.

    Returns
    -------
    dose_low, dose_high : arrays of shape (len(foci),)
        Smallest and largest in-spec dose (the contiguous in-spec interval
        closest to the center of the dose range); nan where none.
    """

    doses = np.asarray(doses, dtype=float)
    fine = np.linspace(doses[0], doses[-1], n_fine)
    low, high = np.full(len(foci), np.nan), np.full(len(foci), np.nan)
    for j in range(len(foci)):
        c = np.asarray(cd)[:, j]
        valid = ~np.isnan(c)
        if np.count_nonzero(valid) < 2:
            continue
        c_fine = np.interp(fine, doses[valid], c[valid], left=np.nan, right=np.nan)
        ok = np.abs(c_fine - cd_target) <= tolerance*cd_target
        if not np.any(ok):
            continue
        # contiguous runs of in-spec doses; take the longest
        idx = np.nonzero(ok)[0]
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        run = max(runs, key=len)
        low[j], high[j] = fine[run[0]], fine[run[-1]]

    return low, high


def el_dof(foci, dose_low, dose_high):
    """Exposure latitude vs. depth of focus.

    For every focus window [f_i, f_j], the common dose range is [max(low),
    min(high)]; its exposure latitude is (high - low)/((high + low)/2).

    Returns
    -------
    dof, el : arrays
        Depth of focus (window widths, increasing) and the largest exposure
        latitude (relative, e.g. 0.1 = 10 %) of any window of that width.
    """

    foci = np.asarray(foci, dtype=float)
    best = {}
    for i in range(len(foci)):
        lo, hi = -np.inf, np.inf
        for j in range(i, len(foci)):
            if np.isnan(dose_low[j]):
                break
            lo, hi = max(lo, dose_low[j]), min(hi, dose_high[j])
            if hi <= lo:
                break
            dof = round(foci[j] - foci[i], 9)
            best[dof] = max(best.get(dof, 0.0), (hi - lo)/((hi + lo)/2))
    if not best:
        return np.array([]), np.array([])
    dof = np.array(sorted(best))
    el = np.array([best[d] for d in dof])
    el = np.maximum.accumulate(el[::-1])[::-1]  # a larger EL at a larger DOF also holds for smaller DOF

    return dof, el


def dof_at_el(dof, el, el_min):
    """Largest depth of focus with an exposure latitude of at least el_min
    (0 if none)."""

    ok = np.asarray(el) >= el_min

    return float(np.max(np.asarray(dof)[ok])) if np.any(ok) else 0.0


def gaussian_blur(I, dx, sigma, dy=None):
    """Blur an image with a Gaussian of standard deviation sigma (a simple
    model of acid diffusion in the resist). The image is taken as periodic
    (sample whole periods, e.g. with utils.grid1d()).

    Parameters
    ----------
    I : array
        1D or 2D (Ny, Nx) image.
    dx, dy : numbers
        Grid spacings (dy defaults to dx).
    sigma : number
        Diffusion length.
    """

    I = np.asarray(I, dtype=float)
    if I.ndim == 1:
        return gaussian_filter(I, sigma/dx, mode="wrap")

    return gaussian_filter(I, (sigma/(dx if dy is None else dy), sigma/dx), mode="wrap")


__all__ = ["contrast", "nils", "image_log_slope", "feature_edges", "threshold_cd",
           "nils_at_threshold", "dose_to_size", "fem_cd", "process_window", "el_dof", "dof_at_el",
           "gaussian_blur"]
