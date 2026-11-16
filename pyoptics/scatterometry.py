#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Scatterometry: the profile of a periodic structure from its zero order
reflection, computed with the RCWA of `litho.rcwa`.

Lines and spaces smaller than the wavelength cannot be imaged, but they
change the polarization and the spectrum of the specularly reflected
light. Optical critical dimension (OCD) metrology measures that zero
order with a spectroscopic ellipsometer or reflectometer, computes it
for a parameterized model of the profile (linewidth CD, height, sidewall
angle, the thicknesses of the films below) with a rigorous solver, and
finds the parameters that reproduce the measurement: an inverse problem,
solved by searching a precomputed library of signatures or by nonlinear
regression (`ellipsometry.fit_psi_delta()`).

This module provides the forward model only; the fits are done in
notebook 136 (scatterometry) as a teaching example of an inverse problem.
It deliberately has no calibration or library management helpers.

Profiles
--------
`trapezoid_slices()` cuts a trapezoidal line (bottom CD, height, sidewall
angle, optional rounding of the top corners) into lamellar slices, the
staircase approximation of RCWA. `grating_layers()` turns them into
`litho.rcwa.Layer` objects at one wavelength, with any uniform films
between the lines and the substrate (and an optional asymmetry of the
two sidewalls, the signature of an etch tilt).

Signatures
----------
`zero_order_jones()` returns the Jones matrix of the specular order in
the p-s basis of the plane of incidence, in the Nebraska convention of
`ellipsometry` (rho = J_pp/J_ss = tan(Psi) exp(i Delta) for a planar
mount). The plane of incidence may be rotated by the azimuth phi against
the grating vector (conical mount); then the lines couple p and s, the
off-diagonal elements are non-zero and the sample is described by a full
Mueller matrix (`mueller_from_jones()`). In the planar mount and for a
symmetric profile the off-diagonal blocks of the Mueller matrix vanish;
a sidewall asymmetry seen in a conical mount makes them non-zero.
`psi_delta()` and `mueller_from_jones()` give the usual signatures.

References
----------
C. J. Raymond, M. R. Murnane, S. S. H. Naqvi and J. R. McNeil,
"Metrology of subwavelength photoresist gratings using optical
scatterometry", J. Vac. Sci. Technol. B 13, 1484-1495 (1995).

X. Niu, N. Jakatdar, J. Bao and C. J. Spanos, "Specular spectroscopic
scatterometry", IEEE Trans. Semicond. Manuf. 14, 97-111 (2001).

H.-T. Huang, W. Kong and F. L. Terry, "Normal-incidence spectroscopic
ellipsometry for critical dimension monitoring", Appl. Phys. Lett. 78,
3983-3985 (2001).

T. Novikova, A. De Martino, S. Ben Hatit and B. Drevillon, "Application
of Mueller polarimetry in conical diffraction for critical dimension
measurements in microelectronics", Appl. Opt. 45, 3688-3697 (2006).

M. G. Moharam, E. B. Grann, D. A. Pommet and T. K. Gaylord, "Formulation
for stable and efficient implementation of the rigorous coupled-wave
analysis of binary gratings", J. Opt. Soc. Am. A 12, 1068-1076 (1995).
"""

import numpy as np

from .litho.rcwa import Layer, rcwa
from .polarization.mueller import M_from_jones
from .ellipsometry import psi_delta_from_rho


def _index(n, wavelength):
    return complex(n(wavelength) if callable(n) else n)


def trapezoid_slices(cd, height, swa, n_slices=10, top_rounding=0.0, asymmetry=0.0):
    """Lamellar slices of a trapezoidal line.

    Parameters
    ----------
    cd : number
        Bottom width of the line.
    height : number
    swa : number
        Sidewall angle (degrees, 90 for vertical walls, < 90 for a line
        that narrows towards the top).
    n_slices : int, optional
    top_rounding : number, optional
        Radius of a circular rounding of both top corners (0 for none).
    asymmetry : number, optional
        Difference of the two sidewall angles (degrees): the left wall has
        swa + asymmetry/2, the right one swa - asymmetry/2.

    Returns
    -------
    z_mid, centers, widths, thicknesses : arrays
        Height of the middle of each slice above the bottom, centre and
        width of the line in each slice, and the slice thicknesses; from
        the top slice down (the order of RCWA layers).
    """

    t = height/n_slices
    z = (np.arange(n_slices)[::-1] + 0.5)*t
    a_l = np.radians(swa + asymmetry/2)
    a_r = np.radians(swa - asymmetry/2)
    left = -cd/2 + z/np.tan(a_l)
    right = cd/2 - z/np.tan(a_r)
    if top_rounding > 0:
        r = min(top_rounding, height, 0.5*(right[0] - left[0]) + 1e-300)
        dz = z - (height - r)
        inset = np.where(dz > 0, r - np.sqrt(np.clip(r**2 - dz**2, 0, None)), 0.0)
        left, right = left + inset, right - inset
    widths = np.clip(right - left, 0, None)
    return z, 0.5*(left + right), widths, np.full(n_slices, t)


def profile_outline(cd, height, swa, n_slices=40, top_rounding=0.0, asymmetry=0.0):
    """Outline of a line for plotting.

    Parameters
    ----------
    cd, height, swa, n_slices, top_rounding, asymmetry
        As in `trapezoid_slices()`.

    Returns
    -------
    x, z : arrays
        Closed polygon around the line (bottom at z = 0).
    """

    z, c, w, t = trapezoid_slices(cd, height, swa, n_slices, top_rounding, asymmetry)
    zz = np.concatenate([[0.0], z[::-1], [height]])
    a_l = np.radians(swa + asymmetry/2)
    a_r = np.radians(swa - asymmetry/2)
    left = np.concatenate([[-cd/2], (c - w/2)[::-1], [(c - w/2)[0]]])
    right = np.concatenate([[cd/2], (c + w/2)[::-1], [(c + w/2)[0]]])
    if top_rounding == 0:
        left[-1] = -cd/2 + height/np.tan(a_l)
        right[-1] = cd/2 - height/np.tan(a_r)
    return np.concatenate([left, right[::-1], [left[0]]]), np.concatenate([zz, zz[::-1], [0.0]])


def grating_layers(wavelength, period, cd, height, swa, n_line, n_space=1.0, films=(),
                   n_slices=10, top_rounding=0.0, asymmetry=0.0):
    """RCWA layers of a line grating of trapezoidal lines on films.

    Parameters
    ----------
    wavelength : number
    period : number
    cd, height, swa : numbers
        Bottom CD, height and sidewall angle (degrees) of the lines.
    n_line, n_space : number or callable of the wavelength
        Indices of the lines and of the medium between them.
    films : sequence of (index, thickness), optional
        Uniform films below the lines, from the top down.
    n_slices, top_rounding, asymmetry : optional
        As in `trapezoid_slices()`.

    Returns
    -------
    list of litho.rcwa.Layer
    """

    nl, ns = _index(n_line, wavelength), _index(n_space, wavelength)
    _, c, w, t = trapezoid_slices(cd, height, swa, n_slices, top_rounding, asymmetry)
    layers = [Layer(ti, ns, [(ci, min(wi, period*(1 - 1e-9)), nl)] if wi > 0 else [])
              for ci, wi, ti in zip(c, w, t)]
    layers += [Layer(d, _index(n, wavelength)) for n, d in films]
    return layers


def zero_order_jones(period, layers, wavelengths, angle, substrate, azimuth=0.0, ambient=1.0,
                     n_orders=10):
    """Jones matrix of the specular order of a grating in the p-s basis.

    Parameters
    ----------
    period : number
    layers : callable
        layers(wavelength) -> list of litho.rcwa.Layer, from the top down
        (e.g. a lambda around `grating_layers()`).
    wavelengths : number or array
    angle : number
        Angle of incidence (degrees).
    substrate : number or callable of the wavelength
        Index of the substrate (may be absorbing).
    azimuth : number, optional
        Angle (degrees) between the plane of incidence and the grating
        vector (0: planar mount, the plane of incidence across the lines).
    ambient : number, optional
        Real index of the incidence medium.
    n_orders : int, optional
        Orders -n_orders..n_orders in the RCWA.

    Returns
    -------
    J : complex array, shape (len(wavelengths), 2, 2)
        [[J_pp, J_ps], [J_sp, J_ss]] (out, in) in the Nebraska convention
        of `ellipsometry`: J_pp/J_ss is rho of an unpatterned sample.
    """

    wls = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    th, ph = np.radians(angle), np.radians(azimuth)
    kx, ky = ambient*np.sin(th)*np.cos(ph), ambient*np.sin(th)*np.sin(ph)
    # rows: unit vectors of the tangential p and s directions
    R = np.array([[np.cos(ph), np.sin(ph)], [-np.sin(ph), np.cos(ph)]])
    flip = np.diag([-1.0, 1.0])  # Born & Wolf sign of the reflected p field
    # the tangential part of a p wave is cos(theta) times its amplitude (for
    # the incident and the specular wave alike); s waves are tangential
    D, D_inv = np.diag([np.cos(th), 1.0]), np.diag([1/np.cos(th), 1.0])
    out = np.zeros((len(wls), 2, 2), dtype=complex)
    for i, wl in enumerate(wls):
        res = rcwa(period, layers(wl), wl, k_x=kx, k_y=ky, n_in=ambient,
                   n_out=_index(substrate, wl), n_orders=n_orders)
        i0 = int(np.argmin(np.abs(res.m)))
        J_xy = res.r[0, i0]
        out[i] = np.conj(flip @ D_inv @ R @ J_xy @ R.T @ D)
    return out


def psi_delta(J):
    """Psi and Delta (degrees) of rho = J_pp/J_ss.

    Parameters
    ----------
    J : complex array, shape (..., 2, 2)
        From `zero_order_jones()`.

    Returns
    -------
    psi, delta : arrays
    """

    return psi_delta_from_rho(J[..., 0, 0]/J[..., 1, 1])


def mueller_from_jones(J):
    """Mueller matrices normalized to M_11 = 1 from Jones matrices (p-s
    basis, Nebraska convention); for an isotropic sample this is
    `ellipsometry.sample_mueller()`.

    Parameters
    ----------
    J : complex array, shape (..., 2, 2)

    Returns
    -------
    M : array, shape (..., 4, 4)
    """

    J = np.asarray(J)
    M = np.zeros(J.shape[:-2] + (4, 4))
    for idx in np.ndindex(J.shape[:-2]):
        m = M_from_jones(J[idx])
        M[idx] = m/m[0, 0]
    return M

