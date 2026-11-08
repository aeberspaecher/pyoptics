#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Colours of spectra: CIE 1931 XYZ and sRGB.

A spectral power distribution S(lambda) is seen as the tristimulus values ::

    X = int S xbar dlambda,  Y = int S ybar dlambda,  Z = int S zbar dlambda

with the CIE 1931 2 degree colour matching functions xbar, ybar, zbar, here
in the analytic multi-lobe Gaussian approximation of Wyman, Sloan and Shirley
(within the scatter of the tabulated functions). `spectrum_to_rgb()` turns
a spectrum (or a stack of spectra) into sRGB: XYZ to linear sRGB (D65 white
point), clipping of out-of-gamut colours, and the sRGB transfer curve.
`planck()` gives the black body spectrum, e.g. the Sun (about 5800 K).

Spectra are sampled at wavelengths in nm.

References
----------
C. Wyman, P.-P. Sloan and P. Shirley, "Simple analytic approximations to the
CIE XYZ color matching functions", J. Comput. Graph. Tech. 2, 1-11 (2013).

IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour measurement
and management - Part 2-1: Default RGB colour space - sRGB".

CIE 015:2018, "Colorimetry", 4th ed., International Commission on
Illumination (2018).
"""

import numpy as np


__all__ = ["cie_xyz", "planck", "spectrum_to_xyz", "xyz_to_linear_srgb",
           "srgb_gamma", "spectrum_to_rgb"]


# XYZ -> linear sRGB (D65)
M_XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                          [-0.9689, 1.8758, 0.0415],
                          [0.0557, -0.2040, 1.0570]])


def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)

    return np.exp(-0.5*(lam - mu)**2/s**2)


def cie_xyz(wavelength_nm):
    """CIE 1931 colour matching functions xbar, ybar, zbar.

    Parameters
    ----------
    wavelength_nm : array
        Wavelengths in nm.

    Returns
    -------
    cmf : array (3, ...)
        xbar, ybar, zbar (ybar peaks at 1 near 555 nm).
    """

    lam = np.asarray(wavelength_nm, dtype=float)
    xb = (1.056*_g(lam, 599.8, 37.9, 31.0) + 0.362*_g(lam, 442.0, 16.0, 26.7)
          - 0.065*_g(lam, 501.1, 20.4, 26.2))
    yb = 0.821*_g(lam, 568.8, 46.9, 40.5) + 0.286*_g(lam, 530.9, 16.3, 31.1)
    zb = 1.217*_g(lam, 437.0, 11.8, 36.0) + 0.681*_g(lam, 459.0, 26.0, 13.8)

    return np.array([xb, yb, zb])


def planck(wavelength_nm, T):
    """Black body spectral radiance (relative), normalized to 1 at 560 nm.

    Parameters
    ----------
    wavelength_nm : array
        Wavelengths in nm.
    T : number
        Temperature in K.

    Returns
    -------
    S : array
    """

    c2 = 1.438777e7   # h c/k_B in nm K
    lam = np.asarray(wavelength_nm, dtype=float)

    def B(l):
        return l**-5/np.expm1(c2/(l*T))

    return B(lam)/B(560.0)


def spectrum_to_xyz(wavelength_nm, S, axis=-1):
    """Tristimulus values of spectra sampled at wavelength_nm.

    Parameters
    ----------
    wavelength_nm : array (L,)
        Equidistant wavelengths in nm.
    S : array
        Spectra, wavelength along axis.
    axis : int, optional

    Returns
    -------
    XYZ : array
        X, Y, Z along the last axis, normalized such that S = 1 everywhere
        (equal energy white) gives Y = 1.
    """

    cmf = cie_xyz(wavelength_nm)
    S = np.moveaxis(np.asarray(S, dtype=float), axis, -1)

    return S @ cmf.T/np.sum(cmf[1])


def xyz_to_linear_srgb(XYZ):
    """Linear sRGB (not clipped) of XYZ values along the last axis."""

    return np.asarray(XYZ) @ M_XYZ_TO_SRGB.T


def srgb_gamma(rgb):
    """sRGB transfer curve of linear values in [0, 1]."""

    rgb = np.clip(rgb, 0, 1)

    return np.where(rgb <= 0.0031308, 12.92*rgb, 1.055*rgb**(1/2.4) - 0.055)


def spectrum_to_rgb(wavelength_nm, S, gain=1.0, axis=-1, desaturate=False):
    """sRGB colours of spectra.

    Parameters
    ----------
    wavelength_nm : array (L,)
        Equidistant wavelengths in nm.
    S : array
        Spectra, wavelength along axis.
    gain : number, optional
        Brightness factor applied to the linear sRGB values before clipping.
    axis : int, optional
        Wavelength axis of S.
    desaturate : bool, optional
        Bring out-of-gamut colours (a negative linear component) into the
        gamut by adding white (keeping the hue) instead of clipping.

    Returns
    -------
    rgb : array
        sRGB values in [0, 1] along the last axis.
    """

    rgb = gain*xyz_to_linear_srgb(spectrum_to_xyz(wavelength_nm, S, axis))
    if desaturate:
        rgb = rgb - np.minimum(rgb.min(axis=-1, keepdims=True), 0)

    return srgb_gamma(rgb)
