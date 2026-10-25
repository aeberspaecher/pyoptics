#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Analytical solutions for testing.
"""

import numpy as np
from scipy.special import j0, j1


def mahajan_on_axis_intensity(a, epsilon, I0, wavelength, z, R=None):
    """On-axis intensity for light diffracted by an annular aperture.

    Parameters
    ----------
    a, epsilon : numbers
        Outer and inner anulus radii defined by a and epsilon*a respecticvely.
    I0 :
    wavelength : number
    z : number
        Axial distance from apeture.
    R : number, optional
        Radius defining curvature 1/R of wavefront in aperture. Defaults to
        None, which is interpreted as the infite curvature of a collimated
        beam. R is the distance of geometric focus from the center of the
        aperture.

    Returns
    -------
    I_z : number
        On axis intensity at distance z from aperture.

    Notes
    -----
    Mahajan's results are obtained using scalar Rayleigh-Sommerfeld
    diffraction theory.

    TODO: Mahajan paper citation
    """

    s1_prime = np.sqrt(z**2 + epsilon**2*a**2)
    s2_prime = np.sqrt(z**2 + a**2)

    cos_alpha1 = z/s1_prime
    cos_alpha2 = z/s2_prime

    if R is None:  # collimated beam
        #I_z = 4*I0*np.sin(np.pi*a**2*(1.-epsilon**2)/(2.*wavelength*z))**2  # far field expression
        # TODO: test switching to far-field expression - advantages?
        s1_prime = np.sqrt(a**2 + z**2)
        s2_prime = np.sqrt((epsilon*a)**2 + z**2)
        I_z = I0*(cos_alpha1**2 + cos_alpha2**2 - 2*cos_alpha1*cos_alpha2*np.cos(2*np.pi*(s1_prime - s2_prime)/wavelength))
    else:  # focussed beam
        A0 = np.sqrt(I0*R**2)
        s1 = np.sqrt(R**2 + epsilon**2*a**2)
        s2 = np.sqrt(R**2 + a**2)
        d1 = s1_prime - s1
        d2 = s2_prime - s2
        I_z = A0**2*((cos_alpha1/d1)**2 + (cos_alpha2/d2)**2
                     - 2*cos_alpha1*cos_alpha2/(d1*d2)*np.cos(2*np.pi*(d1-d2)/wavelength)
                    )

    return I_z



def airy_pattern(r, NA, wavelength):
    """Intensity PSF of an aberration-free imaging system with a circular
    pupil (Airy pattern) in scalar paraxial theory.

    Parameters
    ----------
    r : number or array
        Distance from the optical axis in the image plane.
    NA : number
        Image-side numerical aperture.
    wavelength : number
        Wavelength in the image space medium (vacuum wavelength if NA already
        includes the refractive index).

    Returns
    -------
    I : number or array
        (2*J_1(v)/v)**2 with v = 2*pi*NA*r/wavelength, normalized to I(0) = 1.
        The first zero lies at r = 0.61*wavelength/NA.
    """

    v = 2*np.pi*NA*np.asarray(r, dtype=float)/wavelength
    v_safe = np.where(v == 0, 1.0, v)
    I = np.where(v == 0, 1.0, (2*j1(v_safe)/v_safe)**2)

    return I


def airy_encircled_energy(r, NA, wavelength):
    """Fraction of the energy of the Airy pattern inside a circle of radius r
    (Rayleigh's formula 1 - J_0(v)**2 - J_1(v)**2, v = 2*pi*NA*r/wavelength).
    """

    v = 2*np.pi*NA*np.asarray(r, dtype=float)/wavelength

    return 1.0 - j0(v)**2 - j1(v)**2


def circular_otf(f, NA, wavelength):
    """Diffraction limited optical transfer function (OTF) of an incoherent
    imaging system with a circular pupil.

    Parameters
    ----------
    f : number or array
        Spatial frequency (in 1/length, i.e. cycles per unit length, not
        angular wavenumber).
    NA : number
        Image-side numerical aperture.
    wavelength : number
        Wavelength.

    Returns
    -------
    OTF : number or array
        2/pi*(arccos(s) - s*sqrt(1 - s**2)) with s = ``|f|/f_c`` and the cut-off
        frequency f_c = 2*NA/wavelength; zero for s > 1. The OTF is real and
        non-negative, so it is identical to the MTF.
    """

    s = np.abs(np.asarray(f, dtype=float))/(2*NA/wavelength)
    s_clipped = np.clip(s, 0.0, 1.0)
    otf = 2/np.pi*(np.arccos(s_clipped) - s_clipped*np.sqrt(1 - s_clipped**2))

    return np.where(s <= 1.0, otf, 0.0)
