#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Analytical solutions for testing.
"""

import numpy as np


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
        Radius of curvature of wavefront in aperture. Defaults to None, which
        is interpreted as the infite curvature of a collimated beam.

    Returns
    -------
    I_z : number
        On axis intensity at distance z from aperture.

    Note
    ----
    Mahajan's results are obtained using scalar Rayleigh-Sommerfel
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
        I_z = I0*(cos_alpha1**2 + cos_alpha2**2 - 2*cos_alpha1*cos_alpha2*np.cos(2*np.pi*(s1_prime - s2_prime)/wl))
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

