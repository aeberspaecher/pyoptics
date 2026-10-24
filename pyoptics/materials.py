#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Simple dispersion models for optical materials.

Glass catalogues characterize a glass by its refractive index n_d at the
helium d line and its Abbe number

    V_d = (n_d - 1)/(n_F - n_C),

with the hydrogen F and C lines. The functions here build a dispersion curve
n(wavelength) from these two numbers using the two-term Cauchy formula

    n(wavelength) = A + B/wavelength**2,

whose coefficients are chosen such that n(lambda_d) = n_d and
n(lambda_F) - n(lambda_C) = (n_d - 1)/V_d hold exactly. This is a fair
approximation in the visible for normally dispersive glasses, but not in the
UV or IR, where Sellmeier data should be used instead.

Wavelengths are in nanometres by default. Other units work as long as the
line wavelengths are passed in the same unit as the evaluation wavelengths.
"""

import numpy as np


# Fraunhofer lines [nm]:
LAMBDA_d = 587.5618  # helium d line
LAMBDA_F = 486.1327  # hydrogen F line
LAMBDA_C = 656.2725  # hydrogen C line


def abbe_number(n_d, n_F, n_C):
    """Abbe number V_d = (n_d - 1)/(n_F - n_C).
    """

    return (n_d - 1.0)/(n_F - n_C)


def cauchy_coeffs_from_abbe(n_d, V_d, lambda_d=LAMBDA_d, lambda_F=LAMBDA_F,
                            lambda_C=LAMBDA_C):
    """Coefficients A, B of n = A + B/wavelength**2 for given n_d and V_d.

    Parameters
    ----------
    n_d : double
        Refractive index at lambda_d.
    V_d : double
        Abbe number, must be positive. np.inf gives a dispersion free medium.
    lambda_d, lambda_F, lambda_C : double, optional
        Wavelengths of the d, F and C lines. Defaults are in nm; pass them in
        another unit to get B in that unit squared.

    Returns
    -------
    A, B : double
        Cauchy coefficients; B has the unit of wavelength squared.
    """

    if not V_d > 0:
        raise ValueError("V_d must be positive")

    # n_F - n_C = B*(1/lambda_F**2 - 1/lambda_C**2) = (n_d - 1)/V_d:
    B = (n_d - 1.0)/V_d/(1.0/lambda_F**2 - 1.0/lambda_C**2)
    A = n_d - B/lambda_d**2

    return A, B


def cauchy_n(wavelength, A, B):
    """Two-term Cauchy formula n = A + B/wavelength**2.
    """

    return A + B/np.asarray(wavelength, dtype=float)**2


def abbe_dispersion(wavelength, n_d, V_d, lambda_d=LAMBDA_d,
                    lambda_F=LAMBDA_F, lambda_C=LAMBDA_C):
    """Refractive index n(wavelength) of a glass given by n_d and V_d.

    See cauchy_coeffs_from_abbe() for the parameters.

    Example
    -------
    >>> n_bk7 = abbe_dispersion([450, 550, 650], 1.5168, 64.17)
    """

    A, B = cauchy_coeffs_from_abbe(n_d, V_d, lambda_d, lambda_F, lambda_C)

    return cauchy_n(wavelength, A, B)
