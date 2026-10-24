#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the Abbe/Cauchy dispersion model.
"""

import numpy as np
import pytest

from pyoptics.materials import (LAMBDA_C, LAMBDA_F, LAMBDA_d, abbe_dispersion,
                                abbe_number, cauchy_coeffs_from_abbe, cauchy_n)


@pytest.mark.parametrize("n_d, V_d", [(1.5168, 64.17), (1.7174, 29.51),
                                      (1.4585, 67.82)])
def test_reproduces_n_d_and_V_d(n_d, V_d):
    n_d_model, n_F, n_C = abbe_dispersion([LAMBDA_d, LAMBDA_F, LAMBDA_C], n_d, V_d)

    assert np.isclose(n_d_model, n_d)
    assert np.isclose(abbe_number(n_d_model, n_F, n_C), V_d)
    assert n_F > n_d_model > n_C  # normal dispersion


def test_bk7_against_sellmeier():
    """N-BK7 (Schott Sellmeier coefficients, wavelength in um) agrees with the
    two-term Cauchy model to a few 1e-4 in the visible.
    """

    B = (1.03961212, 0.231792344, 1.01046945)
    C = (0.00600069867, 0.0200179144, 103.560653)

    def n_sellmeier(wl_um):
        wl2 = wl_um**2
        return np.sqrt(1 + sum(b*wl2/(wl2 - c) for b, c in zip(B, C)))

    wl = np.linspace(420, 700, 30)
    n_model = abbe_dispersion(wl, 1.5168, 64.17)

    assert np.max(np.abs(n_model - n_sellmeier(wl/1000))) < 5e-4


def test_units():
    """Line wavelengths in um give the same curve evaluated in um."""

    n_nm = abbe_dispersion(550.0, 1.6, 40.0)
    n_um = abbe_dispersion(0.55, 1.6, 40.0, LAMBDA_d/1000, LAMBDA_F/1000,
                           LAMBDA_C/1000)

    assert np.isclose(n_nm, n_um)


def test_no_dispersion():
    A, B = cauchy_coeffs_from_abbe(1.5, np.inf)

    assert B == 0
    assert np.allclose(cauchy_n([400, 800], A, B), 1.5)


@pytest.mark.parametrize("V_d", [0.0, -10.0])
def test_invalid_V_d(V_d):
    with pytest.raises(ValueError):
        cauchy_coeffs_from_abbe(1.5, V_d)
