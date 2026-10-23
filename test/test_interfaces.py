#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test Fresnel coefficients of a single dielectric interface.
"""

import numpy as np
import pytest

from pyoptics.interfaces import (R_p, R_s, T_p, T_s, brewster_angle, r_p, r_s,
                                 t_p, t_s)


N1, N2 = 1.0, 1.5
ANGLES = np.deg2rad(np.linspace(0, 89, 90))


def test_normal_incidence():
    assert np.isclose(r_s(0.0, N1, N2), (N1 - N2)/(N1 + N2))
    assert np.isclose(r_p(0.0, N1, N2), (N2 - N1)/(N1 + N2))
    assert np.isclose(t_s(0.0, N1, N2), 2*N1/(N1 + N2))
    assert np.isclose(t_p(0.0, N1, N2), 2*N1/(N1 + N2))


@pytest.mark.parametrize("n_1, n_2", [(1.0, 1.5), (1.5, 1.0)])
def test_brewster(n_1, n_2):
    theta_B = brewster_angle(n_1, n_2)

    assert np.isclose(R_p(theta_B, n_1, n_2), 0.0)
    assert R_s(theta_B, n_1, n_2) > 0.01


@pytest.mark.parametrize("R, T", [(R_s, T_s), (R_p, T_p)])
def test_energy_conservation(R, T):
    assert np.allclose(R(ANGLES, N1, N2) + T(ANGLES, N1, N2), 1.0)


@pytest.mark.parametrize("t, T", [(t_s, T_s), (t_p, T_p)])
def test_T_from_t(t, T):
    """T = n_2 cos(theta_2)/(n_1 cos(theta_1)) |t|**2 for field amplitude
    coefficients.
    """

    cos_2 = np.sqrt(1 - (N1/N2*np.sin(ANGLES))**2)
    T_ref = N2*cos_2/(N1*np.cos(ANGLES))*np.abs(t(ANGLES, N1, N2))**2

    assert np.allclose(T(ANGLES, N1, N2), T_ref)


def test_stokes_relations():
    """Stokes relations r' = -r and t*t' = 1 - r**2."""

    for r, t in [(r_s, t_s), (r_p, t_p)]:
        theta_2 = np.arcsin(N1/N2*np.sin(ANGLES))
        assert np.allclose(r(theta_2, N2, N1), -r(ANGLES, N1, N2))
        assert np.allclose(t(ANGLES, N1, N2)*t(theta_2, N2, N1),
                           1 - r(ANGLES, N1, N2)**2)
