#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
Test the thin film module.
"""

import numpy as np

from pyoptics.thin_films import r_t_coeffs, R_T_A_coeffs
from pyoptics.utils import brewster_angle


def test_brewster():
    """
    Mimic a single dielectric interface using the thin_films module and check
    for Brewster's angle.
    """

    n1, n2 = 1.5, 1.0
    n, k = [n1], [0]
    d = [1]
    wl = 633
    theta_B = brewster_angle(n1, n2)
    # at Brewster's angle, there is no reflection in p polarization:
    r, t = r_t_coeffs(theta_B, n, k, d, n1, 0.0, n2, 0.0, "p", wl)

    assert np.isclose(np.abs(r), 0.0) 


def test_energy_conservation():
    """
    Test R + T = 1 in non-absorbing media (should be independent of T
    convention [with cosine factor or without] if embedding medium is equal to
    substrate).
    """

    n, k = [1.5], [0]
    d = [1]
    wl = 633
    angles = np.deg2rad(np.linspace(0, 89, 100))
    E = lambda R, T, _: R + T
    E_p = [E(*R_T_A_coeffs(angle, n, k, d, 1.0, 0.0, 1.0, 0.0, "p", wl)) for
           angle in angles]

    E_s = [E(*R_T_A_coeffs(angle, n, k, d, 1.0, 0.0, 1.0, 0.0, "s", wl)) for
           angle in angles]
    assert all(np.isclose(E_p, 1.0))
    assert all(np.isclose(E_s, 1.0))


def test_TIR_phase():
    r, _ = r_t_coeffs(np.deg2rad(90-0.00001), [1.5], [0.0], [1.0], 3.3, 0.0,
                      1.0, 0.0, "s", 100)
    assert np.isclose(np.abs(r), 1.0)
    assert np.isclose(np.angle(r), -np.pi)  # TODO: document the -pi


# TODO: test filter? Fabry Perot?
