#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
Test the thin film module.
"""

import numpy as np
import pytest

from pyoptics.thin_films import r_t_coeffs, R_T_A_coeffs
from pyoptics.interfaces import R_p, R_s, T_p, T_s, brewster_angle


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


@pytest.mark.parametrize("pol, R_ref, T_ref", [("s", R_s, T_s), ("p", R_p, T_p)])
def test_single_interface(pol, R_ref, T_ref):
    """A layer of embedding medium on a substrate is a single interface, so
    R and T must match the Fresnel equations, also for substrate != embedding
    medium.
    """

    n_0, n_s = 1.0, 1.5
    angles = np.deg2rad(np.linspace(0, 85, 18))
    RTA = np.array([R_T_A_coeffs(angle, [n_0], [0.0], [100.0], n_0, 0.0, n_s, 0.0, pol, 633)
                    for angle in angles])

    assert np.allclose(RTA[:, 0], R_ref(angles, n_0, n_s))
    assert np.allclose(RTA[:, 1], T_ref(angles, n_0, n_s))
    assert np.allclose(RTA[:, 2], 0.0)


def test_incoherent_slab():
    """A thick incoherent slab in air reflects R_slab = 2R/(1 + R) at normal
    incidence (multiple reflections add up in intensity).
    """

    n = 1.5
    R = ((n - 1)/(n + 1))**2
    R_slab, T_slab, A_slab = R_T_A_coeffs(0.0, [n], [0.0], [1E5], 1.0, 0.0, 1.0, 0.0, "s", 633,
                                          incoherent=[True], N_phases=64)

    assert np.isclose(R_slab, 2*R/(1 + R))
    assert np.isclose(T_slab, 1 - 2*R/(1 + R))
    assert np.isclose(A_slab, 0.0)
