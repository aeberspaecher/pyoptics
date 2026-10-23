#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test Müller calculus against Jones calculus.
"""

import numpy as np
import pytest

from pyoptics.interfaces import r_p, r_s, t_p, t_s
from pyoptics.polarization.mueller import (M_from_jones, M_linear_polarizer,
                                           M_reflection_dielectric,
                                           M_reflection_metal,
                                           M_retarder, M_TIR,
                                           M_transmission_dielectric,
                                           stokes_from_complex_E)


def _rot(phi):
    return np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])


def _element_at_angle(J, phi):
    """Rotate Jones matrix J of an element by phi."""

    return _rot(phi).T @ J @ _rot(phi)


E_TEST = [np.array([1.0, 0.0]), np.array([1.0, 1j])/np.sqrt(2),
          np.array([0.3 + 0.5j, -0.7 + 0.1j])]


@pytest.mark.parametrize("J", [_rot(0.3),
                               _element_at_angle(np.diag([1, np.exp(0.9j)]), 0.4),
                               _element_at_angle(np.diag([1.0, 0.0]), 1.1),
                               np.array([[0.2 + 0.1j, 0.5], [-0.3j, 0.8]])])
def test_M_from_jones(J):
    M = M_from_jones(J)

    assert np.isrealobj(M)
    for E in E_TEST:
        assert np.allclose(M @ stokes_from_complex_E(E),
                           stokes_from_complex_E(J @ E))


def test_stokes_is_real():
    assert np.isrealobj(stokes_from_complex_E(E_TEST[-1]))


@pytest.mark.parametrize("phi", [0.0, 0.4, 1.3])
def test_linear_polarizer(phi):
    J = _element_at_angle(np.diag([1.0, 0.0]), phi)
    assert np.allclose(M_linear_polarizer(phi), M_from_jones(J))


@pytest.mark.parametrize("phi, delta", [(0.0, 0.9), (0.4, 0.9), (1.1, np.pi/2)])
def test_retarder(phi, delta):
    J = _element_at_angle(np.diag([1.0, np.exp(1j*delta)]), phi)
    assert np.allclose(M_retarder(phi, delta), M_from_jones(J))


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.2])
def test_reflection_dielectric(theta):
    rs, rp = r_s(theta, 1.0, 1.5), r_p(theta, 1.0, 1.5)
    # x-axis is p, y-axis is s:
    J = np.diag([rp, rs])

    assert np.allclose(M_reflection_dielectric(rs, rp), M_from_jones(J))


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.2])
def test_transmission_dielectric(theta):
    n1, n2 = 1.0, 1.5
    ts, tp = t_s(theta, n1, n2), t_p(theta, n1, n2)
    M = M_transmission_dielectric(ts, tp, theta, n1, n2)
    M_R = M_reflection_dielectric(r_s(theta, n1, n2), r_p(theta, n1, n2))

    # energy conservation for unpolarized and polarized light:
    for E in E_TEST:
        S = stokes_from_complex_E(E)
        assert np.isclose((M @ S)[0] + (M_R @ S)[0], S[0])


def test_TIR():
    n, theta = 1.5, 1.2
    # reference: retardance from the phases of the TIR reflection coefficients
    rs, rp = r_s(theta, n, 1.0), r_p(theta, n, 1.0)
    delta_ref = np.angle(rs) - np.angle(rp)

    M = M_TIR(theta, n)
    delta = np.arctan2(M[3, 2], M[2, 2])

    assert np.isclose(np.abs(rs), 1.0) and np.isclose(np.abs(rp), 1.0)
    assert np.isclose(delta, delta_ref)
    assert delta > 0  # exp(-i omega t) convention
    # x-axis is p, y-axis is s:
    assert np.allclose(M, M_from_jones(np.diag([rp, rs])))


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.2])
def test_reflection_complex_coefficients(theta):
    """Complex r_s, r_p (here from an absorbing medium) must give the same
    Müller matrix as the Jones matrix diag(r_p, r_s), including the sign of
    the S_2 <-> S_3 coupling.
    """

    n_metal = 0.2 + 3j
    rs, rp = r_s(theta, 1.0, n_metal), r_p(theta, 1.0, n_metal)
    M_J = M_from_jones(np.diag([rp, rs]))

    assert np.allclose(M_reflection_dielectric(rs, rp), M_J)
    assert np.allclose(M_reflection_metal(theta, n_metal), M_J)


def test_transmission_complex_coefficients():
    ts, tp = 0.8*np.exp(0.3j), 0.6*np.exp(-0.5j)
    M = M_transmission_dielectric(ts, tp, 0.0, 1.0, 1.0)

    assert np.allclose(M, M_from_jones(np.diag([tp, ts])))


def test_jones_module_conventions():
    from pyoptics.polarization import jones
    from pyoptics.polarization.mueller import S_LCP, S_RCP

    assert np.allclose(stokes_from_complex_E(jones.J_RHCP), S_RCP)
    assert np.allclose(stokes_from_complex_E(jones.J_LHCP), S_LCP)
    # QWP at 45° turns horizontal into circular, HWP at 22.5° into +45°:
    assert np.isclose(abs(stokes_from_complex_E(jones.J_WP(np.pi/2, np.pi/4) @ jones.J_H)[3]), 1.0)
    assert np.allclose(jones.J_WP(np.pi, np.pi/8) @ jones.J_H, jones.J_Lp45)
    for phi in (0.0, 0.4, 1.3):
        assert np.allclose(M_from_jones(jones.J_rot_polarizer(phi)), M_linear_polarizer(phi))
        assert np.allclose(M_from_jones(jones.J_WP(0.7, phi)), M_retarder(phi, 0.7))
