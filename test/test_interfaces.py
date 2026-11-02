#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test Fresnel coefficients of a single dielectric interface.
"""

import numpy as np
import pytest

from pyoptics.interfaces import (R_p, R_s, T_p, T_s, brewster_angle,
                                 critical_angle, r_p, r_s, refracted_angle,
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


@pytest.mark.parametrize("n_1, n_2", [(1.0, 1.5), (1.5, 1.0), (1.33, 2.4)])
def test_born_wolf_angle_forms(n_1, n_2):
    """Coefficients in the forms with phi_1 +- phi_2 (Hecht eqs. 4.42 - 4.45),
    below the critical angle.
    """

    phi_1 = np.deg2rad(np.linspace(1, 80, 50))
    phi_1 = phi_1[n_1*np.sin(phi_1) < n_2]
    phi_2 = np.arcsin(n_1/n_2*np.sin(phi_1))
    s, d = phi_1 + phi_2, phi_1 - phi_2

    assert np.allclose(r_s(phi_1, n_1, n_2), -np.sin(d)/np.sin(s))
    assert np.allclose(r_p(phi_1, n_1, n_2), np.tan(d)/np.tan(s))
    assert np.allclose(t_s(phi_1, n_1, n_2), 2*np.sin(phi_2)*np.cos(phi_1)/np.sin(s))
    assert np.allclose(t_p(phi_1, n_1, n_2),
                       2*np.sin(phi_2)*np.cos(phi_1)/(np.sin(s)*np.cos(d)))


def test_amplitude_relations():
    """t_s = 1 + r_s and n_2 t_p = n_1 (1 + r_p), also for absorbing media
    and under total internal reflection.
    """

    for n_1, n_2 in [(1.0, 1.5), (1.5, 1.0), (1.0, 0.2 + 3.0j), (1.5, 1.2 + 0.1j)]:
        assert np.allclose(t_s(ANGLES, n_1, n_2), 1 + r_s(ANGLES, n_1, n_2))
        assert np.allclose(n_2*t_p(ANGLES, n_1, n_2), n_1*(1 + r_p(ANGLES, n_1, n_2)))


def test_consistent_with_refracted_angle():
    """Coefficients agree with the textbook forms using cos(refracted_angle())."""

    for n_1, n_2 in [(1.0, 1.5), (1.5, 1.0), (1.0, 0.2 + 3.0j)]:
        c_1, c_2 = np.cos(ANGLES), np.cos(refracted_angle(ANGLES, n_1, n_2))
        assert np.allclose(r_s(ANGLES, n_1, n_2),
                           (n_1*c_1 - n_2*c_2)/(n_1*c_1 + n_2*c_2))
        assert np.allclose(r_p(ANGLES, n_1, n_2),
                           (n_2*c_1 - n_1*c_2)/(n_2*c_1 + n_1*c_2))
        assert np.allclose(t_p(ANGLES, n_1, n_2), 2*n_1*c_1/(n_2*c_1 + n_1*c_2))


def test_snell():
    phi_2 = refracted_angle(ANGLES, N1, N2)

    assert np.isrealobj(phi_2)
    assert np.allclose(N1*np.sin(ANGLES), N2*np.sin(phi_2))


def test_total_internal_reflection():
    n_1, n_2 = 1.5, 1.0
    phi = np.deg2rad(np.linspace(42, 89, 30))  # critical angle 41.8 deg

    for r, T in [(r_s, T_s), (r_p, T_p)]:
        assert np.allclose(np.abs(r(phi, n_1, n_2)), 1.0)
        assert np.allclose(T(phi, n_1, n_2), 0.0)
    # TIR phase difference (Fresnel rhomb): tan(delta/2) = cos sqrt(sin**2 - n**2)/sin**2
    n = n_2/n_1
    delta = np.angle(r_s(phi, n_1, n_2)/r_p(phi, n_1, n_2))
    ref = 2*np.arctan(np.cos(phi)*np.sqrt(np.sin(phi)**2 - n**2)/np.sin(phi)**2)
    assert np.allclose(np.abs(delta), ref)


def test_real_coefficients_without_tir():
    """Lossless media below the critical angle give real coefficients."""

    for f in [r_s, r_p, t_s, t_p]:
        assert np.isrealobj(f(ANGLES, N1, N2))
        assert np.isrealobj(f(0.3, N1, N2))


def test_equal_media():
    for n in [1.5, 1.2 + 0.3j]:
        assert np.allclose(r_s(ANGLES, n, n), 0.0)
        assert np.allclose(r_p(ANGLES, n, n), 0.0)
        assert np.allclose(t_s(ANGLES, n, n), 1.0)
        assert np.allclose(t_p(ANGLES, n, n), 1.0)


def test_grazing_incidence():
    assert np.isclose(r_s(np.pi/2, N1, N2), -1.0)
    assert np.isclose(r_p(np.pi/2, N1, N2), -1.0)


def test_metal_normal_incidence():
    n_m = 0.2 + 3.4j  # roughly silver in the visible
    R_ref = np.abs((1 - n_m)/(1 + n_m))**2

    assert np.isclose(R_s(0.0, 1.0, n_m), R_ref)
    assert np.isclose(R_p(0.0, 1.0, n_m), R_ref)


@pytest.mark.parametrize("R, T", [(R_s, T_s), (R_p, T_p)])
@pytest.mark.parametrize("n_2", [0.2 + 3.4j, 1.5 + 0.05j, 0.05 + 0.5j])
def test_energy_conservation_absorbing_substrate(R, T, n_2):
    """R + T = 1 with T the power entering an absorbing medium."""

    R_, T_ = R(ANGLES, 1.0, n_2), T(ANGLES, 1.0, n_2)

    assert np.all(T_ >= 0)
    assert np.allclose(R_ + T_, 1.0)


def test_broadcasting():
    """Angles broadcast against arrays of refractive indices."""

    n_2 = np.array([1.3, 1.5, 1.7])[:, np.newaxis]
    phi = ANGLES[np.newaxis, :]

    r = r_p(phi, N1, n_2)
    assert r.shape == (3, ANGLES.size)
    assert np.allclose(r[1], r_p(ANGLES, N1, 1.5))
    assert T_s(phi, N1, n_2).shape == (3, ANGLES.size)


def test_scalar_and_list_inputs():
    assert np.ndim(r_s(0.3, N1, N2)) == 0
    assert np.ndim(T_p(0.3, N1, N2)) == 0
    assert np.ndim(refracted_angle(0.3, N1, N2)) == 0
    assert np.allclose(R_s([0.0, 0.3], N1, N2), R_s(np.array([0.0, 0.3]), N1, N2))


def test_brewster_angle_values():
    assert np.isclose(brewster_angle(1.0, 1.5), np.arctan(1.5))
    # Brewster angles for both directions add to 90 degrees:
    assert np.isclose(brewster_angle(1.0, 1.5) + brewster_angle(1.5, 1.0), np.pi/2)
    # phi_B + phi_2 = 90 degrees:
    phi_B = brewster_angle(1.0, 1.5)
    assert np.isclose(phi_B + refracted_angle(phi_B, 1.0, 1.5), np.pi/2)


def test_critical_angle():
    phi_c = critical_angle(1.5, 1.0)

    assert np.isclose(phi_c, np.arcsin(1/1.5))
    assert np.isclose(R_s(phi_c + 1e-9, 1.5, 1.0), 1.0)
    assert R_s(phi_c - 1e-3, 1.5, 1.0) < 1.0
    assert np.isnan(critical_angle(1.0, 1.5))
    assert np.isclose(critical_angle(1.5, 1.5), np.pi/2)
    assert np.allclose(critical_angle(np.array([1.5, 1.0]), 1.2),
                       [np.arcsin(1.2/1.5), np.nan], equal_nan=True)
