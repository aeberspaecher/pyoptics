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


# transfer functions for the angular spectrum:

from pyoptics.fft import FT, inv_FT
from pyoptics.thin_films import stack_transfer_functions
from pyoptics.utils import freq_grid, grid1d


def test_transfer_functions_match_r_t_coeffs():
    wl = 0.633
    n, k, t = [2.3, 1.38, 0.2], [0.0, 0.0, 3.5], [0.1, 0.25, 0.01]
    x = grid1d(-10, 10, 64)
    y = grid1d(-10, 10, 48)
    K_X, K_Y = freq_grid(x, y, normal_order=True)
    k0 = 2*np.pi/wl*1.33

    for pol in ("TE", "TM"):
        r, tt = stack_transfer_functions(x, y, n, k, t, 1.33, 0.0, 1.5, 0.0, pol, wl)
        for (i, j) in [(24, 32), (24, 40), (30, 27), (10, 50)]:
            theta = np.arcsin(np.hypot(K_X[i, j], K_Y[i, j])/k0)
            assert np.allclose((r[i, j], tt[i, j]), r_t_coeffs(theta, n, k, t, 1.33, 0.0, 1.5, 0.0, pol, wl))


def test_single_interface_without_layers():
    x = grid1d(-5, 5, 32)
    r, t = stack_transfer_functions(x, x, [], [], [], 1.0, 0.0, 1.5, 0.0, "TE", 0.5)

    assert np.allclose(r[16, 16], (1 - 1.5)/(1 + 1.5))
    assert np.allclose(t[16, 16], 2/(1 + 1.5))


def test_frustrated_total_internal_reflection():
    """An air gap of thickness d between two glass half spaces transmits
    evanescent waves (TE) like a quantum mechanical barrier:
    T = 1/(1 + (k_z**2 + kappa**2)**2/(4 k_z**2 kappa**2)*sinh(kappa d)**2)."""

    wl, n1, d = 1.0, 1.5, 0.4
    x = grid1d(-20, 20, 256)
    y = grid1d(-1, 1, 2)
    r, t = stack_transfer_functions(x, y, [1.0], [0.0], [d], n1, 0.0, n1, 0.0, "TE", wl)

    k0 = 2*np.pi/wl
    k_x = freq_grid(x, y, normal_order=True)[0][1]  # row Ny//2 = 1 holds k_y = 0
    tir = (np.abs(k_x) > 1.01*k0) & (np.abs(k_x) < 0.99*n1*k0)
    k_z = np.sqrt((n1*k0)**2 - k_x[tir]**2)
    kappa = np.sqrt(k_x[tir]**2 - k0**2)
    T = 1/(1 + (k_z**2 + kappa**2)**2/(4*k_z**2*kappa**2)*np.sinh(kappa*d)**2)

    assert np.count_nonzero(tir) > 10
    assert np.allclose(np.abs(t[1, tir])**2, T)
    assert np.allclose(np.abs(r[1, tir])**2 + np.abs(t[1, tir])**2, 1.0)


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_goos_haenchen_shift(pol):
    """A Gaussian beam totally reflected at glass/air is shifted forward along
    the interface by Artmann's D_s = 2 tan(theta)/(k_1 sqrt(sin^2 theta - n^2)),
    D_p = D_s n^2/((1 + n^2) sin^2 theta - n^2), n = n_2/n_1."""

    wl, n1, n2 = 1.0, 1.5, 1.0
    theta = np.deg2rad(50.0)
    k1 = 2*np.pi*n1/wl
    x = grid1d(-400, 400, 8192)
    y = grid1d(-1, 1, 2)
    X, Y = np.meshgrid(x, y)
    E_in = np.exp(-X**2/40.0**2)*np.exp(1j*k1*np.sin(theta)*X)  # incidence towards +x

    r, _ = stack_transfer_functions(x, y, [], [], [], n1, 0.0, n2, 0.0, pol, wl)
    E_r = inv_FT(r*FT(E_in))

    I_r = np.abs(E_r[0])**2
    shift = np.sum(x*I_r)/np.sum(I_r)

    n = n2/n1
    q = np.sqrt(np.sin(theta)**2 - n**2)
    D = 2*np.tan(theta)/(k1*q)
    if pol == "TM":
        D *= n**2/((1 + n**2)*np.sin(theta)**2 - n**2)

    k_x = freq_grid(x, y)[0][1]
    assert np.allclose(np.abs(r[1, (np.abs(k_x) > n2*2*np.pi/wl) & (np.abs(k_x) < k1)]), 1.0)
    assert shift > 0
    assert shift == pytest.approx(D, rel=1E-2)


def test_thick_evanescent_layers_do_not_overflow():
    """Evanescent and absorbing layers far thicker than the decay length give
    finite results: no transmission, total reflection in the TIR range."""

    wl, n1 = 1.0, 1.5
    x = grid1d(-20, 20, 256)
    y = grid1d(-1, 1, 2)
    for pol in ("TE", "TM"):
        r, t = stack_transfer_functions(x, y, [1.0, 0.2, 1.0], [0.0, 3.0, 0.0], [200.0, 50.0, 200.0],
                                        n1, 0.0, n1, 0.0, pol, wl)
        assert np.all(np.isfinite(r)) and np.all(np.isfinite(t))

        k_x = freq_grid(x, y)[0][1]
        tir = (np.abs(k_x) > 1.01*2*np.pi/wl) & (np.abs(k_x) < 0.99*n1*2*np.pi/wl)
        assert np.allclose(t[1, tir], 0.0) and np.allclose(np.abs(r[1, tir]), 1.0)
