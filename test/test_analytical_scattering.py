#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the exact series solutions for a dielectric cylinder and sphere
(pyoptics.analytical) and compare the WPM with the cylinder solution.

All lengths are given in units of the wavelength (wl = 1).
"""

import numpy as np
import pytest
from scipy.special import j0

from pyoptics.analytical import (cylinder_coefficients, cylinder_field, sphere_coefficients,
                                 sphere_field)
from pyoptics.utils import grid1d


WL = 1.0
K0 = 2*np.pi/WL


def test_no_contrast_gives_incident_field():
    x, z = np.meshgrid(np.linspace(-15, 15, 9), np.linspace(-15, 15, 9))
    k_x = np.array([0.0, 0.4*K0])
    A = np.array([1.0, 0.5j])
    k_z = np.sqrt(K0**2 - k_x**2)
    u_inc = np.exp(1j*(np.multiply.outer(x, k_x) + np.multiply.outer(z, k_z))) @ A
    assert np.allclose(cylinder_field(x, z, 4.0, 1.0, WL, k_x=k_x, amplitudes=A), u_inc, atol=1e-8)

    u_inc = (j0(np.multiply.outer(np.abs(x), k_x))*np.exp(1j*np.multiply.outer(z, k_z))) @ A
    assert np.allclose(sphere_field(x, z, 4.0, 1.0, WL, k_rho=k_x, amplitudes=A), u_inc, atol=1e-8)


@pytest.mark.parametrize("coefficients", [cylinder_coefficients, sphere_coefficients])
@pytest.mark.parametrize("n", [1.33, 1.5, 2.5])
def test_lossless_scattering_is_unitary(coefficients, n):
    # without absorption, every partial wave is scattered with |1 + 2 a| = 1
    _, a, _ = coefficients(3.7, n, WL)
    assert np.allclose(np.abs(1 + 2*a), 1, atol=1e-10)


@pytest.mark.parametrize("field", [cylinder_field, sphere_field])
def test_boundary_conditions(field):
    # u and du/dr are continuous at the surface
    a, n, eps = 3.0, 1.5, 1e-5
    theta = np.linspace(0, np.pi, 7)
    u = lambda r: field(r*np.sin(theta), r*np.cos(theta), a, n, WL)
    assert np.allclose(u(a - eps), u(a + eps), atol=1e-3)
    du_in = (u(a - eps) - u(a - 3*eps))/(2*eps)
    du_out = (u(a + 3*eps) - u(a + eps))/(2*eps)
    assert np.allclose(du_in, du_out, rtol=1e-3, atol=1e-2*np.abs(du_out).max())


def test_helmholtz_equation_inside_and_outside():
    # finite difference Laplacian: nabla^2 u + k0^2 n^2 u = 0 on both sides
    a, n, h = 3.0, 1.5, 1e-3
    for x0, z0, n_loc in ((0.7, 1.1, n), (2.0, 3.5, 1.0)):
        u = lambda dx, dz: cylinder_field(x0 + dx, z0 + dz, a, n, WL)
        lap = (u(h, 0) + u(-h, 0) + u(0, h) + u(0, -h) - 4*u(0, 0))/h**2
        assert abs(lap + (K0*n_loc)**2*u(0, 0)) < 1e-3*K0**2*abs(u(0, 0)) + 1e-2


def test_wpm_focus_of_cylinder():
    # ball lens in 2D (notebook 05): the WPM finds the focus of the exact solution
    wpm = pytest.importorskip("pyoptics.wpm")
    a, n, w = 10.0, 1.5, 15.0
    L, N = 96.0, 512
    x = grid1d(-L/2, L/2, N)
    k_x = 2*np.pi*np.fft.fftfreq(N, x[1] - x[0])
    A = np.fft.fft(np.fft.ifftshift(np.exp(-(x/w)**6)))/N
    prop = np.abs(k_x) < K0
    z0 = -a - 2
    u0 = np.exp(1j*(np.outer(x, k_x[prop]) + z0*np.sqrt(K0**2 - k_x[prop]**2))) @ A[prop]
    z_vals = np.arange(z0, 25 + 1e-9, 0.1)
    U = wpm.wpm_propagate_z(lambda X, Y, z: np.where(X**2 + z**2 < a**2, n, 1.0), u0, x, None, z_vals, WL)
    behind = z_vals > a
    I_wpm = np.abs(U[behind, N//2])**2
    I_exact = np.abs(cylinder_field(0*z_vals[behind], z_vals[behind], a, n, WL, k_x=k_x[prop],
                                    amplitudes=A[prop]))**2
    assert abs(z_vals[behind][np.argmax(I_wpm)] - z_vals[behind][np.argmax(I_exact)]) <= 0.3
    assert abs(I_wpm.max()/I_exact.max() - 1) < 0.05
