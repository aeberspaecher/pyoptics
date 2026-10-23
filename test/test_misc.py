#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for smaller helpers across pyoptics: masks, geometry, beams,
processing, interpolation, TIE and utils.
"""

import importlib

import numpy as np
import pytest

from pyoptics.utils import grid1d, grid2d


MODULES = ["analytical", "bases", "beams", "fft", "geometry", "ifta", "imaging",
           "interfaces", "interpolator", "masks", "plot_tools", "processing",
           "propagators", "thin_films", "TIE", "utils", "polarization.jones",
           "polarization.mueller", "polarization._polarization"]


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module("pyoptics." + module)


def test_elliptical_mask():
    from pyoptics.masks import circular_mask, elliptical_mask

    x = grid1d(-2, 2, 64)
    assert np.array_equal(elliptical_mask(x, x, 1.0, 1.0), circular_mask(x, x, 1.0))
    mask = elliptical_mask(x, x, 1.5, 0.5)
    assert mask[32, 32] == 1 and mask[32, 32 + 20] == 1 and mask[32 + 20, 32] == 0


def test_fresnel_number():
    from pyoptics.propagators import fresnel_number, z_from_fresnel_number

    D, z, wl = 2.0, 10.0, 0.001
    F = fresnel_number(D, z, wl)
    assert np.isclose(F, 1.0**2/(wl*z))
    assert np.isclose(z_from_fresnel_number(F, D, wl), z)
    # annulus: area (and number of zones) shrinks by 1 - eps**2:
    assert np.isclose(fresnel_number(D, z, wl, inner_aperture_diameter_factor=0.5), 0.75*F)
    # in the geometric focus there are no zones:
    assert np.isclose(fresnel_number(D, z, wl, wavefront_curvature=1/z), 0.0)


@pytest.mark.parametrize("a, b", [([2., 0, 0], [0, 3., 0]),
                                  ([1., 2, 3], [-1., 0.5, 2]),
                                  ([1., 1, 0], [2., 2, 0]),
                                  ([1., 2, 3], [-2., -4, -6])])
def test_R_onto(a, b):
    from pyoptics.geometry import R_onto

    a, b = np.array(a), np.array(b)
    a_orig = a.copy()
    R = R_onto(a, b)

    assert np.array_equal(a, a_orig)  # input not modified
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ (a/np.linalg.norm(a)), b/np.linalg.norm(b))


def test_spherical_wave():
    from pyoptics.beams import spherical_wave

    x = grid1d(-1, 1, 33, assume_periodicity=False)  # x[16] = 0
    wl = 0.5
    for z in (3.0, 6.0):
        U = spherical_wave(x, x, z, wl, 0.0, 0.0, 0.0)
        assert np.isclose(np.abs(U[16, 16]), 1/z)


def test_z_r_from_divergence_angle():
    from pyoptics.beams import z_r_from_divergence_angle

    w0, wl, n = 2.0, 0.5, 1.5
    theta = wl/(np.pi*n*w0)  # divergence angle in the medium
    assert np.isclose(z_r_from_divergence_angle(w0, theta, n, wl), np.pi*w0**2*n/wl)


def test_sgn():
    from pyoptics.utils import sgn

    assert sgn(0.0) == 1.0
    assert sgn(-1E-300) == -1.0
    assert sgn(2.5) == 1.0
    assert np.array_equal(sgn(np.array([-2.0, 0.0, 3.0])), [-1.0, 1.0, 1.0])


def test_airy_diameter():
    from pyoptics.utils import get_length_scales

    airy_1, _ = get_length_scales(0.5, 0.9, n=1.0)
    airy_2, _ = get_length_scales(0.5, 0.9, n=1.5)
    assert np.isclose(airy_1, 1.22*0.5/0.9)
    assert np.isclose(airy_2, airy_1)  # NA already contains n


def test_get_z_derivative():
    from pyoptics.utils import get_z_derivative

    dz = 0.1
    z = dz*np.arange(-2, 3)
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((2, 8, 6))
    stack = A[:, :, None] + B[:, :, None]*z[None, None, :]  # I(z) = A + B z

    dIdz = get_z_derivative(stack, dz, 2)
    assert dIdz.shape == (8, 6)
    assert np.allclose(dIdz, B)


def test_shifter():
    from pyoptics.processing import Shifter

    x = grid1d(-1, 1, 32)
    y = grid1d(-1, 1, 24)
    rng = np.random.default_rng(2)
    img = rng.standard_normal((24, 32))
    dx, dy = x[1] - x[0], y[1] - y[0]

    shifter = Shifter(3*dx, -2*dy, x, y)
    shifted = shifter(img)
    assert np.isrealobj(shifted)
    assert np.allclose(shifted, np.roll(img, (-2, 3), axis=(0, 1)))

    stack = np.dstack([img, 2*img])
    shifted_stack = shifter(stack)
    assert np.allclose(shifted_stack[:, :, 1], 2*shifted)


def test_trig_interpolator_cutoff():
    from pyoptics.interpolator import TrigInterpolator

    x = grid1d(-1, 1, 32)
    img = np.ones((32, 32))
    interp = TrigInterpolator(img, x, x, f_cutoff=3.0)
    assert np.all(np.abs(interp.F_X) <= 3.0) and np.all(np.abs(interp.F_Y) <= 3.0)
    interp.f_cutoff = 5.0
    assert interp.f_cutoff == 5.0
    assert len(interp.F_X) > 0


def test_TIE_paraxial_propagation():
    from pyoptics.TIE import field_prop_paraxial
    from pyoptics.propagators import fresnel_TF

    x, y, X, Y = grid2d(-20, 20, -20, 20, 64)
    field = np.exp(-(X**2 + Y**2)/3.0**2)
    U_tie = field_prop_paraxial(field, x, y, 30.0, 1.0)
    U_ref, _, _ = fresnel_TF(field, x, y, 30.0, 1.0)

    assert np.allclose(U_tie, U_ref)


def test_TIE_inv_laplacian():
    from pyoptics.TIE import inv_laplacian, laplacian

    x, y, X, Y = grid2d(-np.pi, np.pi, -np.pi, np.pi, 64)
    f = np.cos(2*X)*np.sin(Y)  # mean free
    f_inv = inv_laplacian(f, x, y)

    assert np.all(np.isfinite(f_inv))
    assert np.allclose(f_inv, -f/5)
    assert np.allclose(laplacian(f_inv, x, y), f)


def test_import_without_optional_dependencies():
    """matplotlib and sympy are optional: the computational modules must
    import without them.
    """

    import subprocess
    import sys

    code = ("import sys; sys.modules['matplotlib'] = None; sys.modules['sympy'] = None; "
            "import pyoptics, pyoptics.masks, pyoptics.ifta, pyoptics.TIE, "
            "pyoptics.polarization.mueller")
    subprocess.run([sys.executable, "-c", code], check=True)
