#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check vectorial imaging (pyoptics.vectorial_imaging) and the polarization
helpers it uses.

References: two-beam interference of TE and TM waves (contrast 1 and
|cos(2 theta)|), the scalar model at low NA, the FFT based 2D imaging against
the direct sum over diffraction orders, Stokes/Jones round trips.
"""

import numpy as np
import pytest

from pyoptics.polarization import (DOP, IPS, TE_TM_basis, TE_TM_pol_to_lin,
                                   jones_from_stokes, lin_pol_to_TE_TM)
from pyoptics.polarization.mueller import (S_LHP, S_RCP, S_unpol,
                                           stokes_from_complex_E)
from pyoptics.utils import grid1d
from pyoptics.vectorial_imaging import (contrast, image_1d, image_2d,
                                        lines_spaces_orders, nils,
                                        source_points, two_beam_contrast)


WL = 193.0
N_WATER = 1.437


# polarization helpers:

def test_TE_TM_basis_and_round_trip():
    rng = np.random.default_rng(0)
    s_x, s_y = rng.normal(size=10), rng.normal(size=10)
    e_TE, e_TM = TE_TM_basis(s_x, s_y)

    assert np.allclose(np.sum(e_TE*e_TM, axis=0), 0)
    assert np.allclose(e_TE[0]*s_x + e_TE[1]*s_y, 0)  # TE is perpendicular to the plane of incidence
    E = rng.normal(size=(2, 10)) + 1j*rng.normal(size=(2, 10))
    assert np.allclose(TE_TM_pol_to_lin(lin_pol_to_TE_TM(E, s_x, s_y), s_x, s_y), E)

    field = np.moveaxis(E.reshape(2, 2, 5), 0, -1)  # (2, 5, 2) stack
    assert np.allclose(lin_pol_to_TE_TM(field, s_x.reshape(2, 5), s_y.reshape(2, 5)),
                       np.moveaxis(lin_pol_to_TE_TM(E, s_x, s_y).reshape(2, 2, 5), 0, -1))


def test_TE_TM_in_the_x_z_plane():
    E = lin_pol_to_TE_TM(np.array([1.0, 2.0]), 0.5, 0.0)

    assert np.allclose(E, [2.0, 1.0])  # TE = y, TM = x


@pytest.mark.parametrize("S", [S_unpol, S_LHP, S_RCP, np.array([1.0, 0.3, -0.2, 0.5]),
                               np.array([2.0, 0.0, 0.0, 0.0])])
def test_jones_from_stokes_round_trip(S):
    w, J = jones_from_stokes(S)

    assert np.isclose(w.sum(), S[0])
    assert np.allclose(np.linalg.norm(J, axis=1), 1)
    assert np.isclose(abs(np.vdot(J[0], J[1])), 0, atol=1E-12)
    assert np.allclose(sum(w[i]*stokes_from_complex_E(J[i]) for i in range(2)), S)


def test_DOP_and_IPS():
    assert np.isclose(DOP(np.array([1.0, 1j])), 1)
    # incoherent mix of x and y with equal intensity is unpolarized:
    assert np.isclose(DOP(np.array([[1.0, 0.0], [0.0, 1.0]])), 0)
    assert np.isclose(IPS(S_unpol, [0, 1]), 0.5)
    assert np.isclose(IPS(np.array([0.9, 0.0, 0.0, 0.0]) + 0.1*np.array([1, -1, 0, 0]), [0, 1]), 0.55)
    assert np.isclose(IPS(np.array([[1.0], [1j]]), [1, 0]), 0.5)
    assert np.isclose(IPS(np.array([np.cos(0.3), np.sin(0.3)]), [1, 0]), np.cos(0.3)**2)


# masks and sources:

def test_lines_spaces_orders_match_fft_of_mask():
    pitch, width, N = 100.0, 30.0, 2000
    x = grid1d(-pitch/2, pitch/2, N)
    t = np.where(np.abs(x) < width/2, -0.25, 1.0)
    c_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(t)))/N

    m, c_m = lines_spaces_orders(pitch, width, 5, line=-0.25)

    assert np.allclose(c_m, c_fft[N//2 - 5:N//2 + 6], atol=1E-3)


@pytest.mark.parametrize("shape", ["conventional", "annular", "dipole", "quasar"])
def test_source_points(shape):
    s_x, s_y, w = source_points(shape, sigma=0.9, sigma_in=0.6, n_samples=31)
    r = np.hypot(s_x, s_y)

    assert np.isclose(w.sum(), 1)
    assert np.all(r <= 0.9 + 1E-12)
    if shape != "conventional":
        assert np.all(r >= 0.6 - 1E-12)
    # point symmetric sources:
    pts = set(zip(np.round(s_x, 9), np.round(s_y, 9)))
    assert pts == set(zip(np.round(-s_x, 9), np.round(-s_y, 9)))
    if shape == "dipole":
        assert np.all(np.abs(s_x) > np.abs(s_y))


# imaging:

@pytest.mark.parametrize("pitch", [150.0, 200.0, 300.0])
def test_two_beam_contrast(pitch):
    """Orders -1 and +1 of a phase grating under normal incidence."""

    NA = 1.35
    x = np.linspace(-pitch/2, pitch/2, 801)
    theta = np.arcsin(WL/(pitch*N_WATER))
    orders = ([-1, 1], [1.0, 1.0])

    for pol, name in (("y", "TE"), ("x", "TM"), ("unpolarized", "unpolarized")):
        I = image_1d(x, pitch, orders, NA, WL, polarization=pol, n=N_WATER)
        assert np.isclose(contrast(I), two_beam_contrast(theta, name))


def test_scalar_limit_at_low_NA():
    pitch, NA = 2000.0, 0.15
    x = np.linspace(-pitch/2, pitch/2, 401)
    orders = lines_spaces_orders(pitch, pitch/2, 10)
    src = source_points("conventional", 0.6, n_samples=15)

    I_vec = image_1d(x, pitch, orders, NA, WL, src, "unpolarized")
    I_scalar = image_1d(x, pitch, orders, NA, WL, src, "unpolarized", vectorial=False)

    assert np.abs(I_vec - I_scalar).max() < 0.01*I_scalar.max()


def test_TE_needs_no_vectorial_correction_for_sources_in_the_x_z_plane():
    pitch, NA = 90.0, 1.35
    x = np.linspace(-pitch/2, pitch/2, 201)
    orders = lines_spaces_orders(pitch, pitch/2, 5)
    s_x = np.linspace(-0.9, 0.9, 7)
    src = (s_x, 0*s_x, np.ones(7)/7)

    I_vec, I_xyz = image_1d(x, pitch, orders, NA, WL, src, "y", n=N_WATER, obliquity=False,
                            return_components=True)
    I_scalar = image_1d(x, pitch, orders, NA, WL, src, "y", n=N_WATER, vectorial=False)

    assert np.allclose(I_vec, I_scalar)
    assert np.allclose(I_xyz[[0, 2]], 0)
    _, I_xyz_TM = image_1d(x, pitch, orders, NA, WL, src, "x", n=N_WATER, return_components=True)
    assert I_xyz_TM[2].max() > 0.1*I_xyz_TM[0].max()


def test_unpolarized_is_mean_of_x_and_y_and_stokes_input():
    pitch, NA = 100.0, 1.2
    x = np.linspace(-pitch/2, pitch/2, 101)
    orders = lines_spaces_orders(pitch, 40.0, 5)
    src = source_points("annular", 0.9, 0.5, n_samples=15)

    I_x = image_1d(x, pitch, orders, NA, WL, src, "x", n=N_WATER)
    I_y = image_1d(x, pitch, orders, NA, WL, src, "y", n=N_WATER)
    I_u = image_1d(x, pitch, orders, NA, WL, src, "unpolarized", n=N_WATER)
    I_s = image_1d(x, pitch, orders, NA, WL, src, S_unpol, n=N_WATER)
    I_TE = image_1d(x, pitch, orders, NA, WL, src, "TE", n=N_WATER)
    I_TM = image_1d(x, pitch, orders, NA, WL, src, "TM", n=N_WATER)

    assert np.allclose(I_u, (I_x + I_y)/2)
    assert np.allclose(I_s, I_u)
    # TE and TM are orthogonal at every source point, so they also average to I_u:
    assert np.allclose((I_TE + I_TM)/2, I_u)


def test_open_frame_gives_uniform_unit_intensity():
    x = np.linspace(0, 100, 11)
    for pol in ("x", "y", "TE", "unpolarized"):
        I = image_1d(x, 100.0, ([0], [1.0]), 1.35, WL, None, pol, n=N_WATER)
        assert np.allclose(I, 1)


def test_defocus_is_symmetric_without_aberrations():
    pitch, NA = 120.0, 1.35
    x = np.linspace(-pitch/2, pitch/2, 121)
    orders = lines_spaces_orders(pitch, 60.0, 5)
    src = source_points("annular", 0.9, 0.6, n_samples=15)

    I_plus = image_1d(x, pitch, orders, NA, WL, src, "unpolarized", n=N_WATER, defocus=60.0)
    I_minus = image_1d(x, pitch, orders, NA, WL, src, "unpolarized", n=N_WATER, defocus=-60.0)
    I_focus = image_1d(x, pitch, orders, NA, WL, src, "unpolarized", n=N_WATER)

    assert np.allclose(I_plus, I_minus)
    assert contrast(I_plus) < contrast(I_focus)


@pytest.mark.parametrize("pol", ["x", "y", "unpolarized", "TE"])
def test_image_2d_equals_image_1d_for_lines(pol):
    pitch, NA, n_periods = 90.0, 1.35, 4
    N = 128
    L = n_periods*pitch
    x = grid1d(-L/2, L/2, N)
    y = grid1d(-L/8, L/8, 16)
    X, Y = np.meshgrid(x, y)
    # a mask that is exactly band limited on the grid: orders up to 5
    m, c_m = lines_spaces_orders(pitch, 40.0, 5)
    mask = sum(c*np.exp(2j*np.pi*mm*X/pitch) for mm, c in zip(m, c_m))
    src = source_points("dipole", 0.95, 0.6, n_samples=11)

    I_2d = image_2d(mask, x, y, NA, WL, src, pol, n=N_WATER, defocus=30.0)
    I_1d = image_1d(x, pitch, (m, c_m), NA, WL, src, pol, n=N_WATER, defocus=30.0)

    assert np.allclose(I_2d, I_1d[None, :])


def test_nils_of_sine_image():
    x = np.linspace(-50, 50, 2001)
    pitch = 100.0
    I = 1 + 0.5*np.cos(2*np.pi*x/pitch)
    # at x = pitch/4: I = 1, dI/dx = -0.5*2*pi/pitch
    assert np.isclose(nils(x, I, pitch/4, pitch/2), pitch/2*0.5*2*np.pi/pitch, rtol=1E-4)


def test_NA_larger_than_n_is_rejected():
    with pytest.raises(ValueError):
        image_1d(np.zeros(3), 100.0, ([0], [1.0]), 1.35, WL)
