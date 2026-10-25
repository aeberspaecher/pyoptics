#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the HG/LG mode decomposition, M² from the mode content, LP fibre
modes and coupling efficiencies against analytical results.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import jv, kv

from pyoptics import abcd, modes
from pyoptics.beam_quality import beam_widths, fit_caustic
from pyoptics.propagators import fresnel_TF
from pyoptics.utils import grid2d, scalar_product


WL = 1.0
W0 = 8.0
Z = 30.0


@pytest.fixture(scope="module")
def grid():
    return grid2d(-60, 60, -60, 60, 256)


@pytest.mark.parametrize("l, m", [(0, 0), (1, 0), (2, 3), (4, 1)])
def test_pure_hg_mode_has_one_coefficient(grid, l, m):
    x, y, X, Y = grid
    field = 2.5j*modes.hg_mode(l, m, x, y, W0, WL, Z)
    d = modes.decompose_hg(field, x, y, W0, WL, Z, max_order=6)
    fractions = d.power_fractions
    assert fractions[(l, m)] == pytest.approx(1.0, abs=1E-10)
    assert sum(f for key, f in fractions.items() if key != (l, m)) < 1E-10
    assert d.reconstruction_error < 1E-6


@pytest.mark.parametrize("p, l", [(0, 0), (0, 1), (1, -2), (2, 1)])
def test_pure_lg_mode_has_one_coefficient(grid, p, l):
    x, y, X, Y = grid
    field = modes.lg_mode(p, l, x, y, W0, WL, Z)
    d = modes.decompose_lg(field, x, y, W0, WL, Z, max_order=6)
    assert d.power_fractions[(p, l)] == pytest.approx(1.0, abs=1E-10)
    assert d.captured_power == pytest.approx(1.0, abs=1E-10)


def test_hg_lg_index_sets_have_equal_size():
    for N in range(8):
        assert len(modes.hg_indices(N)) == len(modes.lg_indices(N)) == (N + 1)*(N + 2)//2


@pytest.mark.parametrize("method", ["projection", "lstsq"])
def test_sum_of_modes_is_recovered(grid, method):
    x, y, X, Y = grid
    rng = np.random.default_rng(1)
    keys = modes.hg_indices(4)
    c = rng.normal(size=len(keys)) + 1j*rng.normal(size=len(keys))
    field = sum(ci*modes.hg_mode(*key, x, y, W0, WL, Z, 2.0, -1.0) for ci, key in zip(c, keys))
    d = modes.decompose_hg(field, x, y, W0, WL, Z, max_order=4, x0=2.0, y0=-1.0, method=method)
    # the analytic modes are normalized to unit power:
    assert np.allclose([d.coefficients[key] for key in keys], c, atol=1E-8)
    assert d.reconstruction_error < 1E-8
    assert np.allclose(d.reconstruct(x, y), field, atol=1E-8)


@pytest.mark.parametrize("decompose", [modes.decompose_hg, modes.decompose_lg])
def test_propagated_expansion_matches_fresnel(grid, decompose):
    x, y, X, Y = grid
    field = (modes.hg_mode(0, 0, x, y, W0, WL) + 0.4j*modes.hg_mode(2, 1, x, y, W0, WL)
             + 0.3*modes.hg_mode(0, 3, x, y, W0, WL))
    d = decompose(field, x, y, W0, WL, max_order=3)
    dz = 150.0
    expected = fresnel_TF(field, x, y, dz, WL)[0]
    assert np.allclose(d.propagate(dz).reconstruct(x, y), expected, atol=1E-8)


def test_lg_from_hg_superposition(grid):
    # HG_10 + i*HG_01 is the donut LG_0,+1:
    x, y, X, Y = grid
    field = modes.hg_mode(1, 0, x, y, W0, WL) + 1j*modes.hg_mode(0, 1, x, y, W0, WL)
    d = modes.decompose_lg(field, x, y, W0, WL, max_order=3)
    assert d.power_fractions[(0, 1)] == pytest.approx(1.0, abs=1E-10)


def test_plane_waist_conversion():
    w0, z = 5.0, 37.0
    w, R = modes.plane_from_waist(w0, z, WL)
    assert np.allclose(modes.waist_from_plane(w, R, WL), (w0, z))
    assert np.allclose(modes.waist_from_plane(w0, np.inf, WL), (w0, 0.0))


@pytest.mark.parametrize("method", ["overlap", "moments"])
def test_fit_basis_of_gaussian(grid, method):
    x, y, X, Y = grid
    field = modes.hg_mode(0, 0, x, y, 7.0, WL, z=50.0, x0=3.0, y0=-2.0)
    basis = modes.fit_basis(field, x, y, WL, method=method)
    rtol = 1E-5 if method == "overlap" else 1E-2
    assert basis.w0 == pytest.approx(7.0, rel=rtol)
    assert basis.z == pytest.approx(50.0, rel=rtol)
    assert (basis.x0, basis.y0) == pytest.approx((3.0, -2.0), abs=1E-5)


def test_fitted_decomposition_of_higher_mode(grid):
    x, y, X, Y = grid
    field = modes.hg_mode(1, 2, x, y, 7.0, WL, z=-40.0)
    d = modes.decompose_hg(field, x, y, wavelength=WL, max_order=4)
    assert d.basis.w0 == pytest.approx(7.0, rel=1E-3)
    assert d.basis.z == pytest.approx(-40.0, rel=1E-3)
    assert d.power_fractions[(1, 2)] == pytest.approx(1.0, abs=1E-5)


@pytest.mark.parametrize("l, m", [(0, 0), (1, 0), (2, 1), (0, 3)])
def test_M2_of_pure_hg_modes(grid, l, m):
    x, y, X, Y = grid
    d = modes.decompose_hg(modes.hg_mode(l, m, x, y, W0, WL, Z), x, y, W0, WL, Z, max_order=4)
    assert d.M2() == pytest.approx((2*l + 1, 2*m + 1), rel=1E-6)


def test_M2_of_lg_mode(grid):
    x, y, X, Y = grid
    d = modes.decompose_lg(modes.lg_mode(1, 2, x, y, W0, WL, Z), x, y, W0, WL, Z, max_order=5)
    assert d.M2(x, y) == pytest.approx((5.0, 5.0), rel=1E-6)


def test_M2_from_modes_agrees_with_caustic_fit():
    # coherent superposition with cross terms (centroid shift and widths that
    # change along z differently from an incoherent mixture):
    x, y, X, Y = grid2d(-150, 150, -150, 150, 512)
    w0 = 10.0
    z_R = np.pi*w0**2/WL
    c = {(0, 0): 1.0, (1, 0): 0.6j, (2, 0): 0.5*np.exp(0.7j), (0, 2): 0.4}

    def field(z):
        return sum(ci*modes.hg_mode(*key, x, y, w0, WL, z) for key, ci in c.items())

    M2_x, M2_y = modes.decompose_hg(field(0.3*z_R), x, y, w0, WL, 0.3*z_R, max_order=3).M2()
    # incoherent mixture would give 1 + 2*(0.36 + 2*0.25)/1.77 in x:
    assert M2_x != pytest.approx(1 + 2*(0.36 + 2*0.25)/1.77, rel=1E-2)

    zs = np.linspace(-2.5*z_R, 2.5*z_R, 21)
    widths = [beam_widths(np.abs(field(z))**2, x, y, iterate=False) for z in zs]
    # the principal axes stay aligned with x and y (no x-y cross terms in c):
    d_x = [w.d_x if abs(w.phi) < np.pi/8 else w.d_y for w in widths]
    d_y = [w.d_y if abs(w.phi) < np.pi/8 else w.d_x for w in widths]
    assert fit_caustic(zs, d_x, WL).M2 == pytest.approx(M2_x, rel=1E-3)
    assert fit_caustic(zs, d_y, WL).M2 == pytest.approx(M2_y, rel=1E-3)


# fibres:
def test_v_number():
    a, n1, n2 = 4.1, 1.4504, 1.4447
    assert modes.v_number(a, n1, n2, 1.55) == pytest.approx(2*np.pi*a/1.55*np.sqrt(n1**2 - n2**2))


@pytest.mark.parametrize("l, m, V_c", [(0, 1, 0.3), (1, 1, 2.404826), (0, 2, 3.831706),
                                       (2, 1, 3.831706), (3, 1, 5.135622), (1, 2, 5.520078)])
def test_lp_cutoffs(l, m, V_c):
    if (l, m) != (0, 1):
        assert modes.lp_cutoff(l, m) == pytest.approx(V_c, abs=1E-6)
    keys = lambda V: {(mode.l, mode.m) for mode in modes.lp_modes(V)}
    assert (l, m) in keys(V_c + 1E-3)
    if (l, m) != (0, 1):
        assert (l, m) not in keys(V_c - 1E-3)


def test_single_mode_below_2405():
    assert [(mode.l, mode.m) for mode in modes.lp_modes(2.4)] == [(0, 1)]


@pytest.mark.parametrize("V", [1.5, 1.8, 2.0, 2.2, 2.5])
def test_lp01_b_against_rudolph_neumann(V):
    # b = (1.1428 - 0.996/V)**2, accurate to about 0.2 % for 1.5 < V < 2.5
    assert modes.lp_mode(0, 1, V).b == pytest.approx((1.1428 - 0.996/V)**2, rel=3E-3)


@pytest.mark.parametrize("V", [2.0, 4.5, 8.0])
def test_lp_modes_are_continuous_at_interface(V):
    # field and derivative are continuous at r = a (weakly guiding modes):
    for mode in modes.lp_modes(V):
        l, U, W = mode.l, mode.U, mode.W
        d_core = U*(jv(l - 1, U) - jv(l + 1, U))/2/jv(l, U)
        d_clad = -W*(kv(l - 1, W) + kv(l + 1, W))/2/kv(l, W)
        assert d_core == pytest.approx(d_clad, rel=1E-8, abs=1E-8)
        assert mode.radial(1.0, 1.0) == pytest.approx(1.0)
        assert U**2 + W**2 == pytest.approx(V**2)


def test_effective_index_and_degeneracy():
    a, n1, n2, wl = 25.0, 1.46, 1.44, 1.0
    lp = modes.lp_modes(core_radius=a, n_core=n1, n_clad=n2, wavelength=wl)
    assert all(n2 < mode.n_eff < n1 for mode in lp)
    assert np.all(np.diff([mode.n_eff for mode in lp]) <= 0)
    V = modes.v_number(a, n1, n2, wl)
    # number of guided modes ~ V**2/2 for large V:
    assert sum(mode.degeneracy for mode in lp) == pytest.approx(V**2/2, rel=0.1)


def test_lp_modes_are_orthonormal_on_grid():
    x, y, X, Y = grid2d(-20, 20, -20, 20, 256)
    a = 5.0
    fields = []
    for mode in modes.lp_modes(5.0):
        for orientation in (["cos"] if mode.l == 0 else ["cos", "sin"]):
            fields.append(modes.lp_mode_field(mode, x, y, a, orientation))
    G = np.array([[scalar_product(f, g, x, y) for g in fields] for f in fields])
    assert np.allclose(G, np.eye(len(fields)), atol=1E-3)


def _radial_overlap(mode, w, a=1.0):
    f = lambda r: mode.radial(r, a)
    integrate = lambda g: quad(g, 0, a)[0] + quad(g, a, np.inf)[0]
    num = integrate(lambda r: f(r)*np.exp(-r**2/w**2)*r)
    return num**2/(integrate(lambda r: f(r)**2*r)*w**2/4)


@pytest.mark.parametrize("V", [1.2, 1.5, 1.8, 2.1, 2.4])
def test_marcuse_mode_field_radius(V):
    mode = modes.lp_mode(0, 1, V)
    best = minimize_scalar(lambda w: -_radial_overlap(mode, w), bounds=(0.3, 10), method="bounded",
                           options={"xatol": 1E-10}).x
    assert modes.marcuse_mode_field_radius(V, 1.0) == pytest.approx(best, rel=1E-2)


# coupling:
def test_gaussian_into_gaussian_matches_abcd(grid):
    x, y, X, Y = grid
    beam = modes.hg_mode(0, 0, x, y, 6.0, WL, z=40.0)
    target = modes.hg_mode(0, 0, x, y, 8.0, WL, z=-15.0)
    q1, q2 = 40.0 + 1j*np.pi*6.0**2/WL, -15.0 + 1j*np.pi*8.0**2/WL
    assert modes.coupling_efficiency(beam, target, x, y) == pytest.approx(
        abcd.coupling_efficiency(q1, q2, WL), rel=1E-8)


def test_coupling_sweeps_for_gaussians(grid):
    x, y, X, Y = grid
    w = 6.0
    beam = modes.hg_mode(0, 0, x, y, w, WL)
    offsets = np.linspace(0, 8, 5)
    assert np.allclose(modes.coupling_vs_offset(beam, beam, x, y, offsets), np.exp(-offsets**2/w**2))
    angles = np.linspace(0, 0.06, 5)
    eta_tilt = np.exp(-(np.pi*w*np.sin(angles)/WL)**2)
    assert np.allclose(modes.coupling_vs_tilt(beam, beam, x, y, WL, angles, axis="y"), eta_tilt)
    dz = np.linspace(0, 200, 5)
    q = 1j*np.pi*w**2/WL
    eta_dz = [abcd.coupling_efficiency(q + d, q, WL) for d in dz]
    assert np.allclose(modes.coupling_vs_defocus(beam, beam, x, y, WL, dz, propagator=fresnel_TF), eta_dz, atol=1E-6)
    # the default (non-paraxial) angular spectrum is symmetric in dz and close to it:
    eta_as = modes.coupling_vs_defocus(beam, beam, x, y, WL, np.concatenate([-dz, dz]))
    assert np.all(np.isfinite(eta_as))
    assert np.allclose(eta_as[:len(dz)], eta_as[len(dz):], atol=1E-10)
    assert np.allclose(eta_as[len(dz):], eta_dz, atol=1E-3)


def test_gaussian_into_lp01_matches_radial_overlap():
    x, y, X, Y = grid2d(-30, 30, -30, 30, 256)
    a, V = 4.0, 2.2
    mode = modes.lp_mode(0, 1, V)
    w = 1.1*modes.marcuse_mode_field_radius(V, a)
    eta = modes.coupling_efficiency(modes.hg_mode(0, 0, x, y, w, WL), mode.field(x, y, a), x, y)
    assert eta == pytest.approx(_radial_overlap(mode, w/a), rel=1E-6)


def test_offset_beam_excites_lp11():
    x, y, X, Y = grid2d(-30, 30, -30, 30, 256)
    a, V = 4.0, 3.5
    lp = modes.lp_modes(V)
    fields = [mode.field(x, y, a, o) for mode in lp for o in (["cos"] if mode.l == 0 else ["cos", "sin"])]
    w = modes.marcuse_mode_field_radius(V, a)
    centred = modes.mode_power_fractions(modes.hg_mode(0, 0, x, y, w, WL), fields, x, y)
    shifted = modes.mode_power_fractions(modes.hg_mode(0, 0, x, y, w, WL, x0=1.5), fields, x, y)
    # LP01, LP11 cos (x), LP11 sin (y):
    assert centred[0] > 0.99 and centred[1] < 1E-10 and centred[2] < 1E-10
    assert shifted[1] > 0.05 and shifted[2] < 1E-10
    assert np.sum(shifted) <= 1.0 + 1E-6
