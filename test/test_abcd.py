#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test ABCD matrices, Gaussian beam propagation, resonator modes and the
Collins integral against textbook formulas.
"""

import numpy as np
import pytest

from pyoptics import abcd
from pyoptics.abcd import OpticalSystem
from pyoptics.propagators import fresnel_TF
from pyoptics.utils import grid2d


WL = 1.0


def test_elements_have_unit_determinant():
    matrices = [abcd.free_space(3.0, 1.5), abcd.thin_lens(-20.0), abcd.refraction(10.0, 1.0, 1.7),
                abcd.mirror(50.0, 0.3, "sagittal", 1.3), abcd.grin(7.0, 1.6, 0.2),
                abcd.thick_lens(30.0, -40.0, 5.0, 1.5)]
    for M in matrices:
        assert np.linalg.det(M) == pytest.approx(1.0)


def test_compose_order():
    d, f = 10.0, 25.0
    expected = abcd.thin_lens(f) @ abcd.free_space(d)
    assert np.allclose(abcd.compose(abcd.free_space(d), abcd.thin_lens(f)), expected)
    assert np.allclose(OpticalSystem().free_space(d).thin_lens(f).matrix, expected)


def test_two_lenses_effective_focal_length():
    f1, f2, d = 100.0, -50.0, 30.0
    system = OpticalSystem().thin_lens(f1).free_space(d).thin_lens(f2)
    cp = system.cardinal_points()
    f = 1/(1/f1 + 1/f2 - d/(f1*f2))
    assert cp.f == pytest.approx(f) and cp.f_prime == pytest.approx(f)
    assert cp.bfl == pytest.approx(f*(f1 - d)/f1)
    assert cp.ffl == pytest.approx(f*(f2 - d)/f2)
    assert cp.H_prime == pytest.approx(d - f*d/f1)
    # nodal points coincide with principal points in a homogeneous medium:
    assert cp.N == pytest.approx(cp.H) and cp.N_prime == pytest.approx(cp.H_prime)


def test_thick_lens_lensmaker():
    R1, R2, d, n = 40.0, -60.0, 8.0, 1.5
    system = OpticalSystem().thick_lens(R1, R2, d, n)
    cp = system.cardinal_points()
    f = 1/((n - 1)*(1/R1 - 1/R2 + (n - 1)*d/(n*R1*R2)))
    assert cp.f_prime == pytest.approx(f)
    assert cp.H == pytest.approx(-f*(n - 1)*d/(n*R2))
    assert cp.H_prime == pytest.approx(d - f*(n - 1)*d/(n*R1))
    assert system.length == pytest.approx(d)


def test_spherical_interface_focus_and_nodal_points():
    """Parallel rays focus at n2*R/(n2 - n1) behind a single surface; the
    nodal points lie in the centre of curvature."""

    R, n2 = 20.0, 1.5
    system = OpticalSystem().refraction(R, n2)
    cp = system.cardinal_points()
    assert cp.F_prime == pytest.approx(n2*R/(n2 - 1))
    assert cp.f == pytest.approx(R/(n2 - 1)) and cp.f_prime == pytest.approx(n2*R/(n2 - 1))
    assert cp.N == pytest.approx(R) and cp.N_prime == pytest.approx(R)
    y, theta = system.free_space(100).trace([1.0, 0.0], [cp.F_prime])
    assert y[0] == pytest.approx(0.0, abs=1E-12)


def test_snell_flat_interface():
    system = OpticalSystem().free_space(5).refraction(np.inf, 1.5).free_space(5)
    y, theta = system.trace([0.0, 0.1], [2.0, 7.0])
    assert theta == pytest.approx([0.1, 0.1/1.5])
    assert y[1] == pytest.approx(0.5 + 2*0.1/1.5)


@pytest.mark.parametrize("s", [150.0, 60.0, 40.0])
def test_thin_lens_imaging(s):
    f = 50.0
    s_prime, m = abcd.image(abcd.thin_lens(f), s)
    assert 1/s + 1/s_prime == pytest.approx(1/f)
    assert m == pytest.approx(-s_prime/s)
    if s_prime < 0:  # virtual image
        return
    # rays from an object point meet in the image point:
    system = OpticalSystem().free_space(s).thin_lens(f)
    y, _ = system.trace(np.array([[1.0, 1.0, 1.0], [-0.02, 0.0, 0.01]]), [s + s_prime])
    assert np.allclose(y, m)


def test_image_at_infinity():
    s_prime, m = abcd.image(abcd.thin_lens(50.0), 50.0)
    assert np.isinf(s_prime) and np.isinf(m)


def test_mirror_tilt():
    R, theta = 100.0, 0.4
    assert -1/abcd.mirror(R, theta, "t")[1, 0] == pytest.approx(R*np.cos(theta)/2)
    assert -1/abcd.mirror(R, theta, "s")[1, 0] == pytest.approx(R/np.cos(theta)/2)
    with pytest.raises(ValueError):
        abcd.mirror(R, theta, "x")


def test_grin_quarter_pitch():
    n0, g = 1.6, 0.25
    M = abcd.grin(np.pi/(2*g), n0, g)
    assert M[0, 0] == pytest.approx(0.0, abs=1E-12)
    system = OpticalSystem().grin(4*np.pi/g, n0, g)
    z = np.linspace(0, 4*np.pi/g, 50)
    y, theta = system.trace([1.0, 0.0], z)
    assert np.allclose(y, np.cos(g*z))
    assert np.allclose(theta, -g*np.sin(g*z))
    assert np.allclose(abcd.grin(3.0, n0, 0.0), abcd.free_space(3.0, n0))


def test_matrix_at_inside_and_outside():
    system = OpticalSystem(n=1.2).free_space(10).thin_lens(20).free_space(5, n=1.5)
    assert np.allclose(system.matrix_at(-4), abcd.free_space(-4, 1.2))
    assert np.allclose(system.matrix_at(4), abcd.free_space(4, 1.2))
    assert np.allclose(system.matrix_at(10), abcd.thin_lens(20) @ abcd.free_space(10, 1.2))
    assert np.allclose(system.matrix_at(20), abcd.free_space(5, 1.5) @ system.matrix)
    assert system.n_at(5) == 1.2 and system.n_at(12) == 1.5 and system.n_at(30) == 1.5
    assert system.n_out == 1.5 and system.length == 15


def test_q_conversions():
    w, R, n = 30.0, -500.0, 1.4
    q = abcd.q_from_w_R(w, R, WL, n)
    assert abcd.beam_radius(q, WL, n) == pytest.approx(w)
    assert abcd.curvature_radius(q) == pytest.approx(R)
    w0, z_w = abcd.waist(q, WL, n)
    assert abcd.q_from_waist(w0, -z_w, WL, n) == pytest.approx(q)
    assert z_w > 0  # converging beam
    assert np.isinf(abcd.curvature_radius(abcd.q_from_waist(w0, 0.0, WL, n)))


@pytest.mark.parametrize("n", [1.0, 1.5])
def test_free_space_gaussian_beam(n):
    w0 = 20.0
    z_R = np.pi*n*w0**2/WL
    z = np.linspace(-3*z_R, 3*z_R, 301)
    system = OpticalSystem(n=n).free_space(6*z_R)
    trace = system.trace_beam(abcd.q_from_waist(w0, 0.0, WL, n), WL, z)
    assert np.allclose(trace.w, w0*np.sqrt(1 + (z/z_R)**2))
    nonzero = z != 0
    assert np.allclose(trace.R[nonzero], z[nonzero]*(1 + (z_R/z[nonzero])**2))
    assert np.allclose(trace.gouy, np.arctan(z/z_R))


def test_lens_focus_self_formula():
    """Waist behind a lens (S. A. Self, Appl. Opt. 22, 658 (1983))."""

    w0, s, f = 50.0, -2000.0, 1000.0  # waist 2000 in front of the lens
    z_R = np.pi*w0**2/WL
    q = abcd.q_from_waist(w0, -s, WL)
    q_out = abcd.transform_q(abcd.thin_lens(f), q)
    w0_out, z_out = abcd.waist(q_out, WL)
    s_o = -s
    denominator = (s_o/f - 1)**2 + (z_R/f)**2
    assert w0_out == pytest.approx(w0/np.sqrt(denominator))
    assert z_out == pytest.approx(f*(1 + (s_o/f - 1)/denominator))


def test_gouy_phase_through_focus():
    """A focused beam acquires a Gouy phase of pi from far in front of to far
    behind the focus."""

    f, w = 100.0, 200.0
    system = OpticalSystem().thin_lens(f).free_space(1E6)
    z = np.concatenate([np.linspace(0, 2*f, 4001), np.linspace(2*f, 1E6, 2000)[1:]])
    trace = system.trace_beam(abcd.q_from_waist(w, 0.0, WL), WL, z)
    z_R_lens = np.pi*w**2/WL
    # psi at z -> infinity relative to the lens plane:
    q_focus = abcd.transform_q(abcd.thin_lens(f), abcd.q_from_waist(w, 0.0, WL))
    expected = np.pi/2 - np.arctan(np.real(q_focus)/np.imag(q_focus))
    assert trace.gouy[-1] == pytest.approx(expected, abs=1E-3)
    assert expected == pytest.approx(np.pi, abs=0.02)  # z_R in focus << f
    assert z_R_lens > 10*f


def test_two_mirror_resonator_mode():
    L, R1, R2 = 300.0, 1000.0, 500.0
    g1, g2 = abcd.g_parameters(L, R1, R2)
    round_trip = OpticalSystem().free_space(L).mirror(R2).free_space(L).mirror(R1)
    M = round_trip.matrix
    assert abcd.is_stable(M)
    assert abcd.stability_parameter(M) == pytest.approx(2*g1*g2 - 1)
    q = abcd.eigenmode(M)
    assert abcd.transform_q(M, q) == pytest.approx(q)
    w1 = abcd.beam_radius(q, WL)
    assert w1**2 == pytest.approx(WL*L/np.pi*np.sqrt(g2/(g1*(1 - g1*g2))))
    # the wavefronts fit the mirror: the reference plane lies just behind
    # mirror 1, where the reflected beam converges with R = -R1:
    assert abcd.curvature_radius(q) == pytest.approx(-R1)
    # waist size:
    w0, _ = abcd.waist(q, WL)
    assert w0**2 == pytest.approx(WL*L/np.pi*np.sqrt(g1*g2*(1 - g1*g2)/(g1 + g2 - 2*g1*g2)**2))
    # round trip Gouy phase, from the formula and from the traced beam:
    psi = abcd.round_trip_gouy_phase(M)
    assert psi == pytest.approx(2*np.arccos(np.sqrt(g1*g2)))
    trace = round_trip.trace_beam(q, WL, np.linspace(0, 2*L, 2001))
    assert trace.gouy[-1] == pytest.approx(psi)


def test_unstable_resonator():
    M = OpticalSystem().free_space(300).mirror(100.0).free_space(300).mirror(100.0).matrix
    assert not abcd.is_stable(M)
    with pytest.raises(ValueError):
        abcd.eigenmode(M)


def test_afocal_has_no_focal_points():
    telescope = OpticalSystem().thin_lens(100).free_space(150).thin_lens(50)
    assert telescope.matrix[1, 0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        abcd.cardinal_points(np.array([[-0.5, 150.0], [0.0, -2.0]]))


def test_coupling_efficiency():
    q = abcd.q_from_waist(10.0, 50.0, WL)
    assert abcd.coupling_efficiency(q, q, WL) == pytest.approx(1.0)
    w1, w2 = 10.0, 15.0
    eta = abcd.coupling_efficiency(abcd.q_from_waist(w1, 0, WL), abcd.q_from_waist(w2, 0, WL), WL)
    assert eta == pytest.approx((2*w1*w2/(w1**2 + w2**2))**2)
    # against the overlap integral:
    x, y, X, Y = grid2d(-100, 100, -100, 100, 256)
    q1, q2 = abcd.q_from_w_R(12.0, 800.0, WL), abcd.q_from_w_R(9.0, -2000.0, WL)
    E1, E2 = abcd.gaussian_field(x, y, q1, WL), abcd.gaussian_field(x, y, q2, WL)
    dA = (x[1] - x[0])**2
    assert np.sum(np.abs(E1)**2)*dA == pytest.approx(1.0)
    overlap = np.abs(np.sum(np.conj(E1)*E2)*dA)**2
    assert overlap == pytest.approx(abcd.coupling_efficiency(q1, q2, WL), rel=1E-6)


@pytest.fixture(scope="module")
def grid():
    return grid2d(-400, 400, -400, 400, 256)


def test_collins_free_space_equals_fresnel(grid):
    x, y, X, Y = grid
    E = np.exp(-((X - 20)**2 + Y**2)/40**2)*(np.abs(X) < 100)
    d = 3000.0
    expected = fresnel_TF(E, x, y, d, WL)[0]*np.exp(-2j*np.pi*d/WL)
    assert np.allclose(abcd.collins(E, x, y, abcd.free_space(d), WL), expected)


def test_collins_gaussian_through_lens_system(grid):
    """Wave optics agrees with the ABCD law, including the Gouy phase."""

    x, y, X, Y = grid
    q_in = abcd.q_from_waist(60.0, 0.0, WL)
    E = abcd.gaussian_field(x, y, q_in, WL)
    system = OpticalSystem().free_space(2000).thin_lens(5000).free_space(3500).thin_lens(-3000).free_space(1500)
    M = system.matrix
    q_out = abcd.transform_q(M, q_in)
    E_out = abcd.collins(E, x, y, M, WL)
    expected = abcd.gaussian_field(x, y, q_out, WL)*np.exp(-1j*abcd.gouy_phase(M, q_in))
    assert np.allclose(E_out, expected, atol=1E-6*np.abs(expected).max())


def test_collins_astigmatic(grid):
    x, y, X, Y = grid
    q_in = abcd.q_from_waist(50.0, 0.0, WL)
    E = abcd.gaussian_field(x, y, q_in, WL)
    M_x = abcd.compose(abcd.thin_lens(4000), abcd.free_space(3000))
    M_y = abcd.free_space(3000)
    E_out = abcd.collins(E, x, y, M_x, WL, M_y=M_y)
    expected = (abcd.gaussian_field(x, y, abcd.transform_q(M_x, q_in), WL,
                                    q_y=abcd.transform_q(M_y, q_in))
                *np.exp(-0.5j*(abcd.gouy_phase(M_x, q_in) + abcd.gouy_phase(M_y, q_in))))
    assert np.allclose(E_out, expected, atol=1E-6*np.abs(expected).max())


def test_collins_imaging_plane(grid):
    """B = 0: a 2f-2f system inverts the field."""

    x, y, X, Y = grid
    E = np.exp(-((X - 50)**2 + (Y + 30)**2)/30**2)
    M = OpticalSystem().free_space(2000).thin_lens(1000).free_space(2000).matrix
    assert abs(M[0, 1]) < 1E-9
    E_out = abcd.collins(E, x, y, M, WL)
    expected = -np.exp(-((X + 50)**2 + (Y - 30)**2)/30**2)*np.exp(1j*np.pi/WL*M[1, 0]/M[0, 0]*(X**2 + Y**2))
    inner = (np.abs(X) < 350) & (np.abs(Y) < 350)
    assert np.allclose(E_out[inner], expected[inner], atol=2E-3)
