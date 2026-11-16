import numpy as np
import pytest

from pyoptics import arvr
from pyoptics.litho.rcwa import rcwa
from pyoptics.polarization.jones import J_WP


def test_k_space_round_trip_and_guiding():
    tx, ty = np.radians([10.0, -15.0]), np.radians([5.0, 3.0])
    kx, ky = arvr.field_to_k(tx, ty)
    bx, by = arvr.k_to_field(kx, ky)
    assert np.allclose(bx, tx) and np.allclose(by, ty)
    gx, gy = arvr.diffract(kx, ky, 0.372, 0.53)
    assert np.allclose(gx - kx, 0.53/0.372) and np.allclose(gy, ky)
    assert arvr.is_guided(1.2, 0.0, 1.9) and not arvr.is_guided(0.9, 0.0, 1.9)
    assert not arvr.is_guided(1.89, 0.0, 1.9)                 # beyond 75°
    assert np.isnan(arvr.direction(2.0, 0.0, 1.5)).all()
    assert np.isclose(np.linalg.norm(arvr.direction(0.3, 0.4, 1.0)), 1.0)


def test_max_fov_fits_the_annulus():
    n = 1.9
    fov = arvr.max_fov_1d(n)
    g = 0.53/arvr.symmetric_period(n, 0.53)
    k = np.sin(np.array([-fov/2, fov/2])) + g
    assert np.allclose(k, [1.0, n*np.sin(np.radians(75.0))])


def test_uniform_outcoupling_gives_equal_copies():
    for loss in (0.0, 0.02):
        eta = arvr.uniform_outcoupling(8, loss)
        p = arvr.replicated_powers(eta, loss)
        assert np.allclose(p, p[0])
    assert np.allclose(arvr.uniform_outcoupling(4), [1/4, 1/3, 1/2, 1])


def test_eyebox_efficiency_overlap():
    # one copy of width 4 and power 1 centred at 0, pupil 2: full overlap gives 0.5
    assert np.isclose(arvr.eyebox_efficiency(0.0, 2.0, np.array([0.0]), np.array([1.0]), 4.0), 0.5)
    assert np.isclose(arvr.eyebox_efficiency(3.0, 2.0, np.array([0.0]), np.array([1.0]), 4.0), 0.0)


def test_slanted_grating_breaks_symmetry():
    P, lam, n = 0.372, 0.53, 1.91
    for layers, sym in ((arvr.binary_grating(P, 0.3, 0.5, n), True),
                        (arvr.slanted_grating(P, 0.35, 0.5, np.radians(20), n, n_slices=12), False)):
        r = rcwa(P, layers, lam, k_x=0.0, n_in=1.0, n_out=n, n_orders=10)
        R, T = r.efficiencies(np.array([0, 1], dtype=complex))
        p1, m1 = T[0, r.m == 1][0], T[0, r.m == -1][0]
        assert np.isclose(p1, m1, rtol=1e-6) == sym
    assert p1 > 3*m1


def test_kogelnik_against_rcwa_transmission():
    lam, n0, n1, period = 0.532, 1.5, 0.02, 0.8
    th = arvr.bragg_angle(period, lam, n0, np.pi/2)
    d = lam*np.cos(th)/(2*n1)                       # nu = pi/2
    assert np.isclose(arvr.kogelnik(th, lam, period, d, n0, n1), 1.0)
    for dth in (0.0, np.radians(0.6)):
        r = rcwa(period, arvr.volume_grating(period, d, n0, n1), lam, k_x=n0*np.sin(th + dth),
                 n_in=n0, n_out=n0, n_orders=8)
        R, T = r.efficiencies(np.array([0, 1], dtype=complex))
        assert abs(T[0, r.m == -1][0] - arvr.kogelnik(th + dth, lam, period, d, n0, n1)) < 0.01


def test_kogelnik_reflection_tanh():
    lam, n0, n1, d = 532.0, 1.5, 0.02, 10e3
    eta = arvr.kogelnik(0.0, lam, lam/(2*n0), d, n0, n1, slant=0.0)
    assert np.isclose(eta, np.tanh(np.pi*n1*d/lam)**2)


def test_quartz_and_retardance():
    n_o, n_e = arvr.quartz_indices(589.3)
    assert abs(n_o - 1.5443) < 2e-4 and abs(n_e - 1.5534) < 2e-4
    d = arvr.aplate_retardance(15e3, 550.0, n_o, n_e)
    assert np.isclose(d, 2*np.pi*(n_e - n_o)*15e3/550.0)
    # small angles: the change is second order; opposite signs in and across the axis plane
    d_in = arvr.aplate_retardance(15e3, 550.0, n_o, n_e, 0.3, 0.0)
    d_across = arvr.aplate_retardance(15e3, 550.0, n_o, n_e, 0.3, np.pi/2)
    assert d_in < d < d_across


def test_pancake_ideal_and_errors():
    out = arvr.pancake_jones(J_WP(np.pi/2, -np.pi/4))
    assert np.isclose(out["T_signal"], 0.25) and out["T_direct"] < 1e-20
    out = arvr.pancake_jones(J_WP(np.pi/2 + 0.2, -np.pi/4))
    assert out["T_direct"] > 0 and out["T_signal"] < 0.25
    # matched crossed plates cancel on the direct path
    out = arvr.pancake_jones(J_WP(np.pi/2 + 0.2, -np.pi/4), J_input=J_WP(np.pi/2 + 0.2, np.pi/4))
    assert out["T_direct"] < 1e-20 and out["T_five_pass"] > 0
    # the clean-up polarizer removes the reflective polarizer's leak
    out = arvr.pancake_jones(J_WP(np.pi/2, -np.pi/4), leak=0.2)
    assert out["T_direct"] < 1e-20


def test_windshield_wedge():
    th, t, n = np.radians(62), 4.8, 1.52
    s = arvr.windshield_double_image(th, t, n, 0.0, 2500.0)
    assert 3*np.pi/180/60 < abs(s) < 6*np.pi/180/60
    assert abs(arvr.windshield_double_image(th, t, n, 0.0, np.inf)) < 1e-12
    w = arvr.wedge_for_distance(th, t, n, 2500.0)
    assert abs(arvr.windshield_double_image(th, t, n, w, 2500.0)) < 1e-9
    assert 0.2e-3 < abs(w) < 0.6e-3


def test_hud_vergence():
    flat = arvr.hud_system(R_mirror=np.inf, R_windshield=np.inf)
    V, slope = arvr.vergence(flat, [0, 0, -250.0], [0.0, 0.0], [0, 0, 1])
    # two plane mirrors: the display is seen at its unfolded distance
    assert np.allclose(V, np.eye(2)/(250 + 400 + 900), atol=1e-9)
    assert np.allclose(slope, 0.0, atol=1e-9)
    s = arvr.hud_system()
    V, _ = arvr.vergence(s, [0, 0, -250.0], [0.0, 0.0], [0, 0, 1])
    assert V[0, 0] > V[1, 1] > 0                     # astigmatic, virtual image
