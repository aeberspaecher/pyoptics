#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.crystals: plane waves in crystals, the 4 x 4 matrix
method, refraction at crystal surfaces, polarizing prisms and rotators."""

import numpy as np
import pytest

from pyoptics import crystals as cr
from pyoptics import thin_films as tf
from pyoptics.arvr import aplate_retardance
from pyoptics.polarization import jones
from pyoptics.sensor import walkoff_angle


LAM = 0.6328


def _phase_free(J):
    """J with its global phase removed (determinant made real positive)."""
    return J*np.exp(-0.5j*np.angle(np.linalg.det(J)))


def test_material_data():
    n_o, n_e = cr.calcite_indices(589.3)
    assert n_o == pytest.approx(1.6584, abs=5e-4) and n_e == pytest.approx(1.4864, abs=5e-4)
    assert cr.quartz_rotatory_power(589.3) == pytest.approx(21.72, abs=0.05)
    assert cr.quartz_rotatory_power(404.7) == pytest.approx(48.9, abs=0.5)
    assert cr.tgg_verdet(1064.0) == pytest.approx(-40.0, rel=0.01)
    assert cr.tgg_verdet(632.8) == pytest.approx(-134.0, rel=0.01)


def test_wave_indices_uniaxial():
    # e wave index 1/n**2 = cos**2/n_o**2 + sin**2/n_e**2 and the walk-off of sensor.walkoff_angle
    n_o, n_e = 1.6584, 1.4864
    for th in np.radians([0.0, 20.0, 44.6, 70.0, 90.0]):
        eps = cr.uniaxial_eps(n_o, n_e, (np.sin(th), 0.0, np.cos(th)))
        n, D, E, S = cr.wave_indices(eps, (0.0, 0.0, 1.0))
        n_eth = 1/np.sqrt(np.cos(th)**2/n_o**2 + np.sin(th)**2/n_e**2)
        assert sorted(n.real) == pytest.approx(sorted([n_o, n_eth]), abs=1e-12)
        rho = cr.walkoff(eps, (0.0, 0.0, 1.0))
        assert max(rho) == pytest.approx(abs(walkoff_angle(n_o, n_e, th)), abs=1e-10)
        # D is perpendicular to the wave normal
        assert np.abs(D[..., 2]).max() < 1e-12


def test_wave_indices_biaxial_optic_axes():
    # along an optic axis of a biaxial crystal the two indices coincide (n_y)
    nx, ny, nz = 1.5, 1.6, 1.7
    V = np.arctan(nz/nx*np.sqrt((ny**2 - nx**2)/(nz**2 - ny**2)))
    n = cr.wave_indices(cr.biaxial_eps(nx, ny, nz), (np.sin(V), 0.0, np.cos(V)))[0]
    assert n.real == pytest.approx([ny, ny], abs=1e-9)


@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("theta", [0.0, 35.0, 70.0])
def test_isotropic_stack_against_thin_films(pol, theta):
    n = np.array([1.38, 2.1, 1.46])
    d = np.array([0.11, 0.07, 0.25])
    xi = np.sin(np.radians(theta))
    t, r = cr.stack_jones([(ni, di) for ni, di in zip(n, d)], LAM, xi, 1.0, 1.52)
    R, T, A = tf.R_T_A_coeffs(np.radians(theta), n, 0*n, d, 1.0, 0.0, 1.52, 0.0, pol, LAM)
    i = 1 if pol == "s" else 0
    assert abs(r[i, i])**2 == pytest.approx(R, abs=1e-10)
    assert abs(t[i, i])**2*cr.power_factor(xi, 1.0, 1.52) == pytest.approx(T, abs=1e-10)
    assert abs(r[0, 1]) < 1e-12 and abs(t[1, 0]) < 1e-12


def test_energy_conservation_anisotropic_stack():
    rng = np.random.default_rng(1)
    layers = []
    for _ in range(4):
        R = cr.rotation_matrix(rng.normal(size=3), rng.uniform(0, np.pi))
        layers.append((cr.biaxial_eps(*rng.uniform(1.4, 1.9, 3), R=R), rng.uniform(0.1, 1.0)))
    for xi in (0.0, 0.5, 0.9):
        t, r = cr.stack_jones(layers, LAM, xi, 1.0, 1.0)
        for j in range(2):
            P = np.sum(np.abs(t[:, j])**2) + np.sum(np.abs(r[:, j])**2)
            assert P == pytest.approx(1.0, abs=1e-9)


def test_wave_plate_against_jones():
    # forward-only Jones of a quarter wave plate in index matched media = J_WP
    n_o, n_e = 1.544, 1.553
    d = LAM/(4*(n_e - n_o))
    phi = 0.4
    a = (np.cos(phi), np.sin(phi), 0.0)
    t, _ = cr.stack_jones([(cr.uniaxial_eps(n_o, n_e, a), d)], LAM, 0.0, n_in=1.548, n_out=1.548,
                          coherent=False)
    J = jones.J_WP(2*np.pi*(n_e - n_o)*d/LAM, phi + np.pi/2)       # fast axis perpendicular to a
    assert np.abs(_phase_free(t) - _phase_free(J)).max() < 2e-3


@pytest.mark.parametrize("azimuth", [0.0, np.pi/2])
def test_oblique_a_plate_against_arvr(azimuth):
    # retardance of an A-plate at oblique incidence: the phase between the eigenwaves
    n_o, n_e, d, xi = 1.5, 1.6, 20.0, 0.4
    t, _ = cr.stack_jones([(cr.uniaxial_eps(n_o, n_e, (1.0, 0, 0)), d)], LAM, xi, coherent=False,
                          azimuth=azimuth)
    w = np.linalg.eigvals(t)
    delta = aplate_retardance(d, LAM, n_o, n_e, xi, azimuth)
    assert np.exp(1j*np.angle(w[0]/w[1])) == pytest.approx(np.exp(1j*delta), abs=1e-9) or \
        np.exp(1j*np.angle(w[1]/w[0])) == pytest.approx(np.exp(1j*delta), abs=1e-9)


def test_coherent_slab_has_fringes_forward_only_not():
    eps = cr.uniaxial_eps(1.5, 1.6, (1.0, 0, 0))
    T_c, T_f = [], []
    for lam in np.linspace(0.6, 0.61, 21):
        t, _ = cr.stack_jones([(eps, 50.0)], lam, coherent=True)
        T_c.append(abs(t[1, 1])**2)
        t, _ = cr.stack_jones([(eps, 50.0)], lam, coherent=False)
        T_f.append(abs(t[1, 1])**2)
    assert np.ptp(T_c) > 0.1 and np.ptp(T_f) < 1e-12
    assert np.mean(T_f) == pytest.approx((1 - 0.04)**2, abs=1e-12)


def test_faraday_rotation_is_nonreciprocal():
    n, L = 1.95, 10.0
    rot = np.radians(45.0)/L
    eps = cr.gyrotropic_eps(n**2*np.eye(3), cr.faraday_g(rot, LAM, n))
    t, _ = cr.stack_jones([(eps, L)], LAM, n_in=n, n_out=n, coherent=False)
    assert np.abs(_phase_free(t) - cr.rotator(np.radians(45.0), faraday=True)).max() < 1e-3
    # the same rod seen by a backward wave: turn it by pi about x (y -> -y, z -> -z)
    eps_b = cr.rotate_eps(eps, cr.rotation_matrix((1.0, 0, 0), np.pi))
    t_b, _ = cr.stack_jones([(eps_b, L)], LAM, n_in=n, n_out=n, coherent=False)
    flip = np.diag([1.0, -1.0])
    lab = flip @ _phase_free(t_b) @ flip
    assert np.abs(lab - cr.rotator(np.radians(45.0), faraday=True, backward=True)).max() < 1e-3


def test_rotators_round_trip():
    a = 0.3
    nat = cr.rotator(a, backward=True) @ cr.rotator(a)
    far = cr.rotator(a, faraday=True, backward=True) @ cr.rotator(a, faraday=True)
    assert nat == pytest.approx(np.eye(2))
    assert far == pytest.approx(cr.rotator(2*a))


def test_elliptical_retarder_limits():
    J = cr.elliptical_retarder(0.7, 0.0, 0.3)
    assert np.abs(_phase_free(J) - _phase_free(jones.J_WP(0.7, 0.3))).max() < 1e-12
    assert cr.elliptical_retarder(0.0, 0.4) == pytest.approx(cr.rotator(0.4))
    J = cr.elliptical_retarder(1.3, 0.8, 0.2)
    assert J.conj().T @ J == pytest.approx(np.eye(2))


def test_isolator():
    Tf, Tb = cr.isolator()
    assert Tf == pytest.approx(1.0) and Tb == pytest.approx(0.0, abs=1e-30)
    Tf, Tb = cr.isolator(np.radians(1.0))
    assert -10*np.log10(Tb) == pytest.approx(-10*np.log10(np.sin(np.radians(1.0))**2), abs=1e-9)
    assert Tf == pytest.approx(np.cos(np.radians(1.0))**2)
    _, Tb = cr.isolator(0.0, extinction=1e5)
    assert Tb == pytest.approx(1e-5, rel=1e-6)


def test_refraction_into_calcite_double_image():
    n_o, n_e = cr.calcite_indices(589.3)
    th = np.radians(44.6)
    eps = cr.uniaxial_eps(n_o, n_e, (np.sin(th), 0.0, np.cos(th)))
    tr, rf = cr.refract(1.0, eps, (0, 0, 1.0), (0, 0, 1.0))
    labels = sorted(w.label for w in tr)
    assert labels == ["e", "o"]
    e = [w for w in tr if w.label == "e"][0]
    assert np.degrees(np.arccos(e.S[2])) == pytest.approx(6.24, abs=0.02)
    assert sum(w.power for w in tr + rf) == pytest.approx(1.0, abs=1e-12)


def test_total_internal_reflection():
    # glass to air beyond the critical angle: nothing is transmitted
    k = 1.5*np.array([np.sin(np.radians(50.0)), 0.0, np.cos(np.radians(50.0))])
    tr, rf = cr.refract(1.5, 1.0, (0, 0, 1.0), k)
    assert tr == [] and sum(w.power for w in rf) == pytest.approx(1.0)


def test_glan_taylor_field_of_view():
    n_o, n_e = cr.calcite_indices(589.3)
    th = np.linspace(-8, 8, 161)
    T_e, T_o = cr.glan_prism(th, n_o, n_e, 38.5)
    good = (T_e > 0.5) & (T_o < 1e-6)
    lo, hi = th[good].min(), th[good].max()
    # about 8 degrees of symmetric-ish field (Bennett: 7.5 to 9)
    assert 6.5 < hi - lo < 10 and lo < 0 < hi
    assert T_e[th == 0][0] == pytest.approx(0.877, abs=0.01)


def test_wollaston_and_rochon_split():
    n_o, n_e = cr.calcite_indices(589.3)
    w = cr.wollaston(n_o, n_e, 20.0)
    ang = sorted(np.degrees(np.arctan2(x.k[0], x.k[2])) for x in w)
    assert ang[1] - ang[0] == pytest.approx(np.degrees(2*np.arcsin((n_o - n_e)*np.tan(np.radians(20.0)))),
                                            abs=0.02)
    r = cr.rochon(n_o, n_e, 20.0)
    ang = sorted(abs(np.degrees(np.arctan2(x.k[0], x.k[2]))) for x in r)
    assert ang[0] < 1e-9 and 3.0 < ang[1] < 4.0
