#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check pyoptics.litho.jones_pupil and the Jones pupil of a ray traced
lens: Pauli algebra, orientation Zernike conventions and symmetries, the
polar decomposition, fits, the Projector hook (Abbe and Hopkins) and the
polarization ray trace of the Caldwell lens.
"""

import numpy as np
import pytest

from pyoptics.litho import (Illumination, LithoSimulation, MaskLayout, Projector, ThinMask, hopkins,
                            lens)
from pyoptics.litho import jones_pupil as jp
from pyoptics.materials import transverse_basis
from pyoptics.raytrace import caldwell_krf_lens


@pytest.fixture(scope="module")
def pupil_points():
    rng = np.random.default_rng(1)
    P = rng.uniform(-1, 1, (2, 3000))

    return P[:, np.hypot(*P) <= 1]


def test_pauli_round_trip():
    rng = np.random.default_rng(2)
    J = rng.normal(size=(5, 2, 2)) + 1j*rng.normal(size=(5, 2, 2))
    assert np.allclose(jp.pauli_matrix(jp.pauli_coefficients(J)), J)
    assert np.allclose(jp.pauli_coefficients(jp.SIGMA[2]), [0, 0, 1, 0])


@pytest.mark.parametrize("j", range(2, 16))
def test_orientation_zernike_first_component_is_the_zernike(pupil_points, j):
    P_x, P_y = pupil_points
    assert np.allclose(jp.orientation_zernike(j, P_x, P_y).real, jp.zernike(j, P_x, P_y))
    # +j and -j differ in the sign of the second component (m != 0)
    if j not in (1, 4, 9):
        assert np.allclose(jp.orientation_zernike(-j, P_x, P_y),
                           np.conj(jp.orientation_zernike(j, P_x, P_y)))


def test_orientation_zernike_symmetries(pupil_points):
    P_x, P_y = pupil_points
    rho, phi = np.hypot(P_x, P_y), np.arctan2(P_y, P_x)
    assert np.allclose(jp.orientation_zernike(5, P_x, P_y), rho**2*np.exp(2j*phi))  # radial
    # rotating the lens by alpha: an orientator rotates by 2*alpha, so the
    # rotationally symmetric terms OZ +5, -6 are unchanged
    alpha = 0.7
    c, s = np.cos(alpha), np.sin(alpha)
    for oz in (5, -6):
        o_rot = jp.orientation_zernike(oz, c*P_x + s*P_y, -s*P_x + c*P_y)*np.exp(2j*alpha)
        assert np.allclose(o_rot, jp.orientation_zernike(oz, P_x, P_y))


def test_decomposition_round_trip(pupil_points):
    P_x, P_y = pupil_points
    pupil = jp.PhysicalJonesPupil({5: 0.02, -6: 0.01, 7: -0.015, -2: 0.005}, {5: 0.03, -3: 0.02},
                                  {4: 0.01}, {1: 0.005})
    g, d = pupil.fields(P_x, P_y)
    J = jp.compose(g, d, transmission=0.8, phase=0.3)
    parts = jp.decompose(J)
    assert np.allclose(parts["transmission"], 0.8) and np.allclose(parts["phase"], 0.3)
    assert np.allclose(parts["retardance"], g) and np.allclose(parts["diattenuation"], d)
    assert np.allclose(parts["U"] @ parts["H"]*np.sqrt(0.8), J)
    assert np.allclose(jp.scalar_free(J), pupil(P_x, P_y))


def test_retarder_and_diattenuator():
    # a quarter wave retarder with the fast axis along x and a diattenuator
    # transmitting x better
    U = jp.compose([0.25, 0, 0], [0, 0, 0])
    assert np.allclose(U, np.diag([np.exp(-0.25j*np.pi), np.exp(0.25j*np.pi)]))
    H = jp.compose([0, 0, 0], [0.2, 0, 0])
    T_x, T_y = np.abs(np.diag(H))**2
    assert np.isclose((T_x - T_y)/(T_x + T_y), 0.2) and np.isclose((T_x + T_y)/2, 1.0)
    with pytest.raises(ValueError):
        jp.compose([0, 0, 0], [1.0, 0, 0])


def test_fits(pupil_points):
    P_x, P_y = pupil_points
    pupil = jp.PhysicalJonesPupil({5: 0.02, -6: 0.01, 7: -0.015, -2: 0.005}, {5: 0.03, -3: 0.02},
                                  {4: 0.01}, {1: 0.005})
    J = pupil(P_x, P_y)
    fit, scalar = jp.PhysicalJonesPupil.fit(P_x, P_y, J, n_terms=16)
    big = {k: v for k, v in fit.retardance.items() if abs(v) > 1e-8}
    assert big.keys() == pupil.retardance.keys()
    assert np.allclose([big[k] for k in big], [pupil.retardance[k] for k in big])
    assert np.allclose(fit(P_x, P_y), J, atol=1e-8)
    assert abs(scalar["phase"][1]) < 1e-8 and np.isclose(scalar["transmission"][1], 1)
    pz = jp.PauliZernikePupil.fit(P_x, P_y, J, n_terms=36)
    assert np.abs(pz(P_x, P_y) - J).max() < 1e-5
    assert np.allclose(pz.pauli(P_x, P_y), jp.pauli_coefficients(pz(P_x, P_y)))
    assert np.allclose((pz @ pz)(P_x, P_y), pz(P_x, P_y) @ pz(P_x, P_y))


# imaging:

def dipole_sim(polarization="x"):
    return LithoSimulation(MaskLayout.lines_spaces(90.0, 45.0), ThinMask.binary(),
                           Illumination.dipole(0.9, 0.6, n_samples=9, polarization=polarization),
                           Projector(1.35, 193.0, n=1.437))


def test_identity_and_half_wave_pupils():
    x = np.linspace(-45, 45, 31)
    sim_x, sim_y = dipole_sim("x"), dipole_sim("y")
    identity = sim_x.projector.with_(jones_pupil=jp.PauliZernikePupil())
    assert np.allclose(sim_x.with_(projector=identity).image(x), sim_x.image(x), atol=1e-14)
    swap = sim_x.projector.with_(jones_pupil=jp.PauliZernikePupil(a0={}, a2={1: 1.0}))  # s_2: x <-> y
    assert np.allclose(sim_x.with_(projector=swap).image(x), sim_y.image(x), atol=1e-14)
    scalar = sim_x.projector.with_(vectorial=False, jones_pupil=jp.PauliZernikePupil(a0={}, a2={1: 1.0}))
    assert np.allclose(sim_x.with_(projector=scalar).image(x),
                       sim_y.with_(projector=sim_y.projector.with_(vectorial=False)).image(x))


def test_hopkins_equals_abbe_with_a_jones_pupil():
    x = np.linspace(-45, 45, 31)
    pupil = jp.PhysicalJonesPupil({5: 0.02, -2: 0.01}, {5: 0.1, 3: 0.05})
    sim = dipole_sim().with_(illumination=Illumination.annular(0.9, 0.6, n_samples=9))
    sim = sim.with_(projector=sim.projector.with_(jones_pupil=pupil))
    assert np.allclose(hopkins.TCC(sim).image(x)[0], sim.image(x), atol=1e-12)


def test_radial_diattenuation_favours_tm():
    """A radial diattenuation transmits the TM (p) part of the orders
    better: x polarized dipole on lines along y (TM) loses contrast less."""

    pupil = jp.PhysicalJonesPupil(diattenuation={5: 0.3})
    x = np.linspace(-45, 45, 31)
    for pol, sign in (("x", 1), ("y", -1)):
        sim = dipole_sim(pol)
        I0 = sim.image(x)
        I1 = sim.with_(projector=sim.projector.with_(jones_pupil=pupil)).image(x)
        assert sign*((I1.max() - I1.min()) - (I0.max() - I0.min())) > 0


# polarization ray trace:

@pytest.fixture(scope="module")
def caldwell():
    return caldwell_krf_lens()


def test_uncoated_lens_is_a_radial_diattenuator(caldwell):
    r = lens.lens_jones_pupil(caldwell, N=25)
    parts = jp.decompose(r.J)
    assert 0.1 < r.transmission < 0.3  # 42 uncoated surfaces
    assert np.abs(parts["retardance"]).max() < 1e-12  # real Fresnel coefficients
    assert np.abs(parts["phase"]).max() < 1e-12
    fit, _ = jp.PhysicalJonesPupil.fit(r.P_x, r.P_y, r.J, n_terms=16)
    d = fit.diattenuation
    assert d[5] > 0.15 and all(abs(v) < 0.05*d[5] for k, v in d.items() if k not in (5, 12))
    # rotational symmetry: J(R P) = R J(P) R^T
    rho = np.hypot(r.P_x, r.P_y)
    i = np.argmin(np.abs(rho - 0.8) + np.abs(r.P_y))  # on the x axis
    assert abs(r.J[i, 0, 1]) < 1e-12 and r.J[i, 0, 0].real > r.J[i, 1, 1].real


def test_coated_lens(caldwell):
    coat = lens.quarter_wave_coating(1.5084, caldwell.wavelength)
    assert np.isclose(coat[0][0]*coat[0][1], caldwell.wavelength/4)
    r = lens.lens_jones_pupil(caldwell, N=25, coating=coat)
    assert r.transmission > 0.99
    parts = jp.decompose(r.J)
    assert np.allclose(jp.decompose(r.J[np.argmin(np.hypot(r.P_x, r.P_y))])["phase"], 0, atol=1e-12)
    d_abs = np.hypot(*parts["diattenuation"][:2])
    assert d_abs.max() < 0.01 and np.abs(parts["retardance"][:2]).max() > 1e-3
    same = lens.lens_jones_pupil(caldwell, N=25, coating=lambda i, n: coat)
    assert np.allclose(same.J, r.J)


def test_off_axis_field(caldwell):
    r = lens.lens_jones_pupil(caldwell, field=(0.0, 11.0), N=16)
    fit, _ = jp.PhysicalJonesPupil.fit(r.P_x, r.P_y, r.J, n_terms=9)
    # the chief ray crosses the surfaces obliquely in the y-z plane: y (p)
    # is transmitted better, a diattenuation along y (OZ +1 < 0)
    assert fit.diattenuation[1] < -0.01


def test_cubic_crystal_birefringence():
    def delta_index(crystal, k):
        k = np.atleast_2d(k)/np.linalg.norm(k)
        return np.ptp(np.linalg.eigvalsh(crystal.index_change(k, transverse_basis(k))))

    c100, c111 = lens.CubicCrystal(-3.4e-7, "100"), lens.CubicCrystal(-3.4e-7, "111")
    assert delta_index(c100, [0, 0, 1]) < 1e-20 and delta_index(c111, [0, 0, 1]) < 1e-20
    assert np.isclose(delta_index(c100, [1, 1, 0]), 3.4e-7)
    assert np.isclose(delta_index(lens.CubicCrystal(-3.4e-7, "110"), [0, 0, 1]), 3.4e-7)
    with pytest.raises(ValueError):
        lens.CubicCrystal(1e-7, "123")


def test_crystal_elements_in_the_lens(caldwell):
    def pupil(axis, clocking):
        return lens.lens_jones_pupil(caldwell, N=15, birefringent={38: lens.CubicCrystal(-3e-7, axis,
                                                                                        clocking)})

    r = pupil("111", 0.0)
    g = jp.decompose(r.J)["retardance"]
    assert 0.02 < np.abs(g[0] + 1j*g[1]).max() < 0.2
    # threefold (fourfold) symmetry: clocking by 120 (90) degrees changes nothing
    assert np.allclose(pupil("111", 2*np.pi/3).J, r.J, atol=1e-10)
    assert np.allclose(pupil("100", np.pi/2).J, pupil("100", 0.0).J, atol=1e-10)
    assert not np.allclose(pupil("111", np.pi/3).J, r.J, atol=1e-4)
    fit, _ = jp.PhysicalJonesPupil.fit(r.P_x, r.P_y, r.J, n_terms=16)
    assert abs(fit.retardance[3]) > 0.05 and abs(fit.retardance[3]) > 3*abs(fit.retardance.get(2, 0))
