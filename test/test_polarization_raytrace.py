#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the polarization ray trace of pyoptics.raytrace with coatings
(pyoptics.coatings) and absorbing or birefringent media (pyoptics.materials).
"""

import numpy as np
import pytest
from scipy.linalg import expm

from pyoptics import thin_films
from pyoptics.coatings import Coating, Layer, PolynomialProfile, RadialProfile, ZernikeProfile
from pyoptics.materials import CubicCrystal, Isotropic, Uniaxial, expm_2x2
from pyoptics.raytrace import IdealLens, Rays, Surface, System

WL = 550e-6  # mm
S, P = (1.0, 0.0), (0.0, 1.0)  # Jones vectors on the aplanatic basis: s, p for rays in the y-z plane


def rays(theta=0.0, y=0.0):
    theta, y = np.broadcast_arrays(np.atleast_1d(theta), np.atleast_1d(y))
    d = np.stack([0*theta, np.sin(theta), np.cos(theta)], axis=-1)
    points = np.stack([0*y, y, -np.ones_like(y)], axis=-1)
    return Rays(points, d, np.zeros(len(y)))


def single_surface(coating=None, n=1.5):
    return System([Surface(n_after=n, thickness=5.0, semi_diameter=50.0, coating=coating)], 0, WL)


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.8, 1.2])
def test_bare_interface_is_fresnel(theta):
    traced = single_surface().trace_polarization(rays(theta))
    for E, pol in ((S, "s"), (P, "p")):
        R, T, _ = thin_films.R_T_A_coeffs(theta, [1.5], [0.0], [0.0], 1.0, 0.0, 1.5, 0.0, pol, WL)
        assert np.isclose(traced.transmittance(E)[0], T)
    J = traced.jones()[0]
    assert abs(J[0, 1]) < 1e-14 and abs(J[1, 0]) < 1e-14


def test_coated_interface_matches_thin_films():
    theta = np.linspace(0.0, 1.0, 7)
    layers = [Layer(2.1, 70e-6), Layer(1.38, 100e-6)]
    coating = Coating(layers)
    traced = single_surface(coating).trace_polarization(rays(theta))
    power = coating.power(theta, 1.0, 1.5, WL)
    for E, pol in ((S, "s"), (P, "p")):
        T = [thin_films.R_T_A_coeffs(a, [2.1, 1.38], [0, 0], [70e-6, 100e-6], 1.0, 0.0, 1.5, 0.0,
                                     pol, WL)[1] for a in theta]
        assert np.allclose(traced.transmittance(E), T)
        assert np.allclose(power["T_" + pol], T)
        assert np.allclose(power["R_" + pol] + power["T_" + pol], 1.0)  # lossless
    # a quarter wave MgF2 layer is a better AR coating than none:
    ar = Coating([Layer.quarter_wave(1.38, WL)])
    T_ar = single_surface(ar).trace_polarization(rays(0.0)).transmittance()[0]
    T_bare = single_surface().trace_polarization(rays(0.0)).transmittance()[0]
    assert T_ar > T_bare + 0.02 and np.isclose(T_bare, 0.96)


def test_thickness_profile_follows_the_hit_points():
    t0 = 100e-6
    profile = RadialProfile([0.2], r_norm=10.0)
    coating = Coating([Layer(1.38, t0, profile)])
    y = np.array([0.0, 5.0, 10.0])
    traced = single_surface(coating).trace_polarization(rays(0.0, y))
    for yi, Ti in zip(y, traced.transmittance()):
        T = thin_films.R_T_A_coeffs(0.0, [1.38], [0.0], [t0*(1 + 0.2*(yi/10)**2)], 1.0, 0.0, 1.5,
                                    0.0, "s", WL)[1]
        assert np.isclose(Ti, T)
    assert len(set(np.round(traced.transmittance(), 12))) == 3


def test_profiles():
    x, y = np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 2.0])
    assert np.allclose(RadialProfile([0.1, 0.01], r_norm=2.0)(x, y), [1, 1.11, 1.11])
    assert np.allclose(PolynomialProfile({(1, 0): 0.05}, r_norm=2.0)(x, y), [1, 1.05, 1])
    assert np.allclose(ZernikeProfile({3: 0.1}, r_norm=2.0)(x, y), [1, 1, 1.1])  # Z_3 = y


def test_perfect_and_metal_mirror():
    def mirror(coating, tilt=0.0):
        return System([Surface(mirror=True, thickness=-5.0, semi_diameter=50.0, coating=coating,
                               tilt=(tilt, 0, 0))], 0, WL)

    J = mirror(None).trace_polarization(rays(0.0)).jones()[0]
    assert np.allclose(J, -np.eye(2))
    aluminium = 0.96 + 6.69j
    coating = Coating([], substrate=aluminium)
    traced = mirror(coating, np.pi/8).trace_polarization(rays(0.0))
    J = traced.jones()[0]
    power = coating.power(np.pi/8, 1.0, wavelength=WL)
    assert abs(J[0, 1]) < 1e-14 and abs(J[1, 0]) < 1e-14
    assert np.isclose(abs(J[0, 0])**2, power["R_s"]) and np.isclose(abs(J[1, 1])**2, power["R_p"])
    assert 0.9 < power["R_p"] < power["R_s"] < 1
    with pytest.raises(ValueError):
        mirror(Coating([Layer(1.38, 1e-4)])).trace_polarization(rays(0.0))


def plate(material, L):
    return System([Surface(material=material, thickness=L, semi_diameter=50.0),
                   Surface(n_after=1.0, thickness=5.0)], 0, WL)


def test_quarter_wave_plate():
    dn = 0.009
    L = WL/(4*dn)
    J = plate(Uniaxial(1.54, 1.54 + dn, axis=(1, 0, 0)), L).trace_polarization(rays(0.0)).jones()[0]
    assert abs(J[0, 1]) < 1e-14 and abs(J[1, 0]) < 1e-14
    assert np.isclose(np.angle(J[0, 0]/J[1, 1]), np.pi/2)  # x is the slow (extraordinary) axis
    assert np.isclose(abs(J[0, 0]), abs(J[1, 1]))
    # along the optic axis there is no retardance:
    J = plate(Uniaxial(1.54, 1.54 + dn), L).trace_polarization(rays(0.0)).jones()[0]
    assert np.isclose(J[0, 0], J[1, 1])
    # tilted rays see part of it:
    J = plate(Uniaxial(1.54, 1.54 + dn), L).trace_polarization(rays(0.5)).jones()[0]
    assert 0 < abs(np.angle(J[1, 1]/J[0, 0])) < np.pi/2


def test_dichroism_and_absorption():
    L, k = 2.0, 1e-5
    decay = np.exp(-4*np.pi*k*L/WL)
    traced = plate(Uniaxial(1.5, 1.5, axis=(1, 0, 0), k_e=k), L).trace_polarization(rays(0.0))
    assert np.isclose(traced.transmittance((1, 0))[0]/traced.transmittance((0, 1))[0], decay)
    traced = plate(Isotropic(1.5, k), L).trace_polarization(rays(0.0))
    fresnel = (1 - 0.04)**2
    assert np.isclose(traced.transmittance()[0], fresnel*decay)
    power = traced.power()[:, 0]
    assert np.allclose(power[:2], [1, 0.96]) and np.isclose(power[2], 0.96**2*decay)


def test_crystal_as_material_and_as_medium():
    crystal = CubicCrystal(-3e-6, "100", n=1.5)
    J1 = plate(crystal, 5.0).trace_polarization(rays(0.4)).jones()[0]
    system = System([Surface(n_after=1.5, thickness=5.0, semi_diameter=50.0),
                     Surface(n_after=1.0, thickness=5.0)], 0, WL)
    J2 = system.trace_polarization(rays(0.4), media={0: CubicCrystal(-3e-6, "100")}).jones()[0]
    assert np.allclose(J1, J2)
    assert abs(np.angle(J1[1, 1]/J1[0, 0])) > 1e-3  # birefringent off the [100] axis
    with pytest.raises(ValueError):
        Surface(material=CubicCrystal(-3e-6)).index(550.0, 1.0)


def test_ideal_lens_is_lossless():
    system = System([IdealLens(20.0, thickness=20.0, semi_diameter=5.0)], 0, WL)
    traced = system.trace_pupil_polarization(0.0, np.linspace(-0.7, 0.7, 9), np.linspace(0, 0.7, 9))
    assert np.allclose(traced.transmittance(), 1.0)
    J = traced.jones()
    assert np.allclose(np.einsum("nji,njk->nik", J.conj(), J), np.eye(2), atol=1e-12)
    # an aplanatic focus keeps x polarization at the pupil rim along x:
    assert np.allclose(J[:, 0, 1], 0, atol=1e-12) and np.allclose(J[:, 1, 0], 0, atol=1e-12)


def test_power_decreases_through_an_uncoated_lens():
    lens = System([Surface(50.0, 5.0, n_after=1.5, semi_diameter=10.0),
                   Surface(-50.0, 45.0, n_after=1.0)], 0, WL, field_angle=0.05)
    traced = lens.trace_pupil_polarization(None, np.linspace(-1, 1, 5), 0.0)
    power = traced.power()
    assert np.all(np.diff(power, axis=0) <= 1e-15)
    assert np.allclose(power[-1], power[2]) and np.all(power[-1] < 0.93)
    field = traced.field((1, 0))
    d = lens.to_image_frame(traced.bundle.points[-1], traced.bundle.directions[-1])[1]
    assert np.allclose(np.sum(field*d, axis=-1), 0, atol=1e-14)  # transverse


def test_expm_2x2():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 2, 2)) + 1j*rng.normal(size=(5, 2, 2))
    A = np.concatenate([A, [[[0, 1], [0, 0]], [[2j, 0], [0, 2j]]]])
    assert np.allclose(expm_2x2(A), [expm(a) for a in A])


# quarter waves at oblique incidence and the vectorial focus:

def test_quarter_wave_at_oblique_incidence():
    n, angle = 2.1, np.pi/4
    layer = Layer.quarter_wave(n, WL, angle=angle)
    cos_layer = np.sqrt(1 - (np.sin(angle)/n)**2)
    assert np.isclose(n*layer.thickness*cos_layer, WL/4)
    assert np.isclose(Layer.quarter_wave(n, WL, angle=0.0).thickness, WL/(4*n))
    # a quarter wave stack reflects s and p with real coefficients at its design angle:
    H, L = Layer.quarter_wave(2.1, WL, angle=angle), Layer.quarter_wave(1.45, WL, angle=angle)
    stack = Coating([H, L]*6 + [H], substrate=1.5)
    r_s, r_p = stack.amplitudes(np.cos(angle), 1.0, None, WL)[:2]
    assert abs(r_s.imag) < 1e-12 and abs(r_p.imag) < 1e-12
    assert abs(r_s)**2 > 0.99 and abs(r_p)**2 > 0.9


def _fold(f=10.0, NA=0.1, coating=None):
    from pyoptics.raytrace import CoordinateBreak

    D = 2*f*np.tan(np.arcsin(NA))
    return System([IdealLens(f, thickness=f/2, semi_diameter=D/2),
                   CoordinateBreak(tilt=(np.pi/4, 0, 0)),
                   Surface(mirror=True, semi_diameter=5.0, coating=coating),
                   CoordinateBreak(tilt=(np.pi/4, 0, 0), thickness=-f/2)], 0, WL)


def test_vectorial_focus_is_richards_wolf():
    from pyoptics.fft import inv_FT
    from pyoptics.propagators import vectorial_mansuripur
    from pyoptics.utils import freq_grid, grid1d

    f, a = 10.0, 3.0
    lens = System([IdealLens(f, thickness=f, semi_diameter=a)], 0, WL)
    x = grid1d(-40*WL, 40*WL, 128)
    k = 2*np.pi/WL
    K_x, K_y = freq_grid(x, x, wavenumbers=True, normal_order=True)
    pupil = (K_x**2 + K_y**2 < (k*a/np.hypot(a, f))**2).astype(float)
    S_z = np.sqrt(np.clip(1 - (K_x**2 + K_y**2)/k**2, 1e-12, None))
    norm = np.abs(inv_FT(pupil/np.sqrt(S_z))).max()
    for z, E_in in ((0.0, (1, 0)), (3*WL, (0.6, 0.8j))):
        E = lens.vectorial_focus(x, x, z, field=0.0, E_in=E_in, N=64)
        field = np.stack([inv_FT(pupil*E_in[0]), inv_FT(pupil*E_in[1])], -1)
        E_rw = vectorial_mansuripur(field, x, x, z, WL)[0]/norm
        assert np.abs(E - E_rw).max() < 1e-4
    with pytest.raises(ValueError):
        lens.vectorial_focus(x, x, field=0.0, normalization="power")


def test_vectorial_focus_behind_a_fold_mirror():
    from pyoptics.utils import grid1d

    x = grid1d(-30*WL, 30*WL, 128)
    lens = System([IdealLens(10.0, thickness=10.0, semi_diameter=_fold().surfaces[0].semi_diameter)],
                  0, WL)
    E_lens = lens.vectorial_focus(x, x, field=0.0, E_in=(1, 0))
    E_fold = _fold().vectorial_focus(x, x, field=0.0, E_in=(1, 0))  # perfect mirror
    assert np.allclose(np.abs(E_fold[..., 0]), np.abs(E_lens[..., 0]), atol=1e-6)
    assert np.isclose(np.sum(np.abs(E_fold)**2), np.sum(np.abs(E_lens)**2))
    # a silver mirror reflects less, but leaves the shape:
    silver = Coating([], substrate=0.1 + 7.0j)
    E_ag = _fold(coating=silver).vectorial_focus(x, x, field=0.0, E_in=(1, 0))
    R_s = silver.power(np.pi/4, 1.0, wavelength=WL)["R_s"]
    assert np.isclose(np.sum(np.abs(E_ag)**2)/np.sum(np.abs(E_fold)**2), R_s, rtol=2e-3)


def test_vectorial_focus_shows_the_goos_haenchen_shift():
    from pyoptics.utils import grid1d

    # p polarized light on a dielectric mirror: the focus moves by Artmann's
    # D = -lambda/(2 pi) dphi/dtheta in the plane of incidence
    angle, NA = np.pi/4, 0.03
    H, L = Layer.quarter_wave(2.1, WL, angle=angle), Layer.quarter_wave(1.45, WL, angle=angle)
    stack = Coating([H, L]*10 + [H], substrate=1.5)
    h = 1e-5
    r_1, r_2 = (stack.amplitudes(np.cos(angle + d), 1.0, None, WL)[1] for d in (-h, h))
    D_p = -WL/(2*np.pi)*np.angle(r_2/r_1)/(2*h)
    x = grid1d(-4*WL/NA, 4*WL/NA, 128)
    X, Y = np.meshgrid(x, x)
    airy = np.hypot(X, Y) < 0.61*WL/NA
    y_c = []
    for system in (_fold(NA=NA, coating=stack), _fold(NA=NA)):
        I = np.sum(np.abs(system.vectorial_focus(x, x, field=0.0, E_in=(0, 1)))**2, -1)*airy
        y_c.append(np.sum(I*Y)/np.sum(I))
    assert abs(D_p) > 0.1*WL
    assert np.isclose(y_c[0] - y_c[1], D_p, rtol=0.02)
