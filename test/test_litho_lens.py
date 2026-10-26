#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check pyoptics.litho.lens: ray traced lens aberrations in lithographic
imaging, Zernike sensitivities and the scanner slit.

References: the Zernike fit against the ray traced OPD and the PSF of the
lens, analytic image shifts of Zernike tilts and defocus, parities of the
Zernikes for symmetric features, and the mirror symmetry of the lens.
"""

import numpy as np
import pytest

from pyoptics.litho import (Illumination, LithoSimulation, MaskLayout, Projector, ThinMask,
                            lens, metrics)
from pyoptics.raytrace import caldwell_krf_lens


WL = 248.3
NA = 0.56


@pytest.fixture(scope="module")
def caldwell():
    return caldwell_krf_lens()


def lines_sim(pitch=500.0, cd=250.0, zernikes=None, sigma=0.5):
    return LithoSimulation(MaskLayout.lines_spaces(pitch, cd), ThinMask.binary(),
                           Illumination.conventional(sigma, n_samples=15),
                           Projector(NA, WL, zernikes=zernikes))


# lens to projector:

def test_lens_na_and_projector(caldwell):
    assert np.isclose(lens.lens_NA(caldwell), 0.56, atol=0.005)
    P = lens.projector_from_system(caldwell, (5.0, 2.0), n_terms=16, N=32)
    assert np.isclose(P.wavelength, WL, rtol=1e-6)
    assert np.isclose(P.magnification, 4.0, rtol=1e-3)
    assert set(P.zernikes) == set(range(2, 17))


def test_on_axis_is_rotationally_symmetric(caldwell):
    ab = lens.lens_aberrations(caldwell, (0.0, 0.0), n_terms=25, N=32)
    # non-radial terms vanish up to the sampling of the pupil by a square grid of rays:
    odd = [j for j in ab.zernikes if j not in (4, 9, 16, 25)]
    assert max(abs(ab.zernikes[j]) for j in odd) < 2e-3
    assert abs(ab.zernikes[9]) > 0.01  # residual spherical aberration of the design
    assert np.allclose(ab.distortion, 0, atol=1e-6) and np.allclose(ab.telecentricity, 0, atol=1e-9)


def test_fit_residual_is_small(caldwell):
    ab = lens.lens_aberrations(caldwell, (11.0, 4.0), n_terms=36, N=32)
    assert ab.rms_residual < 0.01 < max(abs(c) for j, c in ab.zernikes.items() if j > 3)


def test_mirror_symmetry_of_the_field(caldwell):
    a = lens.lens_aberrations(caldwell, (8.0, 3.0), n_terms=16, N=24)
    b = lens.lens_aberrations(caldwell, (-8.0, 3.0), n_terms=16, N=24)
    # x -> -x flips the sign of the terms odd in x (cos(m phi) with m odd, sin(m phi) with m even)
    for j in range(2, 17):
        sign = -1 if j in (2, 6, 7, 10, 13, 14) else 1
        assert np.isclose(b.zernikes[j], sign*a.zernikes[j], atol=1e-6)
    assert np.isclose(b.distortion[0], -a.distortion[0]) and np.isclose(b.distortion[1], a.distortion[1])


def test_tilt_matches_the_psf_of_the_lens(caldwell):
    """The Zernike tilts move the image away from the chief ray: compare
    with the centroid of the ray traced PSF."""

    f = (0.0, 4.0)
    ab = lens.lens_aberrations(caldwell, f, n_terms=36, N=48)
    x = np.linspace(-1e-3, 1e-3, 81)
    I = np.abs(caldwell.psf(x, x, field=np.array(f)/caldwell.paraxial().magnification, N=48))**2
    X, Y = np.meshgrid(x, x)
    c_y = (I*Y).sum()/I.sum()*1e6
    assert np.isclose(c_y, -ab.zernikes[3]*WL/ab.NA, rtol=0.1, atol=5.0)


# feature response:

def test_tilt_shifts_the_image():
    c = 0.1
    r = lens.feature_response(lines_sim(zernikes={2: c}))
    assert np.isclose(r.placement, -c*WL/NA, rtol=1e-3)
    assert np.isclose(r.cd, lens.feature_response(lines_sim()).cd, atol=0.05)


@pytest.mark.parametrize("c", [0.05, -0.03])
def test_defocus_zernike_moves_best_focus(c):
    """Three beams (coherent, orders at P = 0, +-f): Z4 = c*(2 rho^2 - 1)
    changes the phase of the first orders by 2*pi*2*c*f^2, which a focus z
    compensates with k*z*(1 - sqrt(1 - NA^2 f^2))."""

    f = 0.8
    pitch = WL/(NA*f)
    expected = 2*c*f**2*WL/(1 - np.sqrt(1 - NA**2*f**2))

    def response(z, foci):
        sim = LithoSimulation(MaskLayout.lines_spaces(pitch, pitch/2), ThinMask.binary(),
                              Illumination.coherent(), Projector(NA, WL, zernikes=z))
        return lens.feature_response(sim, foci=foci)

    # (foci within the contrast reversals at about +-300 nm from best focus)
    assert abs(response({}, np.linspace(-250, 250, 11)).best_focus) < 1e-6
    foci = expected + np.linspace(-250, 230, 11)
    assert np.isclose(response({4: c}, foci).best_focus, expected, rtol=0.01)


def test_best_focus_of_a_parabola():
    f = np.linspace(-100, 100, 9)
    assert np.isclose(metrics.best_focus(f, 50 - 1e-3*(f - 17)**2), 17)
    assert np.isnan(metrics.best_focus(f[:2], f[:2]))


def test_sensitivity_parities():
    sens = lens.zernike_sensitivity(lines_sim(pitch=400.0, cd=200.0), terms=[2, 4, 7, 9],
                                    foci=np.linspace(-500, 500, 11))
    L, Q = sens.linear, sens.quadratic
    t = list(sens.terms)
    # even terms don't move symmetric lines, odd terms (x tilt, x coma) do:
    for j in (4, 9):
        assert abs(L["placement"][t.index(j)]) < 1e-3
    for j in (2, 7):
        assert abs(L["placement"][t.index(j)]) > 10
        # ... and change the CD only to second order
        assert abs(L["cd"][t.index(j)]) < 1e-2*abs(L["placement"][t.index(j)])
    # tilt: linear exactly
    assert np.isclose(L["placement"][t.index(2)], -WL/NA, rtol=1e-3)
    # defocus and spherical aberration move best focus:
    assert abs(L["best_focus"][t.index(4)]) > 1000 and abs(L["best_focus"][t.index(9)]) > 100


def test_predict_reproduces_small_aberrations():
    sim = lines_sim(pitch=400.0, cd=200.0)
    foci = np.linspace(-500, 500, 11)
    sens = lens.zernike_sensitivity(sim, terms=[2, 4, 7, 9], foci=foci)
    z = {2: 0.01, 7: -0.015, 9: 0.01}
    pred = lens.predict(sens, z)
    r = lens.feature_response(sim.with_(projector=sim.projector.with_(zernikes=z)), foci=foci)
    assert np.isclose(pred["placement"], r.placement - sens.nominal.placement, rtol=0.05, atol=0.05)
    assert np.isclose(pred["best_focus"], r.best_focus - sens.nominal.best_focus, rtol=0.2, atol=2.0)


# slit:

def test_through_slit_symmetry_and_scan_average(caldwell):
    sim = lines_sim(pitch=600.0, cd=300.0)
    foci = np.linspace(-400, 400, 9)
    res = lens.through_slit(sim, caldwell, [-10.0, 0.0, 10.0], foci=foci, n_terms=25, N_pupil=24)
    assert np.isclose(res.cd[0], res.cd[2], atol=0.05)
    assert np.isclose(res.placement[0], -res.placement[2], atol=0.05)
    assert abs(res.placement[1]) < 0.05
    # one scan position gives the static image of that field point:
    ab = res.aberrations[2][0]
    r = lens.feature_response(sim.with_(projector=sim.projector.with_(zernikes=ab.zernikes)),
                              foci=foci)
    assert np.isclose(res.cd[2], r.cd, atol=0.02)  # linear interpolation of the shifted cut
    assert np.isclose(res.placement[2], r.placement + ab.distortion[0], atol=1e-3)
    # scanning averages over y: the averaged image is lower in contrast
    scan = lens.through_slit(sim, caldwell, [10.0], y=[-4.0, 0.0, 4.0], foci=foci, n_terms=25,
                             N_pupil=24)
    assert len(scan.aberrations[0]) == 3 and np.isfinite(scan.cd[0])


def test_through_slit_checks_the_wavelength(caldwell):
    sim = LithoSimulation(MaskLayout.lines_spaces(400.0, 200.0), ThinMask.binary(),
                          Illumination.conventional(0.5), Projector(NA, 193.0))
    with pytest.raises(ValueError):
        lens.through_slit(sim, caldwell, [0.0])


def test_linear_correction_removes_linear_placement():
    fields = [(x, y) for x in (-10.0, 0.0, 10.0) for y in (-4.0, 0.0, 4.0)]
    abs_ = [lens.LensAberrations(f, {}, 0.0, (0.0, 0.0), (0.0, 0.0),
                                 (3 + 2*f[0] - 0.5*f[1], -1 + 4*f[1] + 0.25*f[0]), NA)
            for f in fields]
    corr = lens.linear_correction(abs_)
    for ab in abs_:
        assert np.allclose(corr(*ab.field), ab.placement)


def test_placement_includes_the_tilt(caldwell):
    ab = lens.lens_aberrations(caldwell, (6.0, 2.0), n_terms=16, N=24)
    assert np.isclose(ab.placement[0], ab.distortion[0] - ab.zernikes[2]*WL/ab.NA)
    assert np.isclose(ab.placement[1], ab.distortion[1] - ab.zernikes[3]*WL/ab.NA)


def test_best_focus_of_a_dome():
    f = np.linspace(-500, 500, 21)
    nils = 2.5 - 1e-5*(f - 60)**2
    nils[np.abs(f - 60) > 250] = 0.5  # tails that a global fit would follow
    assert np.isclose(metrics.best_focus(f, nils, n_fit=5), 60)


# telecentricity:

def test_telecentricity_on_axis_and_mirror_symmetry(caldwell):
    coat = lens.quarter_wave_coating(1.5084, caldwell.wavelength)
    t = lens.lens_telecentricity(caldwell, (0.0, 0.0), N=24, coating=coat)
    for v in (t.chief, t.geometric, t.energetic):
        assert np.allclose(v, 0.0, atol=1e-9)
    assert np.isclose(np.sum(t.weight), 1.0) and 0.99 < t.transmission <= 1.0
    a = lens.lens_telecentricity(caldwell, (8.0, 0.0), N=24, coating=coat)
    b = lens.lens_telecentricity(caldwell, (-8.0, 0.0), N=24, coating=coat)
    for key in ("chief", "geometric", "energetic"):
        assert np.allclose(getattr(a, key), np.array(getattr(b, key))*[-1, 1], atol=1e-7)
        assert abs(getattr(a, key)[1]) < 1e-9


def test_chief_ray_telecentricity_matches_lens_aberrations(caldwell):
    t = lens.lens_telecentricity(caldwell, (6.0, 4.0), N=16)
    d = lens.lens_aberrations(caldwell, (6.0, 4.0), n_terms=9, N=16).telecentricity
    # slopes and direction sines agree to second order for mrad angles
    assert np.allclose(t.chief, d, rtol=1e-4, atol=1e-9)


def test_symmetric_illumination_is_centred_on_the_chief_ray(caldwell):
    small = lens.lens_telecentricity(caldwell, (10.0, 0.0), N=48, fill=0.3)
    full = lens.lens_telecentricity(caldwell, (10.0, 0.0), N=48)
    # the lens maps object side directions linearly (sine condition):
    assert abs(small.geometric[0] - small.chief[0]) < 1e-3
    # the stop is decentred against the chief ray off axis:
    assert full.geometric[0] - full.chief[0] < -0.01
    ring = lens.lens_telecentricity(caldwell, (10.0, 0.0), N=48,
                                    fill=lambda u_x, u_y: np.hypot(u_x, u_y) <= 0.3)
    assert np.allclose(ring.geometric, small.geometric, atol=3e-4)


def test_energetic_equals_geometric_without_losses():
    from pyoptics.raytrace import IdealLens, System

    ideal = System([IdealLens(50.0, thickness=100.0, semi_diameter=10.0)], 0, 500e-6,
                   object_distance=100.0, object_height=5.0)
    t = lens.lens_telecentricity(ideal, (-2.0, 1.0), N=24)
    assert np.isclose(t.transmission, 1.0)
    assert np.allclose(t.energetic, t.geometric, atol=1e-12)
    # the chief ray through the centre of the lens (the stop) to the image point
    assert np.allclose(t.chief, np.array([-2.0, 1.0])/100.0, rtol=1e-6)


def test_coating_moves_the_energetic_centroid(caldwell):
    bare = lens.lens_telecentricity(caldwell, (10.0, 0.0), N=32, fill=0.8)
    coat = [(1.415, caldwell.wavelength/(4*1.415)), (1.69, caldwell.wavelength/(4*1.69))]
    coated = lens.lens_telecentricity(caldwell, (10.0, 0.0), N=32, fill=0.8, coating=coat)
    assert np.allclose(bare.geometric, coated.geometric)
    # bare: towards Brewster (outwards, +x); the V coat: inwards
    assert bare.energetic[0] - bare.geometric[0] > 2e-4
    assert coated.energetic[0] - coated.geometric[0] < -5e-4
    assert coated.transmission > 3*bare.transmission


def test_lens_apodization(caldwell):
    apo = lens.lens_apodization(caldwell, (0.0, 0.0), N=24)
    assert np.isclose(apo(0.0, 0.0), 1.0)
    rho = np.array([0.0, 0.5, 0.9])
    a = apo(rho, 0*rho)
    assert np.allclose(apo(0*rho, rho), a, rtol=1e-3)  # rotational symmetry
    assert np.all(a[1:] > 1.0)  # bare: the rim transmits better (p near Brewster)
    assert apo(1.05, 0.0) == 0.0
    NA_P = 1.06*lens.lens_NA(caldwell)
    pupil = lens.lens_apodization(caldwell, (10.5, 0.0), N=32, NA=NA_P, transmission=False)
    assert pupil(0.0, 0.0) == 1.0 and pupil(0.0, 0.85) == 1.0
    # decentred to -x:
    assert pupil(-0.92, 0.0) == 1.0 and pupil(0.90, 0.0) == 0.0
    assert pupil(np.zeros((2, 3)), np.zeros((2, 3))).shape == (2, 3)


def test_decentred_pupil_shifts_a_large_feature_through_focus(caldwell):
    NA_L = lens.lens_NA(caldwell)
    foci = np.linspace(-300, 300, 5)

    def slope(field):
        apo = lens.lens_apodization(caldwell, field, N=32, NA=1.06*NA_L, transmission=False)
        sim = LithoSimulation(MaskLayout.iso_line(2000.0, 8000.0), ThinMask.binary(),
                              Illumination.conventional(0.8/1.06, n_samples=15),
                              Projector(1.06*NA_L, WL, apodization=apo))
        return np.polyfit(foci, lens.feature_response(sim, foci=foci).placement_focus, 1)[0]

    assert abs(slope((0.0, 0.0))) < 1e-5
    # the illumination arrives along the chief ray, but the edges of the
    # feature diffract light to the rim of the pupil, which is clipped
    # asymmetrically at the edge of the field
    t = lens.lens_telecentricity(caldwell, (10.5, 0.0), N=48, fill=0.8)
    assert abs(t.geometric[0] - t.chief[0]) < 1.5e-3
    assert slope((10.5, 0.0)) < -0.01
