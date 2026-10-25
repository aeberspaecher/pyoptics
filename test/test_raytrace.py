#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test sequential ray tracing against paraxial optics (abcd), Snell's law,
Seidel sums and wave optics.
"""

import numpy as np
import pytest

from pyoptics import abcd
from pyoptics import raytrace as rt
from pyoptics.analytical import airy_pattern
from pyoptics.utils import grid1d


WL = 0.55e-3  # mm
N_GLASS = 1.5168


def singlet(R1=51.68, R2=np.inf, d=4.0, n=N_GLASS, sd=10.0, field=0.0, image=None):
    """Singlet in air with the stop at the first surface."""

    surfaces = [rt.Surface(R1, d, n_after=n, semi_diameter=sd),
                rt.Surface(R2, 100.0 if image is None else image, n_after=1.0)]
    system = rt.System(surfaces, 0, WL, field_angle=field)

    return system.at_paraxial_focus() if image is None else system


# paraxial optics:
def test_singlet_matches_abcd_thick_lens():
    R1, R2, d = 60.0, -80.0, 6.0
    system = singlet(R1, R2, d)
    M = abcd.thick_lens(R1, R2, d, N_GLASS)
    assert np.allclose(system.paraxial().matrix, M)
    assert np.allclose(system.abcd_matrix(), M)

    cp = abcd.cardinal_points(M, d)
    p = system.paraxial()
    assert p.efl == pytest.approx(cp.f_prime)
    assert p.bfl == pytest.approx(cp.bfl)
    assert p.ffl == pytest.approx(cp.ffl)
    # the paraxial focus is the rear focal point:
    assert d + p.image_distance == pytest.approx(cp.F_prime)
    assert p.f_number == pytest.approx(cp.f_prime/20.0)


def test_paraxial_trace_matches_abcd_with_mirrors_and_ideal_lenses():
    surfaces = [rt.Surface(40.0, 5.0, n_after=1.6), rt.Surface(-30.0, 20.0, n_after=1.0),
                rt.IdealLens(50.0, 30.0, semi_diameter=5.0),
                rt.Surface(-100.0, -40.0, mirror=True, conic=-0.5),
                rt.Surface(60.0, -10.0, mirror=True)]
    system = rt.System(surfaces, 2, WL)
    assert np.allclose(system.paraxial().matrix, system.abcd_matrix())
    # the unfolded abcd.OpticalSystem images onto the same plane:
    M = system.to_abcd().matrix
    p = system.paraxial()
    assert np.allclose(M, abcd.compose(system.abcd_matrix(),
                                       abcd.free_space(abs(surfaces[-1].thickness))))
    assert p.image_distance == pytest.approx(-system.sigma[-1]*system.n[-1]
                                             * system.abcd_matrix()[0, 0]/system.abcd_matrix()[1, 0])


def test_finite_object_image_and_magnification():
    f, s = 50.0, 150.0
    system = rt.System([rt.IdealLens(f, 1.0, semi_diameter=5.0)], 0, WL, object_distance=s,
                       object_height=2.0).at_paraxial_focus()
    p = system.paraxial()
    assert p.image_distance == pytest.approx(1/(1/f - 1/s))
    assert p.magnification == pytest.approx(-0.5)
    assert p.image_height == pytest.approx(-1.0)
    # a perfect lens at finite conjugates still forms a sharp on-axis image:
    spot = system.spot_diagram(0.0)
    assert spot.rms < 1e-12


def test_exit_pupil_is_image_of_stop():
    # stop 20 mm in front of a lens with f = 50: the exit pupil is the
    # virtual image of the stop, 33.3 mm in front of the lens, magnified 1.67x:
    surfaces = [rt.Surface(np.inf, 20.0, semi_diameter=4.0), rt.IdealLens(50.0, 50.0)]
    system = rt.System(surfaces, 0, WL)
    p = system.paraxial()
    s_prime, m = abcd.image(abcd.thin_lens(50.0), 20.0)
    assert p.xp_z - 20.0 == pytest.approx(s_prime)
    assert p.xp_radius == pytest.approx(4.0*abs(m))
    assert p.ep_z == 0.0 and p.ep_radius == 4.0

    # stop behind the lens: the entrance pupil is its image in object space
    surfaces = [rt.IdealLens(50.0, 20.0), rt.Surface(np.inf, 30.0, semi_diameter=4.0)]
    p = rt.System(surfaces, 1, WL).paraxial()
    s_prime, m = abcd.image(abcd.thin_lens(50.0), 20.0)  # by symmetry
    assert p.ep_z == pytest.approx(-s_prime)
    assert p.ep_radius == pytest.approx(4.0*abs(m))


def test_entrance_pupil_radius_sets_stop():
    surfaces = [rt.IdealLens(50.0, 20.0), rt.Surface(np.inf, 30.0)]
    system = rt.System(surfaces, 1, WL, entrance_pupil_radius=5.0)
    assert system.paraxial().ep_radius == pytest.approx(5.0)


# exact ray tracing:
def test_flat_interface_follows_snell():
    n1, n2 = 1.0, 1.7
    system = rt.System([rt.Surface(np.inf, 10.0, n_after=n2, semi_diameter=5.0)], 0, WL)
    for theta in np.radians([0.0, 10.0, 30.0, 60.0]):
        bundle = system.trace_pupil(theta, 0.0, np.linspace(-1, 1, 7))
        d = bundle.directions[1]
        assert np.allclose(n2*d[:, 1], n1*np.sin(theta))
        assert np.allclose(np.sum(d**2, axis=1), 1.0)


def test_total_internal_reflection_and_misses():
    # glass to air at 50 deg (beyond the critical angle of 41.3 deg):
    system = rt.System([rt.Surface(np.inf, 5.0, n_after=1.5, semi_diameter=5.0),
                        rt.Surface(np.inf, 5.0, n_after=1.0)], 0, WL, n_object=1.5)
    bundle = system.trace_pupil(np.radians(50.0), 0.0, [0.0, 0.5])
    assert not np.any(bundle.valid)
    assert np.all(np.isnan(bundle.points[-1]))

    # a small sphere misses rays beyond its radius:
    surfaces = [rt.Surface(np.inf, 1.0, semi_diameter=10.0), rt.Surface(5.0, 10.0, n_after=1.5)]
    bundle = rt.System(surfaces, 0, WL).trace_pupil(0.0, 0.0, [0.0, 0.3, 0.8])
    assert list(bundle.valid) == [True, True, False]


def test_vignetting():
    surfaces = [rt.Surface(np.inf, 10.0, semi_diameter=5.0),
                rt.Surface(np.inf, 10.0, semi_diameter=5.5), rt.IdealLens(50.0, 50.0)]
    system = rt.System(surfaces, 0, WL)
    assert np.all(system.trace_pupil(0.0, 0.0, np.linspace(-1, 1, 11)).valid)
    bundle = system.trace_pupil(np.radians(5.0), 0.0, np.linspace(-1, 1, 11))
    # rays above y = 5.5 - 10 tan(5 deg) = 4.625 at the stop are clipped:
    assert list(bundle.valid) == list(np.linspace(-5, 5, 11) <= 4.625)
    m = system.opd(np.radians(5.0), 21)
    assert np.array_equal(np.isnan(m.W[:, 10]), m.py > 4.625/5)


def test_intersection_lies_on_asphere():
    s = rt.Surface(30.0, 10.0, n_after=1.5, conic=-0.7, aspheric=(1e-5, -2e-8, 1e-11),
                   semi_diameter=8.0)
    system = rt.System([s], 0, WL)
    bundle = system.trace_pupil(np.radians(10.0), *np.random.default_rng(0).uniform(-0.7, 0.7, (2, 50)))
    Q = bundle.points[1]
    assert np.allclose(Q[:, 2], s.sag(Q[:, 0]**2 + Q[:, 1]**2), atol=1e-13, rtol=0)


def test_ray_aiming_hits_stop():
    # stop between two lenses (pupil aberrations):
    surfaces = [rt.Surface(30.0, 4.0, n_after=1.6), rt.Surface(-200.0, 8.0, n_after=1.0),
                rt.Surface(np.inf, 8.0, semi_diameter=3.0),
                rt.Surface(40.0, 4.0, n_after=1.6), rt.Surface(-60.0, 40.0, n_after=1.0)]
    system = rt.System(surfaces, 2, WL, field_angle=np.radians(15.0))
    px, py = np.array([0.0, 1.0, 0.0, -1.0, 0.5]), np.array([0.0, 0.0, 1.0, -0.0, -0.5])
    bundle = system.trace_pupil(None, px, py)
    stop = bundle.points[3]
    assert np.allclose(stop[:, 0], 3.0*px, atol=1e-10)
    assert np.allclose(stop[:, 1], 3.0*py, atol=1e-10)
    # without aiming, the chief ray misses the centre of the stop at 15 deg:
    system.ray_aiming = False
    assert abs(system.trace_pupil(None, 0.0, 0.0).points[3, 0, 1]) > 1e-3


# wavefronts:
def test_parabolic_mirror_is_perfect_on_axis():
    system = rt.System([rt.Surface(-200.0, -100.0, conic=-1.0, mirror=True, semi_diameter=25.0)],
                       0, WL)
    assert np.nanmax(np.abs(system.opd(0.0, 33).W)) < 1e-6
    assert system.strehl(0.0) == pytest.approx(1.0, abs=1e-10)
    # paraboloid: conic and sphere contributions to S_I cancel
    assert system.seidel().total[0] == pytest.approx(0.0, abs=1e-15)
    # a spherical mirror is not perfect; its W040 matches the Seidel sum:
    sphere = rt.System([rt.Surface(-200.0, -100.0, mirror=True, semi_diameter=10.0)], 0, WL)
    c = sphere.zernike(0.0, 9)
    assert c[8]*6 == pytest.approx(sphere.seidel().W040, rel=1e-2)


def test_refracting_ellipsoid_is_perfect():
    # collimated light focused into glass by an ellipsoid with K = -1/n**2:
    n, R = 1.8, 20.0
    surface = rt.Surface(R, n*R/(n - 1), n_after=n, conic=-1/n**2, semi_diameter=10.0)
    system = rt.System([surface], 0, WL)
    assert np.nanmax(np.abs(system.opd(0.0, 33).W)) < 1e-6


def test_ideal_lens_is_perfect_at_any_field():
    system = rt.System([rt.IdealLens(100.0, 100.0, semi_diameter=10.0)], 0, WL,
                       field_angle=np.radians(10.0))
    for field in np.radians([0.0, 5.0, 10.0]):
        assert np.nanmax(np.abs(system.opd(field, 21).W)) < 1e-8
        spot = system.spot_diagram(field)
        assert spot.rms < 1e-10
    assert system.trace_pupil(None).points[-1, 0, 1] == pytest.approx(100*np.tan(np.radians(10)))


def test_defocus_matches_angular_spectrum_phase():
    # image plane dz behind the focus of a perfect system: the pupil phase is
    # that of the angular spectrum propagator, k n dz (cos(alpha) - 1)
    f, dz, a = 100.0, 0.05, 20.0
    system = rt.System([rt.IdealLens(f, f + dz, semi_diameter=a)], 0, WL)
    m = system.opd(0.0, 41, units="length")
    PX, PY = np.meshgrid(m.px, m.py)
    sin_alpha = np.sin(np.arctan(a*np.hypot(PX, PY)/f))
    expected = dz*(np.sqrt(1 - sin_alpha**2) - 1)
    ok = np.isfinite(m.W)
    assert np.allclose(m.W[ok], expected[ok], atol=1e-3*dz*np.max(sin_alpha)**2)


def test_plano_convex_spherical_aberration_against_seidel():
    for R1, R2 in ((51.68, np.inf), (np.inf, -51.68)):
        system = singlet(R1, R2, 2.0, sd=4.0)
        seidel = system.seidel(0.0)
        m = system.opd(0.0, 41)
        PX, PY = np.meshgrid(m.px, m.py)
        ok = np.isfinite(m.W)
        r2 = PX[ok]**2 + PY[ok]**2
        c = np.linalg.lstsq(np.column_stack([np.ones_like(r2), r2, r2**2]), m.W[ok], rcond=None)[0]
        assert c[2] == pytest.approx(seidel.W040, rel=0.02)
        assert seidel.W040 < 0  # undercorrected

        # thin lens formula (Welford, Aberrations of Optical Systems, 7.4):
        n, h = N_GLASS, 4.0
        c1 = 0 if np.isinf(R1) else 1/R1
        c2 = 0 if np.isinf(R2) else 1/R2
        K = (n - 1)*(c1 - c2)
        B = (c1 + c2)/(c1 - c2)
        C = -1.0  # object at infinity
        S_I = h**4*K**3/4*((n/(n - 1))**2 + (n + 2)/(n*(n - 1)**2)*(B + 2*(n**2 - 1)/(n + 2)*C)**2
                           - n/(n + 2)*C**2)
        assert seidel.total[0] == pytest.approx(S_I, rel=0.05)
    # convex side towards the object is much better:
    assert singlet(51.68, np.inf, 2.0).seidel(0.0).total[0] < \
        0.3*singlet(np.inf, -51.68, 2.0).seidel(0.0).total[0]


def test_field_aberrations_against_seidel():
    # stop 15 mm in front of a meniscus, small aperture and field
    surfaces = [rt.Surface(np.inf, 15.0, semi_diameter=1.0),
                rt.Surface(-40.0, 3.0, n_after=N_GLASS), rt.Surface(-20.0, 80.0, n_after=1.0)]
    field = np.radians(3.0)
    system = rt.System(surfaces, 0, WL, field_angle=field).at_paraxial_focus()
    seidel = system.seidel()
    m = system.opd(field, 41)
    PX, PY = np.meshgrid(m.px, m.py)
    ok = np.isfinite(m.W)
    x, y = PX[ok], PY[ok]
    r2 = x**2 + y**2
    A = np.column_stack([np.ones_like(x), r2, r2**2, r2*y, y**2, y])
    c = np.linalg.lstsq(A, m.W[ok], rcond=None)[0]
    assert c[2] == pytest.approx(seidel.W040, rel=0.05)
    assert c[3] == pytest.approx(seidel.W131, rel=0.05)
    assert c[4] == pytest.approx(seidel.W222, rel=0.05)
    assert c[1] == pytest.approx(seidel.W220, rel=0.05)
    # distortion: the real chief ray misses the paraxial image height by
    # eps = -dW311/drho/(n' u') with the marginal ray slope u':
    p = system.paraxial()
    eps = system.trace_pupil(field).points[-1, 0, 1] - p.image_height
    u = p.marginal[-1, 1]
    assert eps == pytest.approx(-seidel.W311*WL/u, rel=0.05)


def test_asphere_corrects_spherical_aberration():
    system = singlet(51.68, np.inf, 2.0, sd=8.0)
    S_I = system.seidel(0.0).total[0]
    # A4 r**4 on the first surface adds 8 A4 (n' - n) h**4 to S_I:
    A4 = -S_I/(8*(N_GLASS - 1)*8.0**4)
    corrected = system.copy()
    corrected.surfaces[0].aspheric = (A4,)
    assert corrected.seidel(0.0).total[0] == pytest.approx(0.0, abs=1e-12)
    assert corrected.rms_opd(0.0) < 0.05*system.rms_opd(0.0)


def test_zernike_fit_of_defocus():
    f, dz, a = 100.0, 0.02, 5.0
    system = rt.System([rt.IdealLens(f, f + dz, semi_diameter=a)], 0, WL)
    c = system.zernike(0.0, 9)
    # paraxial defocus -n dz NA**2 rho**2/2 = -dz NA**2/4 (2 rho**2 - 1) + piston
    NA = a/np.hypot(a, f)
    assert c[3] == pytest.approx(-dz*NA**2/4/WL, rel=1e-2)
    assert np.allclose(c[[1, 2, 4, 5, 6, 7]], 0, atol=1e-9)


# hand-off to imaging:
def test_perfect_lens_psf_is_airy():
    a, f = 5.0, 100.0
    system = rt.System([rt.Surface(-2*f, -f, conic=-1.0, mirror=True, semi_diameter=a)], 0, WL)
    NA = system.pupil(0.0).NA
    assert NA == pytest.approx(np.sin(np.arctan(a/(f - a**2/(4*f)))), rel=1e-3)
    x = grid1d(-0.1, 0.1, 128)
    psf = system.psf(x, x, 0.0)
    assert np.abs(psf[64, 64])**2 == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(np.abs(psf[64])**2, airy_pattern(x, NA, WL), atol=2e-3)
    assert system.strehl(0.0) == pytest.approx(1.0)


def test_psf_centroid_matches_spot_centroid():
    # coma: the PSF is offset from the chief ray in the same direction and by
    # the same amount as the ray centroid (exact for a uniform pupil)
    surfaces = [rt.Surface(np.inf, 30.0, semi_diameter=3.0), rt.Surface(40.0, 3.0, n_after=1.7),
                rt.Surface(np.inf, 60.0, n_after=1.0)]
    field = np.radians(4.0)
    system = rt.System(surfaces, 0, WL, field_angle=field).at_paraxial_focus()
    spot = system.spot_diagram(field, 61)
    x = grid1d(-0.08, 0.08, 256)
    I = np.abs(system.psf(x, x, field, N=64))**2
    X, Y = np.meshgrid(x, x)
    cy = np.sum(Y*I)/np.sum(I)
    assert abs(spot.centroid[1]) > 2e-3
    assert cy == pytest.approx(spot.centroid[1], rel=0.1)


def test_strehl_matches_imaging():
    from pyoptics import imaging
    system = singlet(51.68, np.inf, 2.0, sd=5.0)
    pupil = system.pupil(0.0, 64)
    x = grid1d(-0.1, 0.1, 128)
    S = imaging.strehl_ratio(x, x, pupil.NA, pupil.k, pupil.mask, pupil.phase)
    assert 0.2 < S < 0.95
    assert S == pytest.approx(system.strehl(0.0), rel=0.03)


def test_dispersion_from_material():
    surfaces = [rt.Surface(50.0, 4.0, material=(N_GLASS, 64.17), semi_diameter=5.0),
                rt.Surface(-50.0, 50.0, n_after=1.0)]
    system = rt.System(surfaces, 0, 587.5618e-6)
    assert system.n[1] == pytest.approx(N_GLASS)
    efl_F = system.with_wavelength(486.1327e-6).paraxial().efl
    efl_C = system.with_wavelength(656.2725e-6).paraxial().efl
    # thin lens: longitudinal colour f/V
    assert efl_C - efl_F == pytest.approx(system.paraxial().efl/64.17, rel=0.02)
