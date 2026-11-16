"""Diffractive (kinoform) surfaces in raytrace: local grating equation, paraxial
power, Seidel sums against the Sweatt model, colour, efficiency and the
optimizer variables (notebook 151, hybrid diffractive lenses)."""

import numpy as np
import pytest

from pyoptics import fresnel
from pyoptics import lens_optimization as lo
from pyoptics import raytrace as rt

WL_D, WL_F, WL_C = 587.56e-6, 486.13e-6, 656.27e-6


def doe_lens(f=100.0, wl=WL_D, rn=10.0, extra=(), **kwargs):
    s = rt.DiffractiveSurface.from_power(1/f, wl, thickness=f, norm_radius=rn,
                                         semi_diameter=rn, **kwargs)
    s.phase = s.phase + tuple(extra)
    return s


def test_from_power_and_wavelength_scaling():
    s = rt.System([doe_lens(100.0)], 0, WL_D)
    p = s.paraxial()
    assert p.efl == pytest.approx(100.0, rel=1e-12)
    # f is inversely proportional to the wavelength:
    for wl in (WL_F, WL_C, 1e-3):
        assert s.with_wavelength(wl).paraxial().efl == pytest.approx(100.0*WL_D/wl, rel=1e-12)
    # the ABCD matrix agrees with the paraxial trace:
    M = s.abcd_matrix()
    assert M[1, 0] == pytest.approx(-0.01, rel=1e-12)
    assert s.surface_powers()[0] == pytest.approx(0.01, rel=1e-12)


def test_local_grating_equation_plane():
    A1, rn, m = -400.0, 5.0, 2
    s = rt.DiffractiveSurface(np.inf, 50.0, phase=(A1, 3.0), norm_radius=rn, order=m,
                              semi_diameter=5.0)
    sys = rt.System([s], 0, WL_D)
    py = np.linspace(0, 1, 7)
    b = sys.trace_pupil(0.0, 0.0, py)
    y = b.points[1][:, 1]
    dphi = (2*A1*y/rn**2 + 4*3.0*y**3/rn**4)
    np.testing.assert_allclose(b.directions[-1][:, 1], m*WL_D*dphi/(2*np.pi), atol=1e-14)
    # the optical path gains m lambda phi/(2 pi) at the surface:
    phi = A1*(y/rn)**2 + 3.0*(y/rn)**4
    path = b.opl[1] - b.opl[0] - (b.points[1][:, 2] - b.points[0][:, 2])
    np.testing.assert_allclose(path, m*WL_D*phi/(2*np.pi), atol=1e-12)


def test_evanescent_order_is_lost():
    s = rt.DiffractiveSurface(np.inf, 10.0, phase=(-2e5,), norm_radius=10.0, semi_diameter=10.0)
    b = rt.System([s], 0, WL_D).trace_pupil(0.0, 0.0, np.array([0.01, 1.0]))
    assert np.isfinite(b.directions[-1][0]).all()
    assert not np.isfinite(b.directions[-1][1]).any()


def test_perfect_doe_lens_has_no_opd():
    """phi = -k (sqrt(r**2 + f**2) - f) expanded to r**10 focuses on axis."""

    f, rn = 100.0, 10.0
    k = 2*np.pi/WL_D
    # sqrt(r^2 + f^2) - f = r^2/(2f) - r^4/(8f^3) + r^6/(16 f^5) - 5 r^8/(128 f^7) + 7 r^10/(256 f^9)
    coeff = [1/(2*f), -1/(8*f**3), 1/(16*f**5), -5/(128*f**7), 7/(256*f**9)]
    phase = tuple(-k*c*rn**(2*j) for j, c in enumerate(coeff, start=1))
    s = rt.DiffractiveSurface(np.inf, f, phase=phase, norm_radius=rn, semi_diameter=rn)
    sys = rt.System([s], 0, WL_D)
    assert sys.rms_opd(0.0) < 1e-4
    # the r**2 term alone has spherical aberration, as the Seidel sum says:
    s1 = rt.System([doe_lens(f)], 0, WL_D)
    w040 = s1.seidel(0.0).W040
    z = s1.zernike(0.0, n_terms=9)
    # Fringe Z9 = 6 rho^4 - 6 rho^2 + 1, W040 rho^4 -> Z9 coefficient W040/6
    assert z[8] == pytest.approx(w040/6, rel=0.05)


@pytest.mark.parametrize("R2", [np.inf, -200.0, 80.0])
def test_seidel_matches_sweatt_limit(R2):
    S = [rt.Surface(60.0, 5.0, material=(1.5168, 64.17), semi_diameter=10.0),
         rt.DiffractiveSurface(R2, 90.0, phase=(-300.0, 20.0), norm_radius=10.0, n_after=1.0,
                               semi_diameter=10.0)]
    sys = rt.System(S, 0, WL_D, field_angle=np.radians(5))
    sd = sys.seidel().total
    errors = [np.abs(sys.sweatt_model(K).seidel().total - sd).max() for K in (1e3, 1e4)]
    scale = np.abs(sd).max()
    assert errors[1] < 2e-4*scale
    assert errors[1] < 0.2*errors[0]  # converges like 1/N
    assert sd[3] == pytest.approx(rt.System(S[:1] + [rt.Surface(R2, 90.0, n_after=1.0)], 0, WL_D,
                                            field_angle=np.radians(5)).seidel().total[3])
    cs = sys.chromatic_seidel(WL_C - WL_F).total
    np.testing.assert_allclose(sys.sweatt_model(1e5).chromatic_seidel(WL_C - WL_F).total, cs,
                               rtol=1e-4)


def test_sweatt_rays_converge_to_doe_rays():
    S = [rt.Surface(60.0, 5.0, material=(1.5168, 64.17), semi_diameter=10.0),
         rt.DiffractiveSurface(80.0, 90.0, phase=(-300.0, 20.0), norm_radius=10.0, n_after=1.0,
                               semi_diameter=10.0)]
    sys = rt.System(S, 0, WL_D, field_angle=np.radians(5))
    py = np.linspace(-1, 1, 9)
    b = sys.trace_pupil(np.radians(5), 0.0, py)
    for K, tol in ((1e3, 1e-5), (1e4, 1e-6)):
        bs = sys.sweatt_model(K).trace_pupil(np.radians(5), 0.0, py)
        assert np.abs(bs.directions[-1] - b.directions[-1]).max() < tol


def test_chromatic_seidel_against_real_focus():
    s = rt.System([doe_lens(100.0, rn=5.0)], 0, WL_D)
    band = 10e-6
    cs = s.chromatic_seidel(band, field=0.0)
    dz = (s.with_wavelength(WL_D - band/2).paraxial().image_distance
          - s.with_wavelength(WL_D + band/2).paraxial().image_distance)
    assert cs.axial == pytest.approx(dz, rel=1e-3)
    # f = f_0 lambda_0/lambda; to first order delta f = -f/V with V = -3.45:
    dz = (s.with_wavelength(WL_F).paraxial().image_distance
          - s.with_wavelength(WL_C).paraxial().image_distance)
    assert dz == pytest.approx(100.0*WL_D*(1/WL_F - 1/WL_C), rel=1e-9)
    assert dz == pytest.approx(-100.0/fresnel.diffractive_abbe_number(), rel=0.1)


def test_thin_hybrid_achromat():
    """A thin refractive lens and a diffractive surface with the powers of
    an achromat bring F and C to one focus."""

    n_d, V_d, f = 1.5168, 64.17, 100.0
    V_doe = fresnel.diffractive_abbe_number()
    phi = 1/f
    phi_r, phi_d = phi*V_d/(V_d - V_doe), phi*V_doe/(V_doe - V_d)
    c = phi_r/(n_d - 1)/2
    S = [rt.Surface(1/c, 0.0, material=(n_d, V_d), semi_diameter=5.0),
         rt.DiffractiveSurface.from_power(phi_d, WL_D, R=-1/c, thickness=f, n_after=1.0,
                                          norm_radius=5.0, semi_diameter=5.0)]
    s = rt.System(S, 0, WL_D)
    assert s.paraxial().efl == pytest.approx(f, rel=1e-6)
    shift = (s.with_wavelength(WL_F).paraxial().image_distance
             - s.with_wavelength(WL_C).paraxial().image_distance)
    assert abs(shift) < 1e-3
    # the primary colour of the Seidel sums, over the band d -+ (C - F)/2:
    band = WL_C - WL_F
    dz = (s.with_wavelength(WL_D - band/2).paraxial().image_distance
          - s.with_wavelength(WL_D + band/2).paraxial().image_distance)
    assert s.chromatic_seidel(band, field=0.0).axial == pytest.approx(dz, abs=0.02)


def test_efficiency_and_depth():
    S = [rt.Surface(50.0, 3.0, material=(1.5168, 64.17), semi_diameter=5.0),
         rt.DiffractiveSurface(np.inf, 90.0, phase=(-100.0,), norm_radius=5.0, n_after=1.0,
                               semi_diameter=5.0, design_wavelength=WL_D)]
    s = rt.System(S, 0, WL_C)
    n0 = s.with_wavelength(WL_D).n[1]
    assert s.kinoform_depth(1) == pytest.approx(WL_D/(n0 - 1))
    assert s.with_wavelength(WL_D).diffraction_efficiency() == pytest.approx(1.0)
    wls = np.array([WL_F, WL_D, WL_C])
    eta = s.diffraction_efficiency(wls)
    n = np.array([s.with_wavelength(w).n[1] for w in wls])
    np.testing.assert_allclose(eta, fresnel.kinoform_efficiency(1, wls, WL_D, n, n0))
    assert eta[0] < 1 and eta[2] < 1
    # a harmonic (multi-order) lens: depth of 3 waves, used in the order 3
    S[1] = rt.DiffractiveSurface(np.inf, 90.0, phase=(-100.0/3,), norm_radius=5.0, n_after=1.0,
                                 semi_diameter=5.0, design_wavelength=WL_D, order=3)
    s3 = rt.System(S, 0, WL_D)
    assert s3.kinoform_depth(1) == pytest.approx(3*WL_D/(n0 - 1))
    assert s3.paraxial().efl == pytest.approx(rt.System(
        [S[0], rt.DiffractiveSurface(np.inf, 90.0, phase=(-100.0,), norm_radius=5.0, n_after=1.0,
                                     semi_diameter=5.0)], 0, WL_D).paraxial().efl)


def test_zones_and_local_period():
    A1, rn = -500.0, 4.0
    s = rt.DiffractiveSurface(np.inf, 1.0, phase=(A1,), norm_radius=rn, semi_diameter=4.0)
    r = s.zone_radii()
    j = np.arange(1, len(r) + 1)
    np.testing.assert_allclose(r, rn*np.sqrt(2*np.pi*j/abs(A1)), rtol=1e-6)
    assert len(r) == int(abs(A1)/(2*np.pi))
    # local period 2 pi/|dphi/dr| = pi rn^2/(|A1| r)
    assert s.local_period(2.0) == pytest.approx(np.pi*rn**2/(abs(A1)*2.0))
    assert np.isinf(s.local_period(0.0))


def test_diffractive_mirror():
    s = rt.DiffractiveSurface(np.inf, -100.0, phase=(-50.0,), norm_radius=5.0, mirror=True,
                              semi_diameter=5.0)
    sys = rt.System([s], 0, WL_D)
    p = sys.paraxial()
    b = sys.trace_pupil(0.0, 0.0, np.array([0.05]))
    # rays go back (-z) and cross the axis at the paraxial image distance:
    d = b.directions[-1][0]
    P = b.points[1][0]
    assert d[2] < 0
    z_cross = P[2] - P[1]*d[2]/d[1]
    assert z_cross == pytest.approx(p.image_distance, rel=1e-3)
    assert np.isnan(sys.seidel(0.0).S[0]).all()


def test_diffractive_phase_variable_and_dls():
    """Correct the axial colour of a singlet with the diffractive power."""

    f = 100.0
    S = [rt.Surface(51.68, 4.0, material=(1.5168, 64.17), semi_diameter=10.0),
         rt.DiffractiveSurface(np.inf, f, n_after=1.0, phase=(0.0,), norm_radius=10.0,
                               semi_diameter=10.0)]
    s = rt.System(S, 0, WL_D)
    v = lo.DiffractivePhase(1, 1, 10.0)
    assert v.get(s) == 0.0
    v.set(s, 12.5)
    assert s.surfaces[1].phase == (12.5,)
    v2 = lo.diffractive(1, 5.0, terms=(1, 2))
    assert v2[1].get(s) == 0.0 and v2[0].get(s) == pytest.approx(12.5/4)
    v2[1].set(s, 1.0)
    assert s.surfaces[1].phase[1] == pytest.approx(16.0)

    def f_minus_c(system):
        return lo.focal_shift(WL_F, WL_C, tolerance=1.0)(system)[0]

    s = rt.System(S, 0, WL_D)
    cons = [lo.Constraint(f_minus_c, 0.0, 0.0, 1e-5, "F - C")]
    opt = lo.DampedLeastSquares(s, [lo.Curvature(0), v], [lo.paraxial_target("efl", f, 1e-3)],
                                cons, solves=[lo.solve_paraxial_focus()])
    r = opt.run()
    sys = r.system
    assert sys.paraxial().efl == pytest.approx(f, rel=1e-3)
    assert abs(f_minus_c(sys)) < 1e-3
    # the diffractive part takes about V_doe/(V_doe - V) of the power:
    V_doe = fresnel.diffractive_abbe_number()
    share = sys.surfaces[1].diffractive_power(WL_D)*f
    assert share == pytest.approx(V_doe/(V_doe - 64.17), rel=0.05)


def test_power_needs_wavelength():
    s = rt.DiffractiveSurface(np.inf, 1.0, phase=(-1.0,))
    with pytest.raises(ValueError):
        s.paraxial_power(1.0, 1.0)
    with pytest.raises(ValueError):
        s.deflect(np.zeros((1, 3)), np.array([[0, 0, 1.0]]), 1.0, 1.0)
