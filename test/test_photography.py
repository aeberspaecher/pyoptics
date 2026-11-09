import numpy as np
import pytest

from pyoptics import photography as ph
from pyoptics import raytrace as rt


@pytest.fixture(scope="module")
def lens():
    return ph.double_gauss()


def test_double_gauss_first_order(lens):
    p = lens.paraxial()
    assert p.efl == pytest.approx(ph.DOUBLE_GAUSS_F, abs=0.01)
    assert p.f_number == pytest.approx(1.8, abs=1e-3)
    assert p.bfl > 35.0
    # undercorrected spherical aberration
    assert lens.seidel().W040 < 0


def test_double_gauss_stopped_down_is_sharp():
    s = ph.focus(ph.double_gauss(4.0))
    assert s.spot_diagram(0.0).rms < 0.006


def test_focus_beats_paraxial_focus(lens):
    s = ph.focus(lens)
    par = lens.at_paraxial_focus()
    assert s.spot_diagram(0.0).rms < par.spot_diagram(0.0).rms
    near = ph.focus(lens, 1000.0)
    assert not near.infinite_object
    assert near.surfaces[-1].thickness > s.surfaces[-1].thickness + 2.0


def test_blur_diameter_limits():
    f, N = 50.0, 2.0
    # focused at infinity, point at s: c = f**2/(N s)
    assert ph.blur_diameter(f, N, np.inf, 1000.0) == pytest.approx(f**2/(N*1000.0))
    assert ph.blur_diameter(f, N, 1000.0, 1000.0) == 0.0
    # focused at s_f, point at infinity: c = f**2/(N (s_f - f))
    assert ph.blur_diameter(f, N, 2000.0, np.inf) == pytest.approx(f**2/(N*(2000.0 - f)))


def test_blur_diameter_pupils_thin_lens_limit():
    f, N, s_f = 50.0, 2.0, 1500.0
    x_s = f**2/(s_f - f)
    for s in (300.0, 800.0, 4000.0, np.inf):
        thin = ph.blur_diameter(f, N, s_f, s)
        assert ph.blur_diameter_pupils(f, f/N, 1.0, x_s, s - f) == pytest.approx(thin)


def test_blur_diameter_pupils_matches_paraxial_trace(lens):
    s = ph.focus(lens, 1000.0)
    p = s.paraxial()
    cp = p.cardinal_points
    m_p = p.xp_radius/p.ep_radius
    x_s = s.vertex_z[-1] + s.surfaces[-1].thickness - cp.F_prime
    for dist in (400.0, 3000.0):
        t = rt.System(s.surfaces, s.stop_index, s.wavelength, object_distance=dist,
                      object_height=1.0)
        pm = t.paraxial()
        c_trace = 2*abs(pm.marginal[-1, 0] + pm.marginal[-1, 1]*t.surfaces[-1].thickness)
        c = ph.blur_diameter_pupils(p.efl, 2*p.ep_radius, m_p, x_s, dist + cp.F)
        assert c == pytest.approx(c_trace, rel=1e-9)


def _ideal(f=50.0, epr=10.0, defocus=1.0):
    S = [rt.Surface(np.inf, 0.0, semi_diameter=epr), rt.IdealLens(f, f + defocus)]
    return rt.System(S, 0, 550e-6, field_angle=0.1)


def test_footprint_of_ideal_lens_is_a_uniform_disk():
    s = _ideal()
    fp = ph.point_footprint(s, 0.0, 40000, np.random.default_rng(1))
    r = np.hypot(fp.x, fp.y)
    assert r.max() == pytest.approx(10.0*1.0/50.0, rel=0.01)
    # power: the pupil area
    assert fp.weight.sum() == pytest.approx(np.pi*10.0**2, rel=0.01)
    # uniform: half the power within 1/sqrt(2) of the radius
    assert np.sum(fp.weight[r < 0.2/np.sqrt(2)])/fp.weight.sum() == pytest.approx(0.5, abs=0.01)
    E = ph.splat(fp, np.linspace(-0.3, 0.3, 61), np.linspace(-0.3, 0.3, 61))
    assert E.sum()*0.01**2 == pytest.approx(fp.weight.sum())


def test_footprint_apodization_mean_transmission():
    s = _ideal()
    T = ph.gaussian_apodizer(0.6)
    rng = np.random.default_rng(2)
    clear = ph.point_footprint(s, 0.0, 40000, rng).weight.sum()
    apod = ph.point_footprint(s, 0.0, 40000, rng, apodization=T).weight.sum()
    assert apod/clear == pytest.approx(ph.apodizer_mean_transmission(0.6), rel=0.02)


def _apertures(angle, finite=False):
    """A stop with an aperture 10 mm in front and another 8 mm behind (no
    refraction: the paraxial vignetting circles are exact)."""

    S = [rt.Surface(np.inf, 10.0, semi_diameter=6.0), rt.Surface(np.inf, 8.0, semi_diameter=4.0),
         rt.Surface(np.inf, 5.0, semi_diameter=5.0), rt.IdealLens(30.0, 30.0)]
    if finite:
        return rt.System(S, 1, 550e-6, object_distance=200.0, object_height=200*np.tan(angle))
    return rt.System(S, 1, 550e-6, field_angle=angle)


@pytest.mark.parametrize("finite", [False, True])
def test_vignetting_circles_without_refraction(finite):
    angle = 0.2
    s = _apertures(angle, finite)
    field = s.max_field
    c, r, k = ph.vignetting_circles(s, field)
    assert list(k) == [0, 1, 2]
    p = s.paraxial()
    assert r[1] == pytest.approx(p.ep_radius)
    if not finite:
        t = np.tan(angle)
        # the front aperture 10 mm in front of the stop (the entrance pupil)
        assert c[0][1] == pytest.approx(10.0*t)
        assert c[2][1] == pytest.approx(-8.0*t)
        assert r[0] == pytest.approx(6.0)
    fraction = ph.vignetted_fraction(c, r, p.ep_radius)[0]
    fp = ph.point_footprint(s, field, 60000, np.random.default_rng(3))
    on_axis = ph.point_footprint(s, 0.0, 60000, np.random.default_rng(4))
    weight = np.cos(angle) if not finite else np.cos(angle)**3
    assert fp.weight.sum()/on_axis.weight.sum() == pytest.approx(weight*fraction, rel=0.02)
    assert fraction < 0.95


def test_relative_illumination_without_vignetting():
    s = _apertures(0.0)
    for surf in (0, 2):
        s.surfaces[surf].semi_diameter = None
    ri = ph.relative_illumination(s, [0.1, 0.2], 40000, np.random.default_rng(5))
    assert ri == pytest.approx(np.cos([0.1, 0.2]), rel=0.01)


def test_defocus_profile_pure_defocus_is_flat():
    r = np.linspace(0.1, 9.9, 50)
    E = ph.defocus_profile(r, -5.0)
    assert E == pytest.approx(np.full(50, 1/(np.pi*10.0**2)))
    assert ph.defocus_profile([10.5], -5.0)[0] == 0.0


@pytest.mark.parametrize("W020,W040", [(-6.0, -2.0), (6.0, -0.5), (10.0, -2.0)])
def test_defocus_profile_matches_ray_histogram(W020, W040):
    rng = np.random.default_rng(6)
    n = 400000
    rho = np.sqrt(rng.uniform(0, 1, n))
    r = np.abs(2*W020*rho + 4*W040*rho**3)
    edges = np.linspace(0, r.max(), 41)
    h = np.histogram(r, edges)[0]/n/(np.pi*(edges[1:]**2 - edges[:-1]**2))
    mid = (edges[1:] + edges[:-1])/2
    E = ph.defocus_profile(mid, W020, W040)
    # away from the caustic (bins that contain it average the divergence)
    ok = (np.abs(np.gradient(E)) < 0.2*E/np.gradient(mid)) & (mid > 0.1*r.max())
    assert np.mean(ok) > 0.6
    assert h[ok] == pytest.approx(E[ok], rel=0.08)
    # normalization
    rr = np.linspace(1e-4, r.max(), 20001)
    if W020*W040 > 0 or abs(W020) > 6*abs(W040):
        assert np.trapezoid(2*np.pi*rr*ph.defocus_profile(rr, W020, W040), rr) == \
            pytest.approx(1.0, rel=0.01)


def test_defocus_profile_apodized():
    T = ph.gaussian_apodizer(0.5)
    r = np.linspace(0.05, 7.95, 40)
    E = ph.defocus_profile(r, 4.0, apodization=T)
    rho = r/8.0
    expected = T(rho)/(np.pi*64*ph.apodizer_mean_transmission(0.5))
    assert E == pytest.approx(expected, rel=1e-3)


def test_defocused_psf_approaches_geometry():
    x, I = ph.defocused_psf(-20.0, n_pupil=256, n=1024)
    dx = x[1] - x[0]
    assert I.sum()*dx**2 == pytest.approx(1.0)
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    inner = R < 30
    # geometric: 1/(pi 40**2) inside r = 40
    assert np.mean(I[inner]) == pytest.approx(1/(np.pi*40.0**2), rel=0.05)
    assert I[R > 46].sum()*dx**2 < 0.02


def test_defocused_psf_in_focus_is_airy():
    from scipy.special import j1
    x, I = ph.defocused_psf(0.0, n_pupil=128, n=1024)
    k = len(x)//2
    v = np.pi*2*x[k + 1:k + 40]       # 2 pi NA r/lambda with r in lambda/NA
    airy = (2*j1(v)/v)**2
    assert I[k, k + 1:k + 40]/I[k, k] == pytest.approx(airy, abs=2e-3)


def test_rippled_surface_slope_and_deflection():
    a, p = 1e-4, 0.5
    s = ph.RippledSurface(np.inf, 0.0, amplitude=a, period=p, n_after=1.5)
    r = np.linspace(0.05, 3.0, 200)
    h = 1e-6
    dz = (s.sag((r + h)**2) - s.sag((r - h)**2))/(2*h)
    assert s._slope_over_r(r**2)*r == pytest.approx(dz, rel=1e-5, abs=1e-9)
    # a collimated beam is deflected by (n - 1) dz/dr (small angles)
    s.thickness = 2.0
    S = [s, rt.Surface(np.inf, 10.0, n_after=1.0)]
    system = rt.System(S, 0, 550e-6, entrance_pupil_radius=3.0)
    y = np.linspace(0.1, 2.9, 15)
    P = np.column_stack([0*y, y, np.full(len(y), -1.0)])
    d = np.tile([0.0, 0.0, 1.0], (len(y), 1))
    b = system.trace(rt.Rays(P, d, np.zeros(len(y))))
    slope = b.directions[-1][:, 1]/b.directions[-1][:, 2]
    expected = -(1.5 - 1.0)*(-a*2*np.pi/p*np.sin(2*np.pi*y/p))
    assert slope == pytest.approx(expected, rel=1e-3, abs=1e-9)


def test_onion_ring_contrast_limits():
    wave, rays = ph.onion_ring_contrast(5e-6, 1.0, 10.0, 550e-6)
    assert wave == pytest.approx(rays, rel=1e-4)
    wave, rays = ph.onion_ring_contrast(5e-6, 1.0, 1/550e-6, 550e-6)
    assert abs(wave) < 1e-12        # Talbot: the ripple has disappeared
    z_eff, M = ph.converging_beam_equivalent(60.0, np.array([30.0, 62.0]))
    assert z_eff == pytest.approx([60.0, -60*62/2.0])
    assert M == pytest.approx([0.5, -2/60.0])


def test_onion_ring_contrast_against_waves():
    # a weak ripple in the pupil of a strongly defocused point
    W020, A, n_rings = -40.0, 0.01, 5.0
    period = 1/n_rings
    _, I = ph.defocused_psf(W020, wavefront=lambda rho: A*np.cos(2*np.pi*rho/period),
                            n_pupil=512, n=2048)
    x, I0 = ph.defocused_psf(W020, n_pupil=512, n=2048)
    k = len(x)//2
    rel = I[k, k:]/I0[k, k:] - 1
    r = x[k:]
    # the ripple maps onto the blur r = 2 |W020| rho: compare between 0.3 and 0.7 of the radius
    m = (r > 0.3*80) & (r < 0.7*80)
    amp = np.sqrt(2)*np.std(rel[m])
    talbot = np.pi/(2*abs(W020)*period**2)  # pi lambda z_eff/p**2 in pupil units
    expected = 2*2*np.pi*A*abs(np.sin(talbot))
    assert amp == pytest.approx(expected, rel=0.15)


def test_apodizer_t_number():
    s = np.array([0.3, 0.6, 2.0])
    rho = np.linspace(0, 1, 20001)
    for si, tau in zip(s, ph.apodizer_mean_transmission(s)):
        assert tau == pytest.approx(np.trapezoid(2*rho*ph.gaussian_apodizer(si)(rho), rho),
                                    rel=1e-6)
    s45 = ph.apodizer_for_t_number(2.8, 4.5)
    assert ph.t_number(2.8, ph.apodizer_mean_transmission(s45)) == pytest.approx(4.5)
    with pytest.raises(ValueError):
        ph.apodizer_for_t_number(2.8, 2.0)
