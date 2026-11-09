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


# --- sunstars (notebook 91) ---

def test_iris_geometry():
    for n in (5, 6, 9):
        v = ph.iris(n, 1.0)
        assert ph.polygon_area(v) == pytest.approx(n*np.tan(np.pi/n))
        assert ph.iris_edge_length(n, 1.0) == pytest.approx(2*np.tan(np.pi/n))
        assert ph.iris_apothem(n, ph.polygon_area(v)) == pytest.approx(1.0)
    # rounded blades at the opening they are made for: a circle
    v = ph.iris(9, 2.0, 2.0, n_arc=2000)
    assert ph.polygon_area(v) == pytest.approx(4*np.pi, rel=1e-5)
    assert ph.polygon_perimeter(v) == pytest.approx(4*np.pi, rel=1e-5)
    assert np.allclose(np.hypot(*v.T), 2.0)
    # closed further: between the circle and the straight polygon of the same apothem
    h = ph.iris_apothem(9, np.pi*0.5**2, blade_radius=2.0)
    v = ph.iris(9, h, 2.0)
    assert ph.polygon_area(v) == pytest.approx(np.pi*0.25, rel=1e-6)
    assert np.pi*h**2 < ph.polygon_area(v) < 9*np.tan(np.pi/9)*h**2
    assert np.min(np.hypot(*v.T)) == pytest.approx(h, rel=1e-4)
    with pytest.raises(ValueError):
        ph.iris(9, 2.0, 1.0)
    with pytest.raises(ValueError):
        ph.iris(2, 1.0)


def test_polygon_transform():
    v = ph.iris(7, 1.0, rotation=0.3)
    assert ph.polygon_transform(v, 0.0, 0.0) == pytest.approx(ph.polygon_area(v))
    # against a sum over the antialiased opening
    x = np.linspace(-1.3, 1.3, 1301)
    X, Y = np.meshgrid(x, x)
    dx = x[1] - x[0]
    T = ph.iris_coverage(X, Y, 7, 1.0, rotation=0.3, dx=dx)
    assert T.sum()*dx**2 == pytest.approx(ph.polygon_area(v), rel=1e-5)
    for kx, ky in ((3.0, 1.0), (0.0, 15.0), (-12.0, 7.0)):
        F_num = np.sum(T*np.exp(-1j*(kx*X + ky*Y)))*dx**2
        assert abs(ph.polygon_transform(v, kx, ky) - F_num) < 2e-3*ph.polygon_area(v)
    # a fine polygon approximating the unit circle: the Airy amplitude 2 pi J1(k)/k
    from scipy.special import j1
    k = np.array([0.5, 3.0, 9.0])
    F = ph.polygon_transform(ph.iris(9, 1.0, 1.0, n_arc=2000), k, 0*k)
    assert np.allclose(F.real, 2*np.pi*j1(k)/k, atol=1e-5)


def test_spike_count_and_edge_spike():
    from scipy.signal import find_peaks
    th = np.linspace(0, 2*np.pi, 3600, endpoint=False)
    kb = np.linspace(60, 90, 16)[:, None]
    for n in (5, 6, 7, 8):
        h = ph.iris_apothem(n, np.pi)
        v = ph.iris(n, h, rotation=0.1)
        prof = np.mean(abs(ph.polygon_transform(v, kb*np.cos(th), kb*np.sin(th)))**2, axis=0)
        prof = np.roll(prof, -np.argmin(prof))
        assert len(find_peaks(prof, prominence=0.3*prof.max())[0]) == ph.spike_count(n)
        # along the spike (normal of the first edge, rotation 0): k^2 I averages to (L/A)^2,
        # twice that for even n
        k = np.linspace(30, 300, 4000)
        I = abs(ph.polygon_transform(ph.iris(n, h), k, 0*k)/np.pi)**2
        L = ph.iris_edge_length(n, h)
        model = ph.edge_spike(k, L, np.pi, 2 if n % 2 == 0 else 1, 2*h)
        assert np.mean(I*k**2) == pytest.approx(np.mean(model*k**2), rel=0.05)
    assert [ph.spike_count(n) for n in range(3, 9)] == [6, 4, 10, 6, 14, 8]


def test_iris_psf_against_exact_transform():
    h = ph.iris_apothem(7, np.pi)
    k, I = ph.iris_psf(7, h, n=1024, dx=np.pi/160)
    c = len(k)//2
    assert I[c, c] == pytest.approx(1.0, rel=1e-3)
    v = ph.iris(7, h)
    for kk in (5.0, 10.0, 20.0):
        i = c + int(round(kk/(k[1] - k[0])))
        exact = abs(ph.polygon_transform(v, k[i], 0.0)/np.pi)**2
        assert I[c, i] == pytest.approx(exact, rel=0.02)


def test_rounded_edge_spike():
    A, L, rho = np.pi, 0.7, 4.0
    # limits of the Fresnel edge integral
    assert ph.curved_edge_spike(1e-3, L, rho, A) == pytest.approx(ph.edge_spike(1e-3, L, A), rel=1e-4)
    k = 1e6
    assert ph.curved_edge_spike(k, L, rho, A) == pytest.approx(2*np.pi*rho/(k**3*A**2), rel=0.01)
    assert ph.curved_edge_spike(10.0, L, np.inf, A) == ph.edge_spike(10.0, L, A)
    kc = ph.rounded_edge_crossover(L, rho)
    assert kc == pytest.approx(2*np.pi*rho/L**2)
    # the exact pattern of nine rounded blades follows the model beyond the crossover
    h = ph.iris_apothem(9, A, rho)
    v = ph.iris(9, h, rho, n_arc=400)
    Le = ph.iris_edge_length(9, h, rho)
    kc = ph.rounded_edge_crossover(Le, rho)
    KB = np.geomspace(3*kc, 900, 40)[:, None]*np.linspace(0.9, 1.1, 7)
    I = np.mean(abs(ph.polygon_transform(v, KB, 0*KB)/A)**2, axis=1)
    model = np.mean(ph.curved_edge_spike(KB, Le, rho, A), axis=1)
    assert np.mean(I/model) == pytest.approx(1.0, abs=0.1)


def test_power_outside_matches_airy():
    from scipy.special import j0, j1
    for kR in (30.0, 100.0):
        exact = j0(kR)**2 + j1(kR)**2              # 1 - encircled energy of the Airy pattern
        assert ph.power_outside(kR, 2*np.pi, np.pi) == pytest.approx(exact, rel=0.05)


def test_sunstar_spike_level():
    f, N, lam = 24.0, 16, 550e-6
    x, E = ph.sunstar(7, N, lam, focal_length=f, rotation=0.0, half_width=1.2)
    c = len(x)//2
    assert E.shape == (1, len(x), len(x))
    assert E[0, c, c] == pytest.approx(1.0, abs=0.03)          # inside the Sun's disk
    a = f*np.tan(ph.SUN_RADIUS)
    A = np.pi*(f/(2*N))**2
    L = ph.iris_edge_length(7, ph.iris_apothem(7, A))
    for r in (0.5, 0.8):
        i = c + int(round(r/(x[1] - x[0])))
        theory = a*L*lam*f/(2*np.pi**2*x[i]**2*A)
        assert E[0, c, i] == pytest.approx(theory, rel=0.08)
    # a point source: relative to the centre of the pattern, lowered by the average over the
    # pixel (an odd number of pixels puts one at the centre)
    x, E = ph.sunstar(7, N, lam, focal_length=f, source_radius=0.0, half_width=0.102)
    c = len(x)//2
    assert x[c] == pytest.approx(0.0, abs=1e-12)
    assert E[0, c, c] == E[0].max()
    assert 0.85 < E[0, c, c] < 1.0
    with pytest.raises(ValueError):
        ph.sunstar(7, N, lam, half_width=1.0, n=64)


# flare and ghosts (notebook 92)

def _R(a, b):
    return ((a - b)/(a + b))**2


@pytest.fixture(scope="module")
def lens8():
    return ph.double_gauss(8)


def test_ghost_paths_count(lens8):
    paths = ph.ghost_paths(lens8)
    n_refl = sum(1 for k in range(len(lens8.surfaces)) if lens8.n[k] != lens8.n[k + 1])
    assert len(paths) == n_refl*(n_refl - 1)//2 + n_refl
    assert len(ph.ghost_paths(lens8, sensor=False)) == n_refl*(n_refl - 1)//2


def test_ghost_power_is_product_of_fresnel_factors(lens8):
    n = [float(np.real(v)) for v in lens8.n]
    N = len(lens8.surfaces)
    rays, w = ph.flare_rays(lens8, 0.0, n=21, radius=0.3)
    _, _, p = ph.trace_flare(lens8, rays, w)
    T = np.prod([1 - _R(n[k], n[k + 1]) for k in range(N)])
    assert p.sum()/w.sum() == pytest.approx(T, rel=1e-5)
    for j, i in [(4, 2), (10, 8), (11, 0), (len(lens8.surfaces), 3)]:
        _, _, p = ph.trace_flare(lens8, rays, w, path=(j, i))
        Tg = (_R(n[j], n[j + 1]) if j < N else _R(1.0, 1.65))*_R(n[i], n[i + 1])
        for k in list(range(min(j, N))) + list(range(i + 1, min(j, N))) + list(range(i + 1, N)):
            Tg *= 1 - _R(n[k], n[k + 1])
        assert p.sum()/w.sum() == pytest.approx(Tg, rel=1e-5)


def test_ar_coatings_reflectance(lens8):
    from pyoptics.coatings import Coating, Layer
    n_glass = float(np.real(lens8.n[1]))
    lam = np.linspace(420, 680, 27)
    mean = {}
    for d in ("uncoated", "MgF2", "multilayer"):
        st = Coating([Layer(nl, t) for nl, t in ph.AR_DESIGNS[d]])
        mean[d] = np.mean([st.power(0.0, 1.0, n_glass, x*1e-6)["R_s"].real for x in lam])
    assert mean["uncoated"] == pytest.approx(_R(1.0, n_glass), rel=1e-6)
    assert 0.004 < mean["MgF2"] < 0.01
    assert mean["multilayer"] < 0.003
    c = ph.ar_coatings(lens8, "uncoated")
    assert all(v is None for v in c.values())
    c = ph.ar_coatings(lens8, "MgF2")
    assert set(c) == {k for k in range(len(lens8.surfaces))
                      if (lens8.n[k] == 1.0) != (lens8.n[k + 1] == 1.0)}


def test_ghost_analysis_against_trace():
    """Defocused ghosts at f/16: paraxial irradiance relative to white within
    a few percent of the traced one; it grows as N**2."""
    rng = np.random.default_rng(1)
    s16 = ph.double_gauss(16)
    info16 = {g.path: g for g in ph.ghost_analysis(s16)[0]}
    info8 = {g.path: g for g in ph.ghost_analysis(ph.double_gauss(8))[0]}
    d = ph.ghost_footprint(s16, 0.0, None, n_rays=4000, rng=rng)
    E_w = d.power.sum()/(np.pi*ph.DOUBLE_GAUSS_F**2)
    for path in [(11, 1), (4, 2), (8, 6)]:
        fp = ph.ghost_footprint(s16, 0.0, path, n_rays=8000, rng=rng)
        r2 = np.average(fp.x**2 + fp.y**2, weights=fp.power)
        E = fp.power.sum()/(2*np.pi*r2)/E_w
        assert E == pytest.approx(info16[path].irradiance, rel=0.06)
        assert info16[path].irradiance/info8[path].irradiance == pytest.approx(4.0, rel=0.02)


def test_sp_chains_match_polarization_trace(lens8):
    rng = np.random.default_rng(2)
    coat = ph.ar_coatings(lens8, "multilayer")
    field = (np.radians(-9.0), np.radians(6.0))
    fp = ph.ghost_footprint(lens8, field, (4, 2), coat, n_rays=3000, rng=rng,
                            wavelengths_nm=[ph.D_LINE*1e6])
    assert fp.spectrum.shape == (len(fp.x), 1)
    assert fp.spectrum[:, 0].sum() == pytest.approx(fp.power.sum(), rel=0.01)


def test_ghost_spectra_shapes(lens8):
    rng = np.random.default_rng(3)
    lam = [450.0, 550.0, 650.0]
    sets = [ph.ar_coatings(lens8, d) for d in ("uncoated", "MgF2")]
    x, y, specs = ph.ghost_spectra(lens8, 0.0, (10, 8), sets, lam, n_rays=2000, rng=rng)
    assert len(specs) == 2
    assert all(s.shape == (len(x), 3) for s in specs)
    # the coating takes away more than 95 % of the ghost
    assert specs[1].sum() < 0.05*specs[0].sum()


# depth of field and tilted lenses (notebook 94)

def test_dof_limits_consistent_with_blur():
    f, N, c = 50.0, 8, 0.03
    H = ph.hyperfocal_distance(f, N, c)
    for s_f in (1000.0, 3000.0, H):
        near, far = ph.dof_limits(f, N, c, s_f)
        assert ph.blur_diameter(f, N, s_f, near) == pytest.approx(c)
        if np.isfinite(far):
            assert ph.blur_diameter(f, N, s_f, far) == pytest.approx(c)
    assert np.isinf(ph.dof_limits(f, N, c, H)[1])
    assert ph.dof_limits(f, N, c, H)[0] == pytest.approx(H/2, rel=0.01)


def test_dof_limits_against_trace():
    lens = ph.double_gauss(8)
    par = lens.paraxial()
    Hz = par.cardinal_points.H
    fs = ph.focus(lens, 3000.0 - Hz)
    near, far = ph.dof_limits(par.efl, 8, 0.03, 3000.0)
    for d in (near, far):
        s = rt.System(fs.surfaces, fs.stop_index, fs.wavelength, object_distance=d - Hz, object_height=1.0)
        fp = ph.point_footprint(s, 0.0, 3000, np.random.default_rng(0))
        blur = 2*np.sqrt(2*np.average(fp.x**2 + fp.y**2, weights=fp.weight))
        assert blur == pytest.approx(0.03, rel=0.03)


def test_defocus_mtf_limits():
    from pyoptics import sensor as se
    f = np.linspace(0, 200, 81)
    assert np.allclose(ph.defocus_mtf(f, 8, 0.0, 550e-6), se.diffraction_mtf(f, 550e-6, 8), atol=2e-3)
    # many waves of defocus: the geometric blur disk at low frequencies
    m = ph.defocus_mtf(f[:15], 2.8, 0.3, 550e-6, n_pupil=256)
    assert np.allclose(m, ph.geometric_defocus_mtf(f[:15], 0.3/2.8), atol=0.03)
    assert ph.optimum_f_number(1.282, 550e-6) == pytest.approx(21.9, abs=0.1)


def test_tilted_thin_lens_rules():
    f, h = 50.0, 1000.0
    tilt = np.arcsin(f/h)
    assert ph.hinge_distance(f, tilt) == pytest.approx(h)
    Q = np.array([[0.0, -800.0, -7000.0]])
    assert np.allclose(ph.tilted_conjugate(ph.tilted_conjugate(Q, f, tilt), f, tilt), Q)
    for v in (50.0, 50.3, 51.0):
        (y0, _), ang, yS = ph.plane_of_focus(f, tilt, v)
        assert y0 == pytest.approx(-h)                                   # hinge rule
        assert y0 - np.tan(ang)*v == pytest.approx(yS)                   # Scheimpflug rule
    # the ground is in focus with v = h tan(tilt), so points on it have no blur
    v = h*np.tan(tilt)
    pts = np.array([[0.0, -h, -2000.0], [300.0, -h, -9000.0]])
    assert np.all(ph.tilted_blur(pts, f, 8, tilt, v) < 1e-9)
    # Merklinger: the depth of field limit J above the ground at the hyperfocal distance
    H = ph.hyperfocal_distance(f, 8, 0.03)
    assert ph.tilted_blur(np.array([[0.0, -h + h, -H]]), f, 8, tilt, v)[0] == pytest.approx(0.03, rel=0.02)


def test_tilt_lens_untilted_is_original():
    lens = ph.double_gauss(8)
    t0 = ph.tilt_lens(lens, 0.0, 20.0)
    x0, y0 = ph.trace_point(t0, (0.0, 100.0, -5000.0), 500, np.random.default_rng(1))
    s = rt.System(lens.surfaces, lens.stop_index, lens.wavelength, object_distance=5000.0, object_height=100.0)
    x1, y1 = ph.trace_point(s, (0.0, 100.0, -5000.0), 500, np.random.default_rng(1))
    assert np.allclose(x0, x1) and np.allclose(y0, y1)
    t = ph.tilt_lens(lens, np.radians(3), 20.0)
    R = t.frames()[0][2][1]
    assert np.allclose(R[:, 2], ph.lens_normal(np.radians(3)))
