import numpy as np
import pytest

from pyoptics import fieldtrace as ft
from pyoptics import analytic_phase as ap
from pyoptics.propagators import rayleigh_sommerfeld_I_TF
from pyoptics.fft import czt


WL = 633e-6
K = 2*np.pi/WL


def rel(a, b):
    return np.linalg.norm(a - b)/np.linalg.norm(b)


def second_moment_radius(I, x):
    """1/e**2 radius 2*sqrt(<x**2>) of a 1D irradiance."""
    xc = np.sum(x*I)/np.sum(I)
    return 2*np.sqrt(np.sum((x - xc)**2*I)/np.sum(I))


# --------------------------------------------------------------------------
# Field
# --------------------------------------------------------------------------

def test_from_array_splits_curvature():
    x = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, x)
    u = np.exp(-(X**2 + Y**2)/0.3**2)*np.exp(1j*K*(X**2 + Y**2)/(2*500.0))
    f = ft.from_array(u, x, x, WL, R=500.0)
    np.testing.assert_allclose(f.scalar_amplitude, np.abs(u), atol=1e-12)
    np.testing.assert_allclose(f.E()[0], u, atol=1e-12)
    assert f.radius_of_curvature[0] == pytest.approx(500.0)


def test_field_shapes_and_errors():
    x = np.linspace(-1, 1, 16)
    f = ft.Field(np.ones((16, 16, 2)), x, x, WL)
    assert f.is_vector and f.shape == (16, 16)
    g = ft.Field(np.ones(16), x, None, WL)
    assert g.one_d and g.shape == (16,) and g.y is None
    with pytest.raises(ValueError):
        ft.Field(np.ones((15, 16)), x, x, WL)
    with pytest.raises(ValueError):
        ft.Field(np.ones((16, 16)), x, x)


def test_scalarized_and_vectorized():
    x = np.linspace(-1, 1, 32)
    f = ft.gaussian(0.3, WL, x, x, polarization="rcp")
    v = f.vectorized()
    s, lost = v.scalarized()
    assert lost < 1e-14
    for a, b in zip(s.E(), f.E()):
        np.testing.assert_allclose(a, b, atol=1e-12)
    r = ft.gaussian(0.3, WL, x, x, polarization="radial")
    s, lost = r.scalarized()
    assert lost == pytest.approx(0.5, abs=1e-6)


def test_gaussian_power_normalization():
    x = np.linspace(-2, 2, 128)
    assert ft.gaussian(0.4, WL, x, x, power=2.5).power() == pytest.approx(2.5)
    assert ft.gaussian(0.4, WL, x, power=2.5).power() == pytest.approx(2.5)


# --------------------------------------------------------------------------
# E_z, H, Poynting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["spectral", "local"])
def test_ez_and_h_of_a_tilted_plane_wave(method):
    # E_x polarized plane wave tilted in x-z: k.E = 0 gives E_z = -k_x/k_z E_x
    x = np.linspace(-0.05, 0.05, 64, endpoint=False)
    # 20 periods across the (periodic) window
    kx = 20*2*np.pi/0.1
    theta = np.arcsin(kx/K)
    kz = K*np.cos(theta)
    f = ft.Field(np.ones((64, 64)), x, x, WL, beta=(kx, 0.0))
    Ez = f.Ez(method)
    np.testing.assert_allclose(Ez, -kx/kz*np.ones((64, 64)), rtol=1e-6, atol=1e-9)
    Hx, Hy, Hz = f.H(method)
    # H = k x E/(k0 Z0)
    np.testing.assert_allclose(Hy, (kz*1.0 - kx*Ez)/(K*ft.Z0), rtol=1e-6)
    np.testing.assert_allclose(Hx, 0.0, atol=1e-12)
    Sx, Sy, Sz = f.poynting(method)
    # |E|**2 = 1/cos**2, S = |E|**2/(2 Z0) along k
    np.testing.assert_allclose(Sz, np.cos(theta)/(2*ft.Z0)/np.cos(theta)**2, rtol=1e-6)
    np.testing.assert_allclose(Sx, np.sin(theta)/(2*ft.Z0)/np.cos(theta)**2, rtol=1e-6)


def test_ez_share_grows_with_na():
    x = np.linspace(-0.05, 0.05, 128)
    low = ft.Field(np.ones((128, 128)), x, x, WL, alpha=(K/(2*-100.0),)*2)
    high = ft.Field(np.ones((128, 128)), x, x, WL, alpha=(K/(2*-0.06),)*2)
    assert low.ez_share() < 1e-6 < 1e-2 < high.ez_share()


# --------------------------------------------------------------------------
# free-space operators
# --------------------------------------------------------------------------

@pytest.mark.parametrize("one_d", [False, True])
@pytest.mark.parametrize("method,tol", [("spw", 1e-5), ("fresnel", 1e-6)])
def test_gaussian_beam_propagation(one_d, method, tol):
    w0 = 0.3
    zR = K*w0**2/2
    x = np.linspace(-2, 2, 128)
    y = None if one_d else x
    src = ft.gaussian(w0, WL, x, y)
    out, notes, _ = ft._propagate(src, zR, method)
    ref = ft.gaussian(w0, WL, out.x, out.y, distance=zR)
    assert rel(out.E()[0], ref.E()[0]) < tol


@pytest.mark.parametrize("one_d", [False, True])
def test_gaussian_through_focus(one_d):
    # converging beam 50 Rayleigh lengths before its waist
    w0 = 0.02
    zR = K*w0**2/2
    d = -50*zR
    w = w0*np.sqrt(1 + (d/zR)**2)
    x = np.linspace(-3*w, 3*w, 64)
    y = None if one_d else x
    src = ft.gaussian(w0, WL, x, y, distance=d)
    for z in (49*zR, 50*zR, 51*zR):
        out, _, _ = ft._propagate(src, z, "fresnel")
        ref = ft.gaussian(w0, WL, out.x, out.y, distance=d + z)
        assert rel(out.E()[0], ref.E()[0]) < 1e-4


def test_spw_matches_rayleigh_sommerfeld():
    x = 0.004*(np.arange(128) - 64)
    X, Y = np.meshgrid(x, x)
    u = np.where(np.hypot(X, Y) < 0.1, np.exp(-(X**2 + Y**2)/0.06**2), 0.0)
    z = 0.5
    ref = rayleigh_sommerfeld_I_TF(u, x, x, z, WL)[0]
    out, _, _ = ft._propagate(ft.Field(u, x, x, WL), z, "spw")
    E = out.resample(x, x).E()[0]
    assert rel(E, ref) < 1e-3


def test_farfield_of_gaussian():
    w0 = 0.05
    zR = K*w0**2/2
    x = np.linspace(-0.2, 0.2, 64)
    src = ft.gaussian(w0, WL, x, x)
    z = 200*zR
    out, _, _ = ft._propagate(src, z, "farfield")
    ref = ft.gaussian(w0, WL, out.x, out.y, distance=z)
    assert rel(out.E()[0], ref.E()[0]) < 1e-2


def test_farfield_one_dimensional_against_angular_spectrum():
    # a narrow slit-like super-Gaussian: wide angles, far field exact
    r = 0.005
    x = np.linspace(-1.25*r, 1.25*r, 64)
    u = np.exp(-(np.abs(x)/r)**10)
    z = 10.0
    out, _, _ = ft._propagate(ft.Field(u, x, None, WL), z, "farfield", N=(1024, 1))
    # reference: quadrature of the angular spectrum at a few points
    xs = np.linspace(-1.25*r, 1.25*r, 2001)
    us = np.exp(-(np.abs(xs)/r)**10)
    kx = np.linspace(-0.9999*K, 0.9999*K, 400001)
    # U(kx) = sum_l us[l] exp(-i kx xs[l]) dx on the uniform kx grid: a
    # zoomed DFT, by the chirp-z transform
    dx, dk = xs[1] - xs[0], kx[1] - kx[0]
    U = czt(us*np.exp(-1j*kx[0]*xs), M=len(kx), w=np.exp(-1j*dk*dx))
    U *= np.exp(-1j*(kx - kx[0])*xs[0])*dx
    kz = np.sqrt(K**2 - kx**2)
    peak = None
    for X in (0.0, 1.0, 2.0):
        exact = np.trapezoid(U*np.exp(1j*(kx*X + kz*z)), kx)/(2*np.pi)
        peak = abs(exact) if peak is None else peak
        g = out.resample(np.array([X - 1e-3, X, X + 1e-3]))
        val = (g.scalar_amplitude*np.exp(1j*g.phase()))[1]
        assert abs(val - exact) < 1e-2*peak


def test_geometric_converging_wave():
    # spherical wave far from its focus: geometrical optics is accurate
    x = np.linspace(-6.25, 6.25, 128)
    X, Y = np.meshgrid(x, x)
    A = np.exp(-(np.hypot(X, Y)/2.5)**2)
    f = ft.Field(A, x, x, WL, alpha=(K/(2*-100.0),)*2)
    out, _, _ = ft._propagate(f, 50.0, "geometric")
    spw, _, _ = ft._propagate(f, 50.0, "spw")
    g = out.resample(spw.x, spw.y)
    assert rel(g.E()[0], spw.E()[0]) < 2e-2
    # the beam has half its size and twice its amplitude
    assert np.max(np.abs(out.scalar_amplitude)) == pytest.approx(2.0, rel=2e-2)


def test_geometric_refuses_caustic():
    w = 0.02*np.sqrt(1 + (1000.0/(K*0.02**2/2))**2)
    f = ft.gaussian(0.02, WL, np.linspace(-3*w, 3*w, 64), distance=-1000.0)
    with pytest.raises(ft.PropagationFailure):
        ft._propagate(f, 1000.0, "geometric")
    with pytest.raises(ft.PropagationFailure):
        ft._propagate(f, -1.0, "farfield")


def test_analytic_phase_y_invariant_field():
    # the separable fit must accept spectra that do not extend in k_y
    x = np.linspace(-1, 1, 128)
    y = np.array([-1.0, 0.0, 1.0])
    A = np.repeat(np.exp(-(x/0.3)**2)[None, :], 3, axis=0)
    q = ap.QuadraticPhaseField(A, x, y, WL, alpha=(K/(2*-50.0), 0.0))
    out = ap.propagate(q, 30.0)
    assert np.all(np.isfinite(out.amplitude))
    np.testing.assert_allclose(out.amplitude[0], out.amplitude[2])


# --------------------------------------------------------------------------
# automatic selection
# --------------------------------------------------------------------------

def super_gaussian(radius, R=np.inf, N=64, order=10):
    L = 1.25*radius
    x = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, x)
    a = 0.0 if np.isinf(R) else K/(2*R)
    return ft.Field(np.exp(-(np.hypot(X, Y)/radius)**order), x, x, WL, alpha=(a, a))


def test_selection_paper_example_near_field():
    # Wyrowski and Kuhn, table 1: 50 um super-Gaussian, z = 1 mm: SPW
    choice, table = ft.estimate_propagation(super_gaussian(0.05), 1.0)
    assert choice in ("spw", "fresnel")
    assert table[choice]["error"] < 1e-2
    assert table["farfield"]["error"] > 0.1


def test_selection_paper_example_far_field():
    # table 3: 5 um super-Gaussian, z = 10 mm: the far-field operator
    choice, table = ft.estimate_propagation(super_gaussian(0.005), 10.0)
    assert choice == "farfield"
    assert table["farfield"]["error"] < 1e-2
    assert table["fresnel"]["error"] > 0.1
    assert table["spw"]["effort"] > table["farfield"]["effort"]


def test_selection_spherical_wave_is_cheap_with_analytic_phase():
    # table 5: spherical wave R = 100 mm, 10 mm aperture, z = 50 mm. The paper's
    # SPW needs 10522**2 samples; with the analytic phase it needs ~100**2.
    choice, table = ft.estimate_propagation(super_gaussian(5.0, R=-100.0, order=20), 50.0)
    assert choice == "spw"
    assert table["spw"]["effort"] < 200**2
    assert table["geometric"]["error"] < 1e-2


def test_selection_respects_candidates():
    choice, table = ft.estimate_propagation(super_gaussian(0.05), 10.0,
                                            candidates=("fresnel", "geometric"))
    assert choice == "fresnel"
    assert set(table) == {"fresnel", "geometric", "reference"}


# --------------------------------------------------------------------------
# components and chains
# --------------------------------------------------------------------------

def test_lens_focus_chain():
    w, f = 1.0, 50.0
    x = np.linspace(-3, 3, 128)
    src = ft.gaussian(w, WL, x, x, power=1.0)
    chain = ft.Chain([ft.ThinLens(f), ft.Propagate(f), ft.Detector("focus", ("irradiance", "power"))])
    res = chain.run(src)
    foc = res["focus"]
    assert foc.power == pytest.approx(1.0, rel=1e-4)
    I = foc.irradiance
    c = np.unravel_index(np.argmax(I), I.shape)
    w_focus = second_moment_radius(I[c[0]], foc.field.x)
    # the focused waist of a Gaussian beam, w' = lambda*f/(pi*w) (f >> zR'...)
    w_expected = WL*f/(np.pi*w)/np.sqrt(1 + (WL*f/(np.pi*w**2))**2)
    assert w_focus == pytest.approx(w_expected, rel=2e-2)
    report = res.report()
    assert "thin lens" in report and "candidates" in report


def test_explicit_method_and_report():
    x = np.linspace(-1, 1, 64)
    res = ft.Chain([ft.Propagate(10.0, method="fresnel")]).run(ft.gaussian(0.3, WL, x, x))
    assert res.log[0]["method"] == "fresnel"
    assert res.log[0]["candidates"] is None


def test_thin_element_equals_thin_lens_paraxially():
    f = 200.0
    x = np.linspace(-1, 1, 128)
    src = ft.gaussian(0.3, WL, x, x)
    n = 1.5
    # plano-convex thin lens: h = -(x**2 + y**2)/(2*R), 1/f = (n - 1)/R
    R = (n - 1)*f
    tea = ft.ThinElement(lambda X, Y: -(X**2 + Y**2)/(2*R), n)
    a = ft.Chain([tea, ft.Propagate(f/2, method="fresnel")]).run(src).field
    b = ft.Chain([ft.ThinLens(f), ft.Propagate(f/2, method="fresnel")]).run(src).field
    Ea = a.resample(b.x, b.y).E()[0]
    assert rel(Ea, b.E()[0]) < 1e-3


def test_thin_element_fresnel_losses_and_interface():
    x = np.linspace(-1, 1, 32)
    src = ft.gaussian(0.3, WL, x, x, power=1.0)
    out = ft.Chain([ft.ThinElement(np.zeros((32, 32)), 1.5, fresnel=True)]).run(src).field
    assert out.power() == pytest.approx((4*1.5/2.5**2)**2, rel=1e-12)
    out = ft.Chain([ft.Interface(1.5, fresnel=True)]).run(src).field
    assert out.n == 1.5
    # transmitted power n2/n1 |t|**2 (irradiance in the medium carries n)
    assert 1.5*out.power() == pytest.approx(1 - (0.5/2.5)**2, rel=1e-12)


def test_aperture_and_phase_screen():
    x = np.linspace(-1, 1, 65)
    src = ft.gaussian(10.0, WL, x, x, power=1.0)
    out = ft.Chain([ft.Aperture(radius=0.5), ft.PhaseScreen(lambda X, Y: 0.3*X)]).run(src).field
    assert np.all(out.scalar_amplitude[np.hypot(*np.meshgrid(x, x)) > 0.5] == 0)
    np.testing.assert_allclose(np.angle(out.scalar_amplitude[32, 32:40]), 0.3*x[32:40], atol=1e-12)


def test_polarization_auto_promotes_and_demotes():
    x = np.linspace(-2, 2, 64)

    def radial_polarizer(X, Y):
        t = np.arctan2(Y, X)
        c, s = np.cos(t), np.sin(t)
        J = np.zeros(X.shape + (2, 2), dtype=complex)
        J[..., 0, 0], J[..., 0, 1], J[..., 1, 0], J[..., 1, 1] = c*c, c*s, c*s, s*s
        return J

    chain = ft.Chain([ft.Jones(np.array([[0, 1], [1, 0]])), ft.Jones(radial_polarizer),
                      ft.Propagate(10.0, method="fresnel"), ft.Jones(np.diag([1.0, 0.0]))])
    src = ft.gaussian(0.5, WL, x, x, polarization="45")
    res = chain.run(src)
    pols = [row["polarization"] for row in res.log]
    assert pols == ["scalar", "vector", "vector", "scalar"]
    assert "vector" in res.log[1]["notes"][0]
    # forced modes
    res = chain.run(src, polarization="vector")
    assert all(row["polarization"] == "vector" for row in res.log)
    res = chain.run(src, polarization="scalar")
    assert all(row["polarization"] == "scalar" for row in res.log)
    assert "dropped" in res.log[1]["notes"][0]


def test_vector_and_scalar_agree_for_global_polarization():
    x = np.linspace(-1, 1, 64)
    src = ft.gaussian(0.3, WL, x, x, polarization="rcp")
    chain = ft.Chain([ft.ThinLens(100.0), ft.Propagate(50.0, method="spw")])
    a = chain.run(src, polarization="scalar").field
    b = chain.run(src, polarization="vector").field
    for ea, eb in zip(a.E(), b.E()):
        np.testing.assert_allclose(ea, eb, atol=1e-10*np.max(np.abs(eb)))


def test_detector_includes_ez_at_high_na():
    x = np.linspace(-0.02, 0.02, 64)
    f = ft.Field(np.exp(-(np.hypot(*np.meshgrid(x, x))/0.012)**2), x, x, WL,
                 alpha=(K/(2*-0.02),)*2)
    res = ft.Chain([ft.Detector("d")]).run(f)
    assert "E_z" in res.log[0]["notes"][0]
    I_auto = res["d"].irradiance
    assert np.sum(I_auto) > np.sum(f.irradiance())
    res = ft.Chain([ft.Detector("d")]).run(f, polarization="scalar")
    np.testing.assert_allclose(res["d"].irradiance, f.irradiance())


def test_one_dimensional_chain_and_cylindrical_lens():
    x = np.linspace(-2, 2, 128)
    src = ft.gaussian(0.5, WL, x)
    res = ft.Chain([ft.ThinLens(100.0), ft.Propagate(100.0), ft.Detector("f")]).run(src)
    I = res["f"].irradiance
    w_expected = WL*100.0/(np.pi*0.5)
    assert second_moment_radius(I, res["f"].field.x) == pytest.approx(w_expected, rel=3e-2)


def test_chain_list_behaviour_and_function_step():
    a = ft.Chain([ft.ThinLens(10.0)])
    b = a + [ft.Function(lambda f: f, label="identity")]
    assert len(b) == 2 and isinstance(b[0:1], ft.Chain)
    nested = ft.Chain([a, ft.Propagate(1.0, method="fresnel")])
    assert "thin lens" in repr(nested)
    with pytest.raises(TypeError):
        ft.Chain([lambda f: f])
    x = np.linspace(-1, 1, 32)
    res = nested.run(ft.gaussian(0.3, WL, x, x), keep=True)
    assert len(res.fields) == 3


# --------------------------------------------------------------------------
# Trace: geometrical optics through raytrace.System
# --------------------------------------------------------------------------

def _ideal_lens_system(f=100.0, back=50.0):
    from pyoptics.raytrace import System, IdealLens
    return System([IdealLens(f, thickness=back, semi_diameter=5.0)], 0, WL)


def test_trace_ideal_lens_matches_thin_lens():
    x = np.linspace(-3, 3, 96)
    src = ft.gaussian(1.0, WL, x, x)
    a = ft.Chain([ft.Trace(_ideal_lens_system())]).run(src).field
    b = ft.Chain([ft.ThinLens(100.0), ft.Propagate(50.0, method="spw")]).run(src).field
    assert a.z == pytest.approx(50.0)
    # geometrical optics lacks the (small) Gouy phase difference
    assert rel(a.resample(b.x, b.y).E()[0], b.E()[0]) < 3E-2
    assert a.power() == pytest.approx(src.power(), rel=1E-2)


def test_trace_fresnel_losses_and_power():
    from pyoptics.raytrace import System, Surface
    sysm = System([Surface(R=200.0, thickness=5.0, n_after=1.5, semi_diameter=10.0),
                   Surface(R=np.inf, thickness=20.0, n_after=1.0, semi_diameter=10.0)], 0, WL)
    x = np.linspace(-3, 3, 96)
    src = ft.gaussian(1.0, WL, x, x)
    out = ft.Chain([ft.Trace(sysm)]).run(src).field
    T = (1 - 0.04)**2
    assert out.power()/src.power() == pytest.approx(T, rel=1E-2)


def test_trace_depolarization_of_a_ball_lens():
    from pyoptics.raytrace import System, Surface
    wl, a, n = 0.5, 50.0, 1.5
    sysm = System([Surface(R=a, thickness=2*a, n_after=n, semi_diameter=0.999*a),
                   Surface(R=-a, thickness=0.5, n_after=1.0, semi_diameter=0.999*a)], 0, wl)
    x = np.linspace(-45, 45, 121)
    X, Y = np.meshgrid(x, x)
    src = ft.Field(np.exp(-(np.hypot(X, Y)/25.0)**6), x, x, wl, z=-a)
    res = ft.Chain([ft.Trace(sysm, distance=0.0)]).run(src, polarization="vector")
    Ex, Ey = res.field.E()
    share = np.sum(np.abs(Ey)**2)/np.sum(np.abs(Ex)**2)
    assert 1E-5 < share < 0.1
    # cross polarized lobes in the four quadrants: odd in x and in y
    assert rel(Ey[:, ::-1], -Ey) < 0.05
    assert rel(Ey[::-1, :], -Ey) < 0.05
    assert rel(Ex[:, ::-1], Ex) < 0.05
    # auto mode keeps the vector field only where it matters
    auto = ft.Chain([ft.Trace(sysm)]).run(src, tol=0.5)
    assert not auto.field.is_vector


def test_chain_from_system_hands_over_to_wave_optics():
    sysm = _ideal_lens_system()
    chain = ft.Chain.from_system(sysm, gap=10.0, method="spw")
    assert [type(s) for s in chain] == [ft.Trace, ft.Propagate]
    assert chain[1].z == pytest.approx(40.0)
    assert sysm.surfaces[-1].thickness == 50.0
    x = np.linspace(-3, 3, 96)
    src = ft.gaussian(1.0, WL, x, x)
    a = chain.run(src).field
    b = ft.Chain.from_system(sysm).run(src).field
    assert len(ft.Chain.from_system(sysm)) == 1
    assert a.z == pytest.approx(b.z)
    assert rel(b.resample(a.x, a.y).E()[0], a.E()[0]) < 3E-2
    with pytest.raises(ValueError):
        ft.Chain.from_system(sysm, gap=60.0)


# --------------------------------------------------------------------------
# single-FT Fresnel, parallel arms and Babinet obstructions
# --------------------------------------------------------------------------

def _smooth_square(x, half=4.0, edge=0.1):
    from scipy.special import erf
    X, Y = np.meshgrid(x, x)
    s = lambda u: 0.5*(1 - erf((np.abs(u) - half)/edge))
    return s(X)*s(Y)


def test_fresnel_ft_matches_fresnel_for_a_small_field():
    wl = 0.0008
    x = np.linspace(-0.1, 0.1, 41)
    X, Y = np.meshgrid(x, x)
    src = ft.Field(np.exp(-(X**2 + Y**2)/0.03**2), x, x, wl)
    a = ft.Chain([ft.Propagate(50.0, method="fresnel_ft")]).run(src).field
    b = ft.Chain([ft.Propagate(50.0, method="fresnel")]).run(src).field
    xd = np.linspace(-1, 1, 101)
    Ea, Eb = a.resample(xd, xd).E()[0], b.resample(xd, xd).E()[0]
    assert rel(Ea, Eb) < 1E-2
    # the output window is independent of the (small) input window
    assert a.x[-1] - a.x[0] > 1.0


def test_parallel_arms_interfere():
    wl = 0.0008
    x = np.linspace(-5, 5, 129)
    src = ft.Field(_smooth_square(x), x, x, wl)
    for phi in (0.0, np.pi/2, np.pi):
        chain = ft.Chain([ft.Parallel([ft.Propagate(10.0)], [ft.Propagate(10.0)],
                                      weights=(0.5, 0.5*np.exp(1j*phi))),
                          ft.Detector("out")])
        res = chain.run(src)
        assert isinstance(res.field, ft.FieldSum) and len(res.field) == 2
        single = ft.Chain([ft.Propagate(10.0)]).run(src).field
        ratio = res["out"].field.power()/single.power()
        assert ratio == pytest.approx(np.cos(phi/2)**2, abs=1E-4)
    assert [row["index"] for row in res.log] == ["0.1.0", "0.2.0", "0", "1"]
    assert isinstance(ft.Combine().apply(res.field, {})[0], ft.Field)


def test_babinet_obstruction_against_a_fine_grid():
    from pyoptics.propagators import rayleigh_sommerfeld_I_TF
    from scipy.interpolate import RegularGridInterpolator
    wl = 0.0008
    x = np.linspace(-5, 5, 257)
    alpha = -np.pi/wl/2000
    X, Y = np.meshgrid(x, x)
    src = ft.Field(_smooth_square(x), x, x, wl, alpha=(alpha, alpha))
    xd, yd = np.linspace(0.0, 2.0, 101), np.linspace(-0.5, 1.5, 101)
    res = ft.Chain([ft.Obstruction(radius=0.1, center=(1.0, 0.5)), ft.Propagate(100.0),
                    ft.Detector("d", x=xd, y=yd)]).run(src)
    assert [row["method"] for row in res.log][0] == "Babinet"
    # the small part is cheap: far fewer samples than the main beam
    small = res.field.parts[1]
    assert small.shape[0]*small.shape[1] < 0.1*len(x)**2
    xb = np.linspace(-6, 6, 2048)
    XB, YB = np.meshgrid(xb, xb)
    u = (_smooth_square(xb)*np.exp(1j*alpha*(XB**2 + YB**2))
         * (((XB - 1.0)**2 + (YB - 0.5)**2) > 0.01))
    u = rayleigh_sommerfeld_I_TF(u, xb, xb, 100.0, wl)[0]
    pts = np.stack(np.meshgrid(yd, xd, indexing="ij"), -1)
    Ib = RegularGridInterpolator((xb, xb), np.abs(u)**2)(pts)
    assert rel(res["d"].irradiance, Ib) < 2E-2


# --------------------------------------------------------------------------
# Mode overlap (fibre coupling)
# --------------------------------------------------------------------------

def test_overlap_of_gaussians_matches_analytic_values():
    x = np.linspace(-0.05, 0.05, 201)
    w = 5e-3
    mode = ft.gaussian(w, WL, x, x)
    assert np.isclose(ft.gaussian(w, WL, x, x, power=3.0).overlap(mode), 1.0, atol=1e-9)
    # waist mismatch: (2 w1 w2/(w1**2 + w2**2))**2
    w2 = 8e-3
    eta = ft.gaussian(w2, WL, x, x).overlap(mode)
    assert np.isclose(eta, (2*w*w2/(w**2 + w2**2))**2, rtol=1e-6)
    # lateral offset d: exp(-d**2/w**2)
    X, Y = np.meshgrid(x, x)
    d = 4e-3
    shifted = ft.from_array(np.exp(-((X - d)**2 + Y**2)/w**2), x, x, WL)
    assert np.isclose(shifted.overlap(mode), np.exp(-d**2/w**2), rtol=1e-6)
    # crossed polarization does not couple
    assert ft.gaussian(w, WL, x, x, polarization="y").overlap(mode) < 1e-12


def test_overlap_counts_curvature_and_field_outside_the_mode_grid():
    w, z = 5e-3, 0.2
    x = np.linspace(-0.1, 0.1, 401)
    xm = np.linspace(-0.03, 0.03, 121)
    mode = ft.gaussian(w, WL, xm, xm)
    beam = ft.gaussian(w, WL, x, x, distance=z)
    # analytic: two Gaussians with q parameters q1 = i zR and q2 = z + i zR
    zR = np.pi*w**2/WL
    wz = w*np.sqrt(1 + (z/zR)**2)
    eta = 4/((wz/w + w/wz)**2 + (np.pi*w*wz/(WL*z*(1 + (zR/z)**2)))**2)
    assert np.isclose(beam.overlap(mode), eta, rtol=2e-3)
    # a beam wider than the mode window loses the part outside
    wide = ft.gaussian(0.05, WL, x, x)
    assert wide.overlap(mode) < (2*w*0.05/(w**2 + 0.05**2))**2*1.001


def test_detector_coupling_into_a_fibre_mode():
    # a weak thin lens (a paraxial phase, so a long focal length) focuses a
    # collimated beam onto the matched mode
    wl, f, w = 0.78e-3, 50.0, 0.33
    x = np.linspace(-1.0, 1.0, 256)
    w_f = wl*f/(np.pi*w)
    xm = np.linspace(-4*w_f, 4*w_f, 161)
    mode = ft.gaussian(w_f, wl, xm, xm)
    r = ft.Chain([ft.ThinLens(f), ft.Propagate(f),
                  ft.Detector("fibre", ("coupling",), x=xm, y=xm, mode=mode)]).run(ft.gaussian(w, wl, x, x))
    assert r["fibre"].coupling > 0.99
    with pytest.raises(ValueError):
        ft.Detector("fibre", ("coupling",))
