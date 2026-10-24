#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the lithography toolbox (pyoptics.litho).

References: vectorial_imaging.image_1d()/image_2d() (exact sums over the
diffraction orders), FFTs of sampled masks, Fresnel coefficients
(interfaces), the analytic standing wave in a resist on a substrate,
symmetries (rotation of the layout, x <-> y for quadrupoles), the
equivalence of a boundary layer with t = t_absorber and a bias, and
threshold metrics of analytic images.
"""

import numpy as np
import pytest

from pyoptics.interfaces import t_p, t_s
from pyoptics.litho import (BoundaryLayer, FilmMask, Illumination, LithoSimulation, MaskLayout,
                            Projector, ThinMask, WaferStack, metrics)
from pyoptics.polarization import IPS
from pyoptics.utils import grid1d
from pyoptics.vectorial_imaging import (image_1d, image_2d, lines_spaces_orders,
                                        source_points)


WL = 193.0
N_WATER = 1.437
NA = 1.35


# illumination:

@pytest.mark.parametrize("pole", ["sector", "circle"])
@pytest.mark.parametrize("shape, angles", [("dipole", (0, 180)), ("quadrupole", (0, 90, 180, 270)),
                                           ("quasar", (45, 135, 225, 315))])
def test_multipole_sources(shape, angles, pole):
    ill = getattr(Illumination, shape)(0.9, 0.6, 30.0, pole=pole, n_samples=41)
    r, phi = np.hypot(ill.s_x, ill.s_y), np.degrees(np.arctan2(ill.s_y, ill.s_x))

    assert np.isclose(ill.weights.sum(), 1)
    assert np.all(r <= 0.9 + 1E-9) and np.all(r >= 0.6 - 1E-9)
    dist = np.min([np.abs((phi - a + 180) % 360 - 180) for a in angles], axis=0)
    assert np.all(dist <= 15 + 1E-6 if pole == "sector" else dist <= 90/2)
    # all poles carry the same weight:
    pole_index = np.argmin([np.abs((phi - a + 180) % 360 - 180) for a in angles], axis=0)
    w = [ill.weights[pole_index == i].sum() for i in range(len(angles))]
    assert np.allclose(w, 1/len(angles))


def test_conventional_annular_coherent_and_map():
    conv = Illumination.conventional(0.5, n_samples=41)
    ann = Illumination.annular(0.9, 0.7, n_samples=41)

    assert np.all(np.hypot(conv.s_x, conv.s_y) <= 0.5 + 1E-9)
    assert conv.sigma_max <= 0.5 + 1E-9 and ann.sigma_max > 0.85
    assert len(Illumination.coherent(0.3, 0.1)) == 1
    S_x, S_y = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    free = Illumination.from_map(S_x, S_y, np.exp(-(S_x**2 + S_y**2)/0.2))
    assert np.isclose(free.weights.sum(), 1) and free.sigma_max <= 1 + 1E-9
    with pytest.raises(ValueError):
        Illumination.dipole(0.9, 0.85, 5.0, n_samples=5)  # no point inside


def test_polarization_states_and_dop():
    ill = Illumination.annular(0.9, 0.6, n_samples=21, polarization="TE")
    for s_x, s_y, w, states in ill:
        assert len(states) == 1
        assert np.isclose(np.real(states[0][1] @ [s_x, s_y]), 0)  # azimuthal

    for dop in (0.0, 0.5, 0.9):
        st = ill.with_polarization("y", dop).polarization_states(0.7, 0.0)
        assert np.isclose(sum(w for w, J in st), 1)
        S = np.array([1.0, -dop, 0.0, 0.0])
        ips = sum(w*abs(J[1])**2 for w, J in st)
        assert np.isclose(ips, IPS(S, [0, 1]))


# mask layouts:

@pytest.mark.parametrize("width", [0.0, 6.0])
def test_region_coefficients_match_fft(width):
    Px, Py, N = 120.0, 100.0, 1200
    layout = MaskLayout.staggered_contacts(Px, Py/2, 40.0, 30.0)
    m, n = np.arange(-4, 5), np.arange(-4, 5)
    c = layout.region_coefficients(m, n, width)
    assert np.allclose(sum(c.values()), np.outer(n == 0, m == 0))

    x, y = grid1d(-Px/2, Px/2, N), grid1d(-Py/2, Py/2, N)
    X, Y = np.meshgrid(x, y)
    region = np.zeros(X.shape)
    for x0, y0, a, b in layout.features:
        dx, dy = (X - x0 + Px/2) % Px - Px/2, (Y - y0 + Py/2) % Py - Py/2  # periodic
        region += (np.abs(dx) < (a - width)/2) & (np.abs(dy) < (b - width)/2)
    spec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(region)))/N**2
    assert np.allclose(c["feature"], spec[N//2 - 4:N//2 + 5, N//2 - 4:N//2 + 5], atol=2E-3)


def test_transmission_sampling():
    layout = MaskLayout.lines_spaces(100.0, 30.0)
    x = np.linspace(-50, 50, 1001)

    assert np.isclose(np.mean(layout.transmission(x) == 0), 0.3, atol=0.01)
    holes = MaskLayout.contact_holes(100.0, 50.0)
    assert np.isclose(np.mean(holes.transmission(x, x) == 1), 0.25, atol=0.01)


# imaging against vectorial_imaging:

@pytest.mark.parametrize("pol", ["x", "y", "unpolarized", "TM"])
@pytest.mark.parametrize("vectorial", [True, False])
def test_lines_against_image_1d(pol, vectorial):
    pitch, cd = 90.0, 40.0
    src = source_points("dipole", 0.95, 0.6, opening_angle=35, n_samples=21)
    ill = Illumination(*src, polarization=pol)
    proj = Projector(NA, WL, N_WATER, vectorial=vectorial, zernikes={9: 0.03})
    sim = LithoSimulation(MaskLayout.lines_spaces(pitch, cd), ThinMask.att_psm(0.06), ill, proj,
                          normalize=False)
    x = np.linspace(-pitch, pitch, 201)

    I_ref = image_1d(x, pitch, lines_spaces_orders(pitch, cd, 6, line=-np.sqrt(0.06)), NA, WL,
                     src, pol, n=N_WATER, defocus=40.0, vectorial=vectorial,
                     phase=lambda P_x, P_y: proj.pupil_phase(P_x, P_y))
    assert np.allclose(sim.image(x, focus=40.0), I_ref, atol=1E-12)


def test_contacts_against_image_2d():
    pitch, cd, N = 120.0, 55.0, 64
    layout = MaskLayout.contact_holes(pitch, cd, cd_y=45.0)
    ill = Illumination.quasar(0.9, 0.6, 30.0, n_samples=15, polarization="TE")
    sim = LithoSimulation(layout, ThinMask.att_psm(0.06), ill, Projector(NA, WL, N_WATER),
                          normalize=False)
    x = grid1d(-pitch/2, pitch/2, N)
    X, Y = np.meshgrid(x, x)
    # band limited mask from the analytic orders:
    m = np.arange(-6, 7)
    c = layout.region_coefficients(m, m)
    J = sim._region_jones(0.0, 0.0)
    orders = sum(J[r][0, 0]*c[r] for r in c)
    mask = sum(orders[j, i]*np.exp(2j*np.pi*(m[i]*X + m[j]*Y)/pitch)
               for i in range(len(m)) for j in range(len(m)))

    I_ref = image_2d(mask, x, x, NA, WL, ill.points(), "TE", n=N_WATER, defocus=-25.0)
    assert np.allclose(sim.image(x, x, focus=-25.0), I_ref, atol=1E-10)


def test_rotated_lines_give_rotated_image():
    proj = Projector(NA, WL, N_WATER)
    u = np.linspace(-80, 80, 81)
    sim_y = LithoSimulation(MaskLayout.lines_spaces(80.0, 40.0, "y"), ThinMask.binary(),
                            Illumination.dipole(0.98, 0.75, axis="x", polarization="y"), proj)
    sim_x = LithoSimulation(MaskLayout.lines_spaces(80.0, 40.0, "x"), ThinMask.binary(),
                            Illumination.dipole(0.98, 0.75, axis="y", polarization="x"), proj)

    assert np.allclose(sim_y.image(u), sim_x.image([0.0], u)[:, 0])


@pytest.mark.parametrize("pol", ["TE", "TM", "unpolarized"])
def test_quadrupole_contact_image_is_symmetric(pol):
    layout = MaskLayout.contact_holes(110.0, 50.0)
    sim = LithoSimulation(layout, ThinMask.binary(),
                          Illumination.quadrupole(0.95, 0.7, 30.0, polarization=pol), Projector(NA, WL, N_WATER))
    x = np.linspace(-55, 55, 23)
    I = sim.image(x, x)

    assert np.allclose(I, I.T)
    assert np.allclose(I, I[::-1])


def test_normalization_to_open_frame():
    sim = LithoSimulation(MaskLayout.lines_spaces(1000.0, 0.2), ThinMask.binary(),
                          Illumination.annular(0.9, 0.6, polarization="TE"), Projector(NA, WL, N_WATER))
    x = np.linspace(-400, 400, 9)

    assert np.allclose(sim.image(x), 1, atol=0.01)  # 0.2 nm line on 1 µm pitch: nearly clear


def test_zernike_defocus_equals_focus_at_low_NA():
    NA_, pitch = 0.3, 1500.0
    ill = Illumination.conventional(0.5, n_samples=15)
    layout = MaskLayout.lines_spaces(pitch, pitch/2)
    z = 2000.0
    sim_f = LithoSimulation(layout, ThinMask.binary(), ill, Projector(NA_, WL, vectorial=False))
    sim_z = sim_f.with_(projector=Projector(NA_, WL, vectorial=False,
                                            zernikes={4: -z*NA_**2/(4*WL)}))
    x = np.linspace(-pitch/2, pitch/2, 101)

    assert np.allclose(sim_f.image(x, focus=z), sim_z.image(x), atol=0.02)


@pytest.mark.parametrize("vectorial", [True, False])
def test_apodization_of_two_beam_image(vectorial):
    """Orders -+1 of an alternating PSM at the pupil radius rho: an apodization
    exp(a*Z_4) scales the image relative to the open frame (rho = 0) by
    exp(2*a*(Z_4(rho) - Z_4(0))) and keeps the contrast."""

    pitch, a = 200.0, -0.3
    rho = WL/(pitch*NA)
    layout = MaskLayout.lines_spaces(pitch, pitch/2)
    ill = Illumination.coherent(polarization="y")
    proj = Projector(NA, WL, N_WATER, vectorial=vectorial)
    x = np.linspace(-pitch/2, pitch/2, 201)
    I = LithoSimulation(layout, ThinMask(-1.0), ill, proj).image(x)
    I_apo = LithoSimulation(layout, ThinMask(-1.0), ill, proj.with_(apodization_zernikes={4: a})).image(x)

    assert np.allclose(I_apo, I*np.exp(2*a*(2*rho**2 - 1 - (-1))))
    assert np.isclose(metrics.contrast(I_apo), metrics.contrast(I))


def test_apodization_callable_and_constant_term():
    layout = MaskLayout.contact_holes(130.0, 65.0)
    ill = Illumination.quasar(0.95, 0.7, n_samples=15, polarization="TE")
    proj = Projector(NA, WL, N_WATER)
    x = np.linspace(-65, 65, 21)
    sim = LithoSimulation(layout, ThinMask.att_psm(0.06), ill, proj)
    I_z = sim.with_(projector=proj.with_(apodization_zernikes={4: -0.2, 9: 0.1})).image(x, x)
    rho2 = lambda P_x, P_y: P_x**2 + P_y**2
    apo = lambda P_x, P_y: np.exp(-0.2*(2*rho2(P_x, P_y) - 1) + 0.1*(6*rho2(P_x, P_y)**2 - 6*rho2(P_x, P_y) + 1))
    I_c = sim.with_(projector=proj.with_(apodization=apo)).image(x, x)

    assert np.allclose(I_z, I_c)
    assert not np.allclose(I_z, sim.image(x, x))
    # a constant factor (Z_1) cancels in the normalized image:
    assert np.allclose(sim.with_(projector=proj.with_(apodization_zernikes={1: 0.5})).image(x, x),
                       sim.image(x, x))


# mask stacks:

def test_film_mask_matches_thin_mask_at_normal_incidence():
    film = FilmMask([2.343], [0.586], [68.0])
    t_abs, t_clear = film.transmissions(0.0, WL)
    t = t_abs[0]/t_clear[0]
    assert np.isclose(t_abs[0]/t_clear[0], t_abs[1]/t_clear[1])

    ill = Illumination.coherent(0.0, 0.0, polarization="x")
    proj = Projector(NA, WL, N_WATER)
    layout = MaskLayout.lines_spaces(150.0, 70.0)
    x = np.linspace(-75, 75, 51)
    I_film = LithoSimulation(layout, film, ill, proj).image(x)
    I_thin = LithoSimulation(layout, ThinMask(t), ill, proj).image(x)
    assert np.allclose(I_film, I_thin)


def test_film_mask_jones_is_diagonal_in_TE_TM():
    film = FilmMask([0.84, 2.0], [1.65, 0.1], [40.0, 20.0])
    s_x, s_y = 0.6, 0.5
    J_clear, J_abs = film.jones(s_x, s_y, NA, WL, 4.0)
    sin_theta = NA*np.hypot(s_x, s_y)/(4.0*1.563)
    t_abs, t_clear = film.transmissions(sin_theta, WL)
    e_TE = np.array([-s_y, s_x])/np.hypot(s_x, s_y)
    e_TM = np.array([s_x, s_y])/np.hypot(s_x, s_y)

    assert np.allclose(J_clear, np.eye(2))
    assert np.allclose(J_abs @ e_TE, t_abs[0]/t_clear[0]*e_TE)
    assert np.allclose(J_abs @ e_TM, t_abs[1]/t_clear[1]*e_TM)
    # an "absorber" of exit medium is clear:
    J_c, J_a = FilmMask([1.0], [0.0], [50.0]).jones(s_x, s_y, NA, WL, 4.0)
    assert np.allclose(J_a, J_c)


@pytest.mark.parametrize("layout", [MaskLayout.lines_spaces(120.0, 50.0),
                                    MaskLayout.contact_holes(120.0, 50.0, cd_y=60.0, tone="clear")])
def test_boundary_layer_with_absorber_transmission_is_a_bias(layout):
    w = 8.0
    ill = Illumination.annular(0.9, 0.5, n_samples=15, polarization="TE")
    proj = Projector(NA, WL, N_WATER)
    stack = ThinMask.binary(boundary_layer=BoundaryLayer(w, 0.0, 0.0))
    x = np.linspace(-60, 60, 25)
    y = None if layout.is_1d else x

    I_bl = LithoSimulation(layout, stack, ill, proj).image(x, y)
    I_bias = LithoSimulation(layout.scaled(bias=w), ThinMask.binary(), ill, proj).image(x, y)
    assert np.allclose(I_bl, I_bias)


def test_boundary_layer_polarization():
    layout = MaskLayout.lines_spaces(90.0, 45.0)
    ill = Illumination.dipole(0.95, 0.7, polarization="y")  # E parallel to the edges: TE
    proj = Projector(NA, WL, N_WATER)
    t_TE, t_TM = 0.3 + 0.2j, -0.2
    x = np.linspace(-45, 45, 31)
    I = LithoSimulation(layout, ThinMask.binary(boundary_layer=BoundaryLayer(5.0, t_TE, t_TM)), ill,
                        proj).image(x)
    I_TE = LithoSimulation(layout, ThinMask.binary(boundary_layer=BoundaryLayer(5.0, t_TE, 7.0)),
                           ill, proj).image(x)

    assert np.allclose(I, I_TE)
    I_x = LithoSimulation(layout, ThinMask.binary(boundary_layer=BoundaryLayer(5.0, t_TE, t_TM)),
                          ill.with_polarization("x"), proj).image(x)
    assert not np.allclose(I_x, I)


# wafer stack:

def test_index_matched_resist_gives_aerial_image():
    resist = WaferStack((N_WATER, 0.0, 200.0), substrate=(N_WATER, 0.0))
    ill = Illumination.dipole(0.95, 0.6, polarization="TM", n_samples=15)
    layout = MaskLayout.lines_spaces(90.0, 45.0)
    proj = Projector(NA, WL, N_WATER)
    sim = LithoSimulation(layout, ThinMask.binary(), ill, proj)
    x = np.linspace(-45, 45, 31)

    I_resist = sim.with_(wafer=resist).image(x, focus=-20.0, depth=70.0)
    assert np.allclose(I_resist, sim.image(x, focus=50.0))


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_resist_on_substrate_single_plane_wave(pol):
    n_r, d = 1.7 + 0.02j, 120.0
    sub = 0.883 + 2.778j
    stack = WaferStack((n_r.real, n_r.imag, d), substrate=(sub.real, sub.imag))
    k0 = 2*np.pi/WL
    z = np.linspace(0, d, 7)

    # normal incidence: analytic two-interface solution (tangential = total field)
    r01, t01 = (N_WATER - n_r)/(N_WATER + n_r), 2*N_WATER/(N_WATER + n_r)
    r10, r12 = -r01, (n_r - sub)/(n_r + sub)
    kz = k0*n_r
    E = t01*(np.exp(1j*kz*z) + r12*np.exp(1j*kz*(2*d - z)))/(1 - r10*r12*np.exp(2j*kz*d))
    down, up, _ = stack.factors(np.zeros(1), k0, N_WATER, z[:, None], pol)
    assert np.allclose((down + up)[:, 0], E)

    # oblique incidence on a semi-infinite, lossless resist: Fresnel
    semi = WaferStack((1.7, 0.0, d), substrate=(1.7, 0.0))
    theta = np.radians(50.0)
    K_x = np.array([[k0*N_WATER*np.sin(theta)]])
    E_in = np.zeros((3, 1, 1), dtype=complex)
    if pol == "TE":
        E_in[1] = 1.0
    else:
        E_in[0], E_in[2] = np.cos(theta), -np.sin(theta)
    E_t = semi.transmit(E_in, K_x, np.zeros_like(K_x), k0, N_WATER, 30.0)
    t_ref = (t_s if pol == "TE" else t_p)(theta, N_WATER, 1.7)
    assert np.isclose(np.linalg.norm(E_t), abs(t_ref))
    assert np.isclose(np.real(np.sum(E_t[:, 0, 0]*[K_x[0, 0], 0.0, k0*np.sqrt(1.7**2 - (K_x[0, 0]/k0)**2)])), 0, atol=1E-12)


@pytest.mark.parametrize("pol, contrast_ref", [("y", lambda th: 1.0),
                                               ("x", lambda th: abs(np.cos(2*th)))])
def test_two_beam_contrast_in_resist(pol, contrast_ref):
    """Orders -+1 of an alternating PSM (no order 0) under normal incidence
    interfere in a semi-infinite resist at the angles of the resist."""

    pitch, n_r = 200.0, 1.7
    layout = MaskLayout.lines_spaces(pitch, pitch/2)
    sim = LithoSimulation(layout, ThinMask(-1.0), Illumination.coherent(polarization=pol),
                          Projector(NA, WL, N_WATER), wafer=WaferStack((n_r, 0.0, 1E3), substrate=(n_r, 0.0)),
                          normalize=False)
    x = np.linspace(-pitch/2, pitch/2, 401)
    theta_r = np.arcsin(WL/(pitch*n_r))

    for depth in (0.0, 77.0):
        assert np.isclose(metrics.contrast(sim.image(x, depth=depth)), contrast_ref(theta_r))
    theta_w = np.arcsin(WL/(pitch*N_WATER))
    assert np.isclose(metrics.contrast(sim.with_(wafer=None).image(x)), contrast_ref(theta_w))


# metrics:

def test_threshold_cd_and_dose_to_size_of_cosine_image():
    p, V = 100.0, 0.6
    u = np.linspace(-p/2, p/2, 2001)
    I = 1 - V*np.cos(2*np.pi*u/p)  # dark line at 0
    th = 0.8

    assert np.isclose(metrics.threshold_cd(u, I, th), p/np.pi*np.arccos((1 - th)/V), atol=1E-3)
    bright = metrics.threshold_cd(u, I, th, center=p/2 - 1E-9, bright=True)
    assert np.isnan(bright)  # cut ends inside the bright region
    D = metrics.dose_to_size(u, I, 40.0, th)
    assert np.isclose(metrics.threshold_cd(u, D*I, th), 40.0, atol=1E-3)
    assert np.isnan(metrics.threshold_cd(u, I, 0.3))  # does not print
    slope = V*2*np.pi/p*np.sin(2*np.pi*20/p)/(1 - V*np.cos(2*np.pi*20/p))
    assert np.isclose(metrics.nils_at_threshold(u, D*I, th), 40*slope, rtol=1E-3)


def test_process_window_of_a_model_fem():
    c0, alpha, beta = 40.0, 2.0, 1E-3
    foci, doses = np.linspace(-100, 100, 41), np.linspace(0.8, 1.2, 81)
    D, F = np.meshgrid(doses, foci, indexing="ij")
    cd = c0*(1 - alpha*(D - 1)) + beta*F**2

    low, high = metrics.process_window(foci, doses, cd, c0, 0.1)
    d_low = 1 + (beta*foci**2/c0 - 0.1)/alpha
    d_high = 1 + (beta*foci**2/c0 + 0.1)/alpha
    assert np.allclose(low, np.clip(d_low, 0.8, 1.2), atol=2E-3)
    assert np.allclose(high, np.clip(d_high, 0.8, 1.2), atol=2E-3)

    dof, el = metrics.el_dof(foci, low, high)
    assert np.isclose(el[0], 0.2/alpha, rtol=0.02)
    assert np.all(np.diff(el) <= 1E-12)
    # window [-f, f]: common doses [1 + (beta*f^2/c0 - 0.1)/alpha, 1 + 0.1/alpha]
    f = np.linspace(0, 100, 100001)
    lo, hi = 1 + (beta*f**2/c0 - 0.1)/alpha, 1 + 0.1/alpha
    f_max = f[np.nonzero((hi - lo)/((hi + lo)/2) >= 0.05)[0][-1]]
    assert abs(metrics.dof_at_el(dof, el, 0.05) - 2*f_max) <= 2*(foci[1] - foci[0])


def test_fem_and_meef_of_large_features():
    layout = MaskLayout.lines_spaces(800.0, 400.0)
    sim = LithoSimulation(layout, ThinMask.binary(), Illumination.conventional(0.7, n_samples=15),
                          Projector(0.9, WL))
    assert np.isclose(sim.meef(0.3, dose=sim.dose_to_size(threshold=0.3)), 1.0, atol=0.05)

    foci, doses = np.linspace(-200, 200, 5), np.linspace(0.9, 1.1, 3)
    cd = sim.fem(foci, doses, 0.3)
    assert cd.shape == (3, 5)
    assert np.all(np.diff(cd, axis=0) < 0)  # lines shrink with dose
    assert np.allclose(cd, cd[:, ::-1], atol=1E-6)  # symmetric through focus


def test_contact_hole_cd_and_blur():
    layout = MaskLayout.contact_holes(110.0, 55.0)
    sim = LithoSimulation(layout, ThinMask.att_psm(0.06), Illumination.quasar(0.95, 0.7, polarization="TE"),
                          Projector(NA, WL, N_WATER))
    D = sim.dose_to_size(threshold=0.3)

    assert np.isclose(sim.cd(0.3, D), 55.0, atol=0.2)
    assert np.isclose(sim.cd(0.3, D, direction="y"), 55.0, atol=0.2)
    assert sim.nils() > 0
    x = grid1d(-55, 55, 64)
    I = sim.image(x, x)
    I_b = metrics.gaussian_blur(I, x[1] - x[0], 5.0)
    assert np.isclose(I_b.mean(), I.mean()) and metrics.contrast(I_b) < metrics.contrast(I)
