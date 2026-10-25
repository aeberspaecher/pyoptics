#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check the swing curve functions of pyoptics.litho.wafer and the resist
model of pyoptics.litho.resist.

References: energy conservation against pyoptics.thin_films, the period of
the swing curve, analytic solutions of diffusion and of the reactions in
the bake, the properties of the Mack rate, planar development fronts, and
the tone and dose dependence of the full chain.
"""

import numpy as np
import pytest

from pyoptics.litho import (Illumination, LithoSimulation, MaskLayout, Projector, ThinMask,
                            WaferStack, resist, wafer)
from pyoptics.thin_films import R_T_A_coeffs


WL = 193.0
N_WATER = 1.437


# swing curves:

@pytest.mark.parametrize("pol", ["TE", "TM"])
@pytest.mark.parametrize("na", [0.0, 0.7, 1.3])
def test_absorbed_power_conserves_energy(pol, na):
    """Only the resist absorbs: A = 1 - R - T of the whole stack."""

    stack = WaferStack((1.70, 0.04, 180.0), below=[(1.9, 0.0, 60.0)], substrate=(1.5, 0.0),
                       above=[(1.55, 0.0, 30.0)])
    R, T, A = R_T_A_coeffs(np.arcsin(na/N_WATER), np.array([1.55, 1.70, 1.9]), np.array([0, 0.04, 0]),
                           np.array([30.0, 180.0, 60.0]), N_WATER, 0, 1.5, 0, pol, WL)[:3]
    assert np.isclose(stack.reflectance(na, WL, N_WATER, pol), R, rtol=1e-12)
    assert np.isclose(stack.absorbed(na, WL, N_WATER, pol), A, rtol=1e-10)


def test_absorbed_in_a_semi_infinite_resist():
    n_r = 1.7 + 0.05j
    d = 150.0
    stack = WaferStack((n_r.real, n_r.imag, d), substrate=(n_r.real, n_r.imag))
    R = abs((N_WATER - n_r)/(N_WATER + n_r))**2
    expected = (1 - R)*(1 - np.exp(-4*np.pi*n_r.imag*d/WL))
    assert np.isclose(stack.absorbed(0.0, WL, N_WATER), expected, rtol=1e-10)


def test_swing_period_and_barc():
    n_r = 1.70
    stack = WaferStack((n_r, 0.02, 100.0))
    d = np.linspace(100.0, 300.0, 401)
    sc = wafer.swing_curve(stack, d, WL, N_WATER)
    minima = d[1:-1][(sc.reflectance[1:-1] < sc.reflectance[:-2]) & (sc.reflectance[1:-1] < sc.reflectance[2:])]
    assert np.allclose(np.diff(minima), WL/(2*n_r), atol=1.0)
    # absorption maxima where the reflection is minimal
    assert sc.swing(sc.absorbed) > 0.3

    na = np.linspace(0.0, 1.2, 7)
    best, R_b = wafer.optimize_barc(stack, (1.82, 0.34), np.linspace(10, 120, 56), WL, N_WATER, na=na)
    with_barc = stack.with_(below=[(1.82, 0.34, best)])
    assert R_b.min() < 0.01 < with_barc.with_(below=[]).substrate_reflectance(na, WL, N_WATER).mean()
    assert np.isclose(with_barc.substrate_reflectance(na, WL, N_WATER).mean(), R_b.min(), atol=1e-3)
    sc_barc = wafer.swing_curve(with_barc, d, WL, N_WATER)
    assert np.ptp(sc_barc.reflectance) < 0.2*np.ptp(sc.reflectance)


def test_with_thickness():
    stack = WaferStack((1.7, 0.02, 100.0), below=[(1.8, 0.3, 40.0)])
    assert stack.with_(thickness=250.0).resist == (1.7, 0.02, 250.0)
    assert stack.with_(thickness=250.0).below == stack.below


# resist model parts:

def test_mack_rate():
    r_max, r_min, n, m_th = 120.0, 0.2, 6.0, 0.4
    assert np.isclose(resist.mack_rate(1.0, r_max, r_min, n, m_th), r_min)
    assert np.isclose(resist.mack_rate(0.0, r_max, r_min, n, m_th), r_max + r_min)
    h = 1e-4  # inflection point at m_th
    r = resist.mack_rate(m_th + np.array([-h, 0, h]), r_max, r_min, n, m_th)
    assert abs(r[0] - 2*r[1] + r[2])/h**2 < 1e-3*r_max


def grid(N_z=24, N_x=32, d=120.0, pitch=160.0):
    x = pitch*(np.arange(N_x)/N_x - 0.5)
    z = (np.arange(N_z) + 0.5)*d/N_z
    return x, z, pitch/N_x, d/N_z


def test_bake_without_diffusion():
    x, z, dx, dz = grid()
    h_0 = 0.2 + 0.5*np.cos(2*np.pi*x/160.0)**2*np.ones((len(z), 1))
    model = resist.ResistModel(diffusion_length=0.0, amplification=2.5)
    h, m = model.bake(h_0, dx, dz)
    assert np.allclose(h, h_0) and np.allclose(m, np.exp(-2.5*h_0))
    # instantaneous neutralization by the quencher:
    h, m = model.with_(quencher=0.3).bake(h_0, dx, dz)
    assert np.allclose(h, np.maximum(h_0 - 0.3, 0)) and np.allclose(m, np.exp(-2.5*h))
    # a fast finite rate approaches it:
    h_fast = model.with_(quencher=0.3, neutralization=1e4).bake(h_0, dx, dz)[0]
    assert np.allclose(h_fast, h, atol=2e-3)


def test_bake_with_diffusion():
    """Cosines in x and z (no flux through the top and the bottom) decay as
    exp(-k**2*sigma**2*tau/2); the inhibitor is exp(-K*int h dtau)."""

    x, z, dx, dz = grid()
    d, sigma, K = 120.0, 25.0, 2.0
    k2 = (2*np.pi/80.0)**2 + (np.pi/d)**2
    h_0 = 0.4 + 0.3*np.cos(2*np.pi*x/80.0)[None, :]*np.cos(np.pi*z/d)[:, None]
    model = resist.ResistModel(diffusion_length=sigma, amplification=K, n_steps=200)
    h, m = model.bake(h_0, dx, dz)
    g = np.exp(-k2*sigma**2/2)
    assert np.allclose(h, 0.4 + (h_0 - 0.4)*g, atol=1e-12)
    integral = 0.4 + (h_0 - 0.4)*(1 - g)/(k2*sigma**2/2)
    assert np.allclose(m, np.exp(-K*integral), rtol=1e-4)
    assert np.isclose(h.mean(), h_0.mean())


def test_planar_development():
    x, z, dx, dz = grid()
    T = resist.develop(np.full((len(z), len(x)), 3.0), dx, dz)
    assert np.allclose(T, z[:, None]/3.0*np.ones(len(x)))


def test_development_of_a_trench():
    """A soluble column in an insoluble resist develops straight down; the
    front spreads sideways at the slow rate only."""

    x, z, dx, dz = grid(N_x=64)
    w = 40.0
    rate = np.where(np.abs(x) < w/2, 100.0, 0.05)*np.ones((len(z), 1))
    T = resist.develop(rate, dx, dz)
    t_dev = 1.5*120.0/100.0
    latent = resist.LatentImage(x, z, None, 0.0, True, 120.0, 160.0)
    p = resist.profile(latent, T, t_dev, rate)
    w_cells = dx*np.sum(np.abs(x) < w/2)  # the width of the soluble cells
    assert np.all((p.cd >= w_cells) & (p.cd < w_cells + 2*0.05*t_dev))
    assert np.isclose(p.sidewall_angle, 90.0, atol=1.0)
    assert np.isclose(p.top_loss, 0.05*t_dev, rtol=1e-6)
    # a resist line between such trenches: complementary profile
    latent_line = resist.LatentImage(x, z, None, 0.0, False, 120.0, 160.0)
    T_line = resist.develop(np.where(np.abs(x) < w/2, 0.05, 100.0)*np.ones((len(z), 1)), dx, dz)
    p_line = resist.profile(latent_line, T_line, t_dev, np.where(np.abs(x) < w/2, 0.05, 100.0)*np.ones((len(z), 1)))
    assert np.all((p_line.cd <= w_cells) & (p_line.cd > w_cells - 2*0.05*t_dev))


def test_sloped_walls():
    x, z, dx, dz = grid(N_x=128)
    half = 30.0 - 0.1*z  # an opening that narrows towards the bottom: 84.3 degrees
    rate = np.where(np.abs(x)[None, :] < half[:, None], 100.0, 1e-3)
    latent = resist.LatentImage(x, z, None, 0.0, True, 120.0, 160.0)
    p = resist.profile(latent, resist.develop(rate, dx, dz), 2.0, rate)
    assert np.isclose(p.sidewall_angle, 90 - np.degrees(np.arctan(0.1)), atol=1.5)
    assert p.cd_top > p.cd_middle > p.cd_bottom


def test_dose_to_clear_of_a_uniform_resist():
    """Uniform intensity, no diffusion: the front moves at the constant rate
    r(m) with m = exp(-K*(1 - exp(-C*D*I))); it clears d in t_dev for
    r = d/t_dev."""

    d, t_dev, I, C, K = 100.0, 60.0, 0.8, 0.05, 2.0
    model = resist.ResistModel(C=C, diffusion_length=0.0, amplification=K, develop_time=t_dev)
    z = (np.arange(20) + 0.5)*d/20
    lat = resist.LatentImage(np.zeros(1), z, np.full((20, 1), I), 0.0, True, d, 1.0)
    ms = np.linspace(0, 1, 200001)
    r = model.rate(ms)
    m_star = np.interp(d/t_dev, r[::-1], ms[::-1])
    D_expected = -np.log(1 + np.log(m_star)/K)/(C*I)
    assert np.isclose(model.dose_to_clear(lat), D_expected, rtol=1e-4)


# the full chain:

def immersion_sim(layout, **kwargs):
    stack = WaferStack((1.70, 0.03, 100.0), below=[(1.82, 0.34, 40.0)])
    return LithoSimulation(layout, ThinMask.att_psm(0.06),
                           Illumination.dipole(0.95, 0.7, polarization="y"),
                           Projector(1.35, WL, n=N_WATER), wafer=stack, **kwargs)


def test_intensity_in_resist():
    sim = immersion_sim(MaskLayout.lines_spaces(90.0, 45.0))
    lat = resist.intensity_in_resist(sim, N_x=32, N_z=10)
    assert lat.I.shape == (10, 32) and np.isclose(lat.dz, 10.0) and np.isclose(lat.dx, 90/32)
    open_frame = sim.with_(wafer=None, normalize=False).open_frame()
    raw = sim.with_(normalize=False).image(lat.x, depth=lat.z[3])
    assert np.allclose(lat.I[3], raw/open_frame)
    # the normalization of the simulation does not matter:
    assert np.allclose(resist.intensity_in_resist(sim.with_(normalize=False), N_x=32, N_z=10).I, lat.I)
    with pytest.raises(ValueError):
        resist.intensity_in_resist(sim.with_(wafer=None))
    # the open frame in the resist: clear-mask intensities through depth
    of = resist.open_frame_in_resist(sim, N_z=10)
    assert np.allclose(of.I[:, 0], [sim.with_(normalize=False).open_frame(0.0, z)/open_frame for z in lat.z])
    model = resist.ResistModel()
    E_0 = model.dose_to_clear(of)
    assert np.all(model.process(of, 1.01*E_0).arrival[-1] < model.develop_time)
    assert np.all(model.process(of, 0.9*E_0).arrival[-1] > model.develop_time)


@pytest.mark.parametrize("tone", ["clear", "dark"])
def test_dose_and_tone(tone):
    """Lines (clear field mask) shrink with dose, trenches (dark field mask)
    grow; dose to size hits the target."""

    sim = immersion_sim(MaskLayout.lines_spaces(90.0, 45.0, tone=tone))
    lat = resist.intensity_in_resist(sim, N_x=64, N_z=20)
    assert lat.opening == (tone == "dark")
    model = resist.ResistModel(diffusion_length=10.0, quencher=0.1)
    D = model.dose_to_size(lat, 45.0)
    assert np.isclose(model.process(lat, D).profile.cd_middle, 45.0, atol=0.05)
    cds = [model.process(lat, f*D).profile.cd_middle for f in (0.9, 1.1)]
    assert (cds[0] > 45.0 > cds[1]) if tone == "clear" else (cds[0] < 45.0 < cds[1])
    res = model.process(lat, D)
    assert res.resist.shape == lat.I.shape
    assert np.all((res.inhibitor > 0) & (res.inhibitor <= 1)) and np.all(res.acid <= res.acid_0.max())
