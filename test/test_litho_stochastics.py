#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.litho.stochastics, flare and focus_blur (notebook 152,
litho stochastics, flare and focus blur)."""

import numpy as np
import pytest

from pyoptics.litho import (Illumination, LithoSimulation, MaskLayout, Projector, ThinMask, flare,
                            focus_blur, lens, resist, stochastics as st)


# ------------------------------------------------------------------ stochastics

def test_photons_per_area():
    assert st.photons_per_area(1.0, 193.368) == pytest.approx(9.73, rel=2e-3)
    assert st.photons_per_area(1.0, 248.3) == pytest.approx(12.5, rel=2e-3)
    assert st.photons_per_area(1.0, 13.5) == pytest.approx(0.680, rel=2e-3)
    assert st.photons_per_area(30.0, 193.368)/st.photons_per_area(30.0, 13.5) == pytest.approx(
        193.368/13.5)
    assert st.photon_energy(13.5) == pytest.approx(1.4714e-17, rel=1e-4)


@pytest.fixture(scope="module")
def lines():
    sim = LithoSimulation(MaskLayout.lines_spaces(90.0, 45.0), ThinMask.binary(),
                          Illumination.dipole(0.9, 0.7, n_samples=9, polarization="y"),
                          Projector(1.35, 193.368, n=1.437))
    x = np.arange(-45.0, 45.0, 1.0)
    return x, sim.image(x)


def test_mean_chain_matches_resist_model():
    model = resist.ResistModel(C=0.05, diffusion_length=8.0, amplification=3.0, quencher=0.2,
                               quencher_diffusion_length=4.0, n_steps=10)
    R = st.StochasticResist(model, development_length=0.0)
    x = np.arange(0.0, 64.0)
    I = 0.5 + 0.4*np.cos(2*np.pi*x/64.0)
    out = R.mean(np.tile(I, (8, 1)), 30.0, 1.0)
    acid_0 = model.expose(I, 30.0)
    _, m = model.bake(np.tile(acid_0, (3, 1)), 1.0, 1.0)  # uniform in z: no z flux either way
    assert np.allclose(out.m, m[0][None, :], atol=1e-12)
    assert np.allclose(out.photons, R.absorbed_photons(30.0)*I[None, :])


def test_bake_quencher_array_and_periodic():
    model = resist.ResistModel(quencher=0.1, quencher_diffusion_length=5.0, n_steps=8)
    rng = np.random.default_rng(0)
    acid = rng.uniform(0, 1, (6, 32))
    h1, m1 = model.bake(acid, 1.0, 1.0)
    h2, m2 = model.bake(acid, 1.0, 1.0, quencher=np.full_like(acid, 0.1))
    assert np.allclose(m1, m2) and np.allclose(h1, h2)
    # without diffusion along the first axis the two boundary conditions agree
    a = np.tile(acid[:1], (6, 1))
    assert np.allclose(model.bake(a, 1.0, 1.0)[1], model.bake(a, 1.0, 1.0, periodic_z=True)[1])


def test_count_statistics():
    R = st.StochasticResist(noise=("photon", "acid"))
    I = np.full((200, 200), 0.5)
    r = R.process(I, 20.0, 1.0, rng=1)
    n_bar = R.absorbed_photons(20.0)*0.5
    assert np.mean(r.photons) == pytest.approx(n_bar, rel=0.02)
    assert np.var(r.photons) == pytest.approx(n_bar, rel=0.05)
    V = R.thickness
    mean_acid = R.pag*V*(1 - np.exp(-R.model.C*20.0*0.5))
    y = R.acid_yield(0.5, 20.0, 1.0)
    assert y == pytest.approx(mean_acid/n_bar) and y < 1
    # binomial thinning of Poisson photons: a Poisson acid count with the mean of the mean chain
    assert np.mean(r.acid_0) == pytest.approx(mean_acid, rel=0.02)
    assert np.var(r.acid_0) == pytest.approx(mean_acid, rel=0.05)
    assert np.all(r.acid_0 <= r.photons)
    photons_only = R.with_(noise=("photon",)).process(I, 20.0, 1.0, rng=1)
    assert np.var(photons_only.acid_0) == pytest.approx(y*mean_acid, rel=0.05)
    # more than one acid per photon (EUV): over-dispersed by 1 + y
    euv = R.with_(wavelength=13.5)
    y_euv = euv.acid_yield(0.5, 20.0, 1.0)
    assert y_euv > 1
    big = euv.process(np.full((400, 400), 0.5), 20.0, 1.0, rng=2).acid_0
    assert np.mean(big) == pytest.approx(mean_acid, rel=0.02)
    assert np.var(big) == pytest.approx((1 + y_euv)*mean_acid, rel=0.05)
    assert np.all(r.quencher_0 == R.model.quencher*R.pag*V)  # no quencher noise


def test_line_roughness(lines):
    x, I1 = lines
    R = st.StochasticResist()
    D = R.dose_to_size(x, I1, 45.0)
    assert R.mean_cd(x, I1, D) == pytest.approx(45.0, abs=1e-3)
    y = np.arange(0.0, 512.0, 1.0)
    I = np.tile(I1, (len(y), 1))
    flat = R.mean(I, D, 1.0)
    rf = st.roughness(*st.line_edges(x, flat.m, 0.5))
    assert rf.ler < 1e-9 and rf.cd == pytest.approx(45.0, abs=1e-3)
    rough = st.roughness(*st.line_edges(x, R.process(I, D, 1.0, rng=2).m, 0.5))
    assert 1.0 < rough.ler < 6.0 and rough.lwr > rough.ler and rough.broken == 0.0
    # the same resist at EUV: 14 times fewer photons, more roughness from the photons
    photons = R.with_(noise=("photon",))
    ler_arf = st.roughness(*st.line_edges(x, photons.process(I, D, 1.0, rng=3).m, 0.5)).ler
    ler_euv = st.roughness(*st.line_edges(x, photons.with_(wavelength=13.5).process(
        I, D, 1.0, rng=3).m, 0.5)).ler
    assert ler_euv > 2.5*ler_arf


def test_roughness_independent_of_grid(lines):
    x, I1 = lines
    R = st.StochasticResist(noise=("acid", "quencher"))
    D = R.dose_to_size(x, I1, 45.0)
    lers = []
    for dx in (1.0, 0.5):
        xs = np.arange(-45.0, 45.0, dx)
        Is = np.interp(xs, x, I1, period=90.0)
        y = np.arange(0.0, 1024.0, dx)
        ler = [st.roughness(*st.line_edges(xs, R.process(np.tile(Is, (len(y), 1)), D, dx,
                                                         rng=s).m, 0.5)).ler for s in range(3)]
        lers.append(np.mean(ler))
    assert lers[1] == pytest.approx(lers[0], rel=0.2)


def test_psd_normalization_and_fit():
    N, dy = 1024, 1.0
    y = np.arange(N)*dy
    e = st.random_edges(N, dy, 2.0, 15.0, 0.7, n_edges=200, rng=4)
    f, p = st.edge_psd(y, e)
    assert np.sum(p)*(f[1] - f[0]) == pytest.approx(np.mean(np.var(e, axis=1)), rel=1e-10)
    assert np.mean(np.var(e, axis=1)) == pytest.approx(4.0, rel=0.1)
    sigma, xi, H = st.fit_psd(f, p)
    assert sigma == pytest.approx(2.0, rel=0.1)
    assert xi == pytest.approx(15.0, rel=0.15)
    assert H == pytest.approx(0.7, abs=0.1)
    assert st.fit_psd(f, p, H=0.7)[1] == pytest.approx(15.0, rel=0.1)
    # model normalization: one-sided integral is sigma**2
    from scipy.integrate import quad
    for H in (0.3, 0.5, 1.0):
        assert quad(lambda f: st.psd_model(f, 1.5, 3.0, H), 0, np.inf)[0] == pytest.approx(2.25,
                                                                                            rel=1e-6)


def test_independent_edges_lwr():
    e = st.random_edges(2048, 1.0, 1.0, 10.0, 0.5, n_edges=2, rng=5)
    r = st.roughness(e[0] - 20.0, e[1] + 20.0)
    assert r.cd == pytest.approx(40.0, abs=1e-9)
    assert abs(r.correlation) < 0.15
    assert r.lwr == pytest.approx(np.sqrt(2*(1 - r.correlation))*r.ler, rel=0.05)


def test_contact_stats():
    p = 20.0
    x = np.arange(-p/2, -p/2 + 4*p, 0.5)
    y = np.arange(-p/2, -p/2 + 3*p, 0.5)
    X, Y = np.meshgrid(x, y)
    open_ = np.zeros(X.shape, dtype=bool)
    for i in range(4):
        for j in range(3):
            if (i, j) == (1, 1):
                continue  # missing
            open_ |= np.hypot(X - i*p, Y - j*p) < 4.0 + (i == 3)
    open_ |= (np.abs(Y) < 2.0) & (X > 0) & (X < p)  # merges holes (0, 0) and (1, 0)
    s = st.contact_stats(x, y, open_, p)
    assert s.cd.shape == (3, 4)
    assert s.missing[1, 1] and s.missing.sum() == 1
    assert s.merged[0, 0] and s.merged[0, 1] and s.merged.sum() == 2
    assert s.good.sum() == 9
    assert s.mean_cd == pytest.approx((6*8.0 + 3*10.0)/9, rel=0.03)
    assert s.lcdu > 0


def test_stochastic_contacts_fail_at_low_dose():
    p = 130.0
    sim = LithoSimulation(MaskLayout.contact_holes(p, 65.0), ThinMask.att_psm(),
                          Illumination.quasar(0.9, 0.6, n_samples=9), Projector(1.35, 193.368, n=1.437))
    dx = 2.0
    x = np.arange(-p/2, -p/2 + 4*p, dx)
    I = sim.image(x, x)
    R = st.StochasticResist()
    D = R.dose_to_size(x[:int(p/dx)], I[int(p/dx)//2, :int(p/dx)], 60.0, feature="space")
    good = st.contact_stats(x, x, R.process(I, D, dx, rng=1).cleared, p)
    assert good.missing_rate == 0 and good.merged_rate == 0
    assert good.mean_cd == pytest.approx(60.0, rel=0.1)
    bad = st.contact_stats(x, x, R.process(I, 0.6*D, dx, rng=1).cleared, p)
    assert bad.missing_rate > 0.5


def test_unknown_noise_and_feature():
    with pytest.raises(ValueError):
        st.StochasticResist(noise=("photons",))
    with pytest.raises(ValueError):
        st.line_edges(np.arange(4.0), np.ones((2, 4)), 0.5, feature="hole")


# ------------------------------------------------------------------------ flare

def test_tis():
    assert flare.tis(0.1, 13.5) == pytest.approx((4*np.pi*0.1/13.5)**2)
    assert flare.tis(0.3, 193.0, n=1.5) == pytest.approx((np.pi*0.3/193.0)**2)
    assert flare.tis(0.2, 13.5)/flare.tis(0.2, 193.0) == pytest.approx((193.0/13.5)**2)


def test_power_law_flare():
    F = flare.Flare.power_law(0.03, 1.5, 1e3, 1e6)
    assert F.encircled([1e6, 1e7]) == pytest.approx([0.03, 0.03], rel=1e-6)
    assert F.encircled(1e4)[0] < 0.01
    assert F.level(0.4) == pytest.approx(0.012)
    assert F.apply(1.0, F.level(1.0)) == pytest.approx(1.0)  # a clear field stays at 1
    F2 = flare.Flare.power_law(0.02, 2.0, 1e3, 1e5)  # the logarithmic case
    assert F2.encircled(1e5)[0] == pytest.approx(0.02, rel=1e-6)


def test_flare_level_map():
    F = flare.Flare.power_law(0.03, 1.5, 1e3, 1e6)
    dx = 1e4
    uniform = F.level_map(np.full((256, 256), 0.5), dx, pad=False)
    assert np.allclose(uniform, 0.015, rtol=1e-3)
    d = np.zeros((100, 100))
    d[:, 50:] = 1.0
    L = F.level_map(d, dx)
    row = L[50]
    assert np.all(np.diff(row[5:45]) > 0)   # rises towards the clear half
    assert row[75] > row[25] and row.max() < 0.03


def test_flare_from_roughness():
    psd = flare.ABCRoughness(0.5, B=2e4, C=1.5, f_min=1e-6, f_max=1e-3)
    f = np.logspace(-6, -3, 20001)
    assert np.trapezoid(2*np.pi*f*psd(f), f) == pytest.approx(0.25, rel=1e-3)
    F = flare.Flare.from_roughness(psd, 193.0, 1e8, n=1.56, n_surfaces=20)
    assert F.total == pytest.approx(20*flare.tis(0.5, 193.0, 1.56))
    assert F.r_min == pytest.approx(193.0*1e8*1e-6)
    assert F.encircled(1.01*F.r_max)[0] == pytest.approx(F.total, rel=2e-3)
    mirrors = flare.Flare.from_roughness(psd, 13.5, 1e8, n_surfaces=6)
    assert mirrors.total == pytest.approx(6*flare.tis(0.5, 13.5))


# ------------------------------------------------------------------- focus blur

def test_focus_distribution():
    B = focus_blur.FocusBlur(z_msd=30.0, n_foci=21)
    z, w, t = B.focus_distribution()
    assert np.sum(w) == pytest.approx(1.0) and t < 1e-4
    assert np.sum(w*z) == pytest.approx(0.0, abs=1e-9)
    assert np.sqrt(np.sum(w*z**2)) == pytest.approx(30.0, rel=0.03)
    G = focus_blur.FocusBlur(dz_dlambda=200.0, bandwidth=0.3, spectrum="gaussian", z_msd=20.0)
    s = 60.0/(2*np.sqrt(2*np.log(2)))
    z, w, _ = G.focus_distribution()
    assert np.sqrt(np.sum(w*z**2)) == pytest.approx(np.hypot(s, 20.0), rel=0.03)
    L = focus_blur.FocusBlur(dz_dlambda=200.0, bandwidth=0.3)
    z, w, t = L.focus_distribution()
    assert t == pytest.approx(1 - 2/np.pi*np.arctan(120.0/30.0), rel=1e-3)
    assert np.allclose(z, -z[::-1]) and np.allclose(w, w[::-1])
    none = focus_blur.FocusBlur()
    assert none.focus_distribution()[0].tolist() == [0.0]
    with pytest.raises(ValueError):
        focus_blur.FocusBlur(dz_dlambda=1.0, bandwidth=1.0, spectrum="flat").focus_distribution()


def test_e95():
    assert focus_blur.e95(1.0) == pytest.approx(12.706, rel=1e-3)
    assert focus_blur.e95(1.0, "gaussian") == pytest.approx(1.6646, rel=1e-3)


@pytest.fixture(scope="module")
def krf():
    sim = LithoSimulation(MaskLayout.lines_spaces(500.0, 250.0), ThinMask.binary(),
                          Illumination.conventional(0.6, n_samples=9), Projector(0.56, 248.3))
    return sim, np.arange(-250.0, 250.0, 2.0)


def test_blurred_images(krf):
    sim, x = krf
    assert np.allclose(focus_blur.FocusBlur().image(sim, x, focus=100.0), sim.image(x, focus=100.0))
    two = focus_blur.FocusBlur(dz_dlambda=250.0, spectrum=([-0.2, 0.2], [1.0, 3.0]))
    expected = 0.25*sim.image(x, focus=50.0) + 0.75*sim.image(x, focus=-50.0)
    assert np.allclose(two.image(sim, x), expected)
    G = focus_blur.FocusBlur(z_msd=80.0)
    z, w, _ = G.focus_distribution()
    I = G.through_focus(sim, x, [0.0, 300.0])
    assert np.allclose(I[1], np.tensordot(w, sim.through_focus(x, 300.0 + z), axes=1))
    # the focus spread lowers the contrast at best focus
    I0 = sim.image(x)
    c = lambda I: (I.max() - I.min())/(I.max() + I.min())
    assert c(I[0]) < c(I0)
    blurred = G.with_(z_msd=0.0, xy_msd=20.0).image(sim, x)
    assert c(blurred) < c(I0) and np.mean(blurred) == pytest.approx(np.mean(I0))


def test_axial_colour_caldwell():
    from pyoptics.raytrace import caldwell_krf_lens
    assert lens.axial_colour(caldwell_krf_lens()) == pytest.approx(241.0, rel=0.02)
