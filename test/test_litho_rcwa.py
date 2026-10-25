#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Check pyoptics.litho.rcwa: RCWA of lamellar gratings and the rigorous
mask stack.

References: unpatterned stacks against pyoptics.thin_films, energy
conservation of lossless gratings (planar and conical), the Born
approximation of a weak thin grating, Li's factorization against Laurent's,
symmetries, the film mask limit of large pitches, Hopkins = Abbe for
coherent normal incidence, and the boundary layer calibration.
"""

import numpy as np
import pytest

from pyoptics.litho import (FilmMask, Illumination, LithoSimulation, MaskLayout, Projector,
                            RigorousMask, hopkins, rcwa)
from pyoptics.litho.mask import _rect_coeffs
from pyoptics.polarization import TE_TM_basis
from pyoptics.thin_films import _r_t


WL = 193.0
N_BLANK = 1.563
MOSI = 2.343 + 0.586j


@pytest.mark.parametrize("k_x, k_y", [(0.0, 0.0), (0.3, 0.0), (0.2, 0.25)])
def test_uniform_layers_are_thin_films(k_x, k_y):
    layers = [rcwa.Layer(40.0, MOSI), rcwa.Layer(30.0, 1.9)]
    res = rcwa.rcwa(500.0, layers, WL, k_x, k_y, n_in=N_BLANK, n_out=1.0, n_orders=3)
    n = np.array([N_BLANK, MOSI, 1.9, 1.0])
    sin_theta = np.hypot(k_x, k_y)/N_BLANK
    t_TE, t_TM = (_r_t(sin_theta, n, [40.0, 30.0], pol, WL)[1] for pol in ("TE", "TM"))
    e_TE, e_TM = np.array(TE_TM_basis(k_x, k_y)) if k_x or k_y else (np.array([0, 1.0]), np.array([1.0, 0]))
    J = t_TE*np.outer(e_TE, e_TE) + t_TM*np.outer(e_TM, e_TM)
    assert np.allclose(res.t[0, 3], J, atol=1e-13)
    assert np.allclose(np.delete(res.t[0], 3, axis=0), 0)


@pytest.mark.parametrize("k_x, k_y", [(0.0, 0.0), (0.3, 0.0), (0.2, 0.4), (-0.5, 0.7)])
def test_energy_conservation(k_x, k_y):
    layers = [rcwa.Layer(120.0, 1.0, [(0.0, 250.0, 1.5)]), rcwa.Layer(50.0, 1.3, [(100.0, 100.0, 2.0)])]
    res = rcwa.rcwa(500.0, layers, WL, k_x, k_y, n_in=N_BLANK, n_out=1.0, n_orders=15)
    for jones in ([1, 0], [0, 1], [1, 1j]):
        R, T = res.efficiencies(jones)
        assert np.isclose(R.sum() + T.sum(), 1.0, atol=1e-12)
        assert np.all(R >= -1e-15) and np.all(T >= -1e-15)


def test_weak_thin_grating_born_approximation():
    """A thin, weak grating (thickness h, ridges of permittivity eps)
    scatters the order m with the amplitude i*k_0*h*(eps - 1)*c_m/(2*K_z,m)
    for E along the lines (TE)."""

    P, w, h, n = 600.0, 200.0, 0.2, 1.5
    res = rcwa.rcwa(P, [rcwa.Layer(h, 1.0, [(0.0, w, n)])], WL, n_orders=12)
    for m in (1, 2):
        Kz = np.sqrt(1 - (m*WL/P)**2)
        born = 1j*2*np.pi/WL*h*(n**2 - 1)*_rect_coeffs(m, P, 0.0, w)/(2*Kz)
        assert np.isclose(res.t[0, 12 + m, 1, 1], born, rtol=0.02)


def test_li_factorization_converges_faster():
    layers = [rcwa.Layer(70.0, 1.0, [(0.0, 240.0, 0.84 + 1.65j)])]  # chrome lines, TM

    def t1(M, factorization):
        return rcwa.rcwa(480.0, layers, WL, 0.1, 0.0, n_in=N_BLANK, n_orders=M,
                         factorization=factorization).t[0, M + 1, 0, 0]

    ref = t1(100, "li")
    assert abs(t1(10, "li") - ref) < 0.3*abs(t1(10, "laurent") - ref)
    assert abs(t1(40, "li") - ref) < 1e-3


def test_symmetric_grating_at_normal_incidence():
    res = rcwa.rcwa(400.0, [rcwa.Layer(68.0, 1.0, [(0.0, 200.0, MOSI)])], WL, n_in=N_BLANK, n_orders=12)
    assert np.allclose(res.t[0], res.t[0, ::-1], atol=1e-12)
    assert np.allclose(res.t[0, :, 0, 1], 0) and np.allclose(res.t[0, :, 1, 0], 0)


# the rigorous mask stack:

def immersion():
    return Projector(1.35, WL, n=1.437)


def test_large_pitch_approaches_the_film_mask():
    rig = RigorousMask([(MOSI, 68.0)])
    film = rig.film_mask()
    layout = MaskLayout.lines_spaces(2000.0, 1000.0)
    m = np.arange(-2, 3)
    z = np.zeros(1)
    J_o = rig.order_jones(layout, m, [0], z, z, 1.35, WL, 4.0)[0][:, :, 0]
    c = layout.region_coefficients(m, [0])
    J_abs = film.jones(z, z, 1.35, WL, 4.0)[1][0]
    thin = c["field"][0] + c["feature"][0]*J_abs[0, 0]
    assert np.allclose(J_o[0, 0], thin, atol=0.01) and np.allclose(J_o[1, 1], thin, atol=0.01)


def test_open_frame_and_orientation():
    clear = RigorousMask([(1.0, 68.0)])  # "absorber" of exit medium: an open frame
    x = np.linspace(-90, 90, 31)
    sim = LithoSimulation(MaskLayout.lines_spaces(180.0, 90.0), clear,
                          Illumination.annular(0.9, 0.6, n_samples=9), immersion())
    assert np.allclose(sim.image(x), 1.0, atol=1e-10)

    rig = RigorousMask([(MOSI, 68.0)])
    lines_y = LithoSimulation(MaskLayout.lines_spaces(180.0, 90.0), rig,
                              Illumination.dipole(0.9, 0.6, n_samples=11, polarization="x"), immersion())
    lines_x = LithoSimulation(MaskLayout.lines_spaces(180.0, 90.0, orientation="x"), rig,
                              Illumination.dipole(0.9, 0.6, axis="y", n_samples=11, polarization="y"),
                              immersion())
    assert np.allclose(lines_y.cut(N=41)[1], lines_x.cut(N=41)[1], atol=1e-12)
    with pytest.raises(ValueError):
        LithoSimulation(MaskLayout.contact_holes(200.0, 100.0), rig, Illumination.coherent(),
                        immersion()).image(x)


def test_hopkins_with_a_rigorous_mask():
    rig = RigorousMask([(MOSI, 68.0)])
    layout = MaskLayout.lines_spaces(180.0, 90.0)
    x = np.linspace(-90, 90, 31)
    coherent = LithoSimulation(layout, rig, Illumination.coherent(polarization="x"), immersion())
    assert np.allclose(hopkins.TCC(coherent).image(x)[0], coherent.image(x), atol=1e-12)
    # extended source: the Hopkins approximation (orders at normal incidence)
    sim = coherent.with_(illumination=Illumination.annular(0.9, 0.6, n_samples=9))
    tcc = hopkins.TCC(sim)
    assert tcc.channels == [(0, 0), (1, 1)]
    assert 1e-4 < np.abs(tcc.image(x)[0] - sim.image(x)).max() < 0.05


def test_order_cache():
    rig = RigorousMask([(MOSI, 68.0)])
    layout = MaskLayout.lines_spaces(180.0, 90.0)
    args = (layout, np.arange(-2, 3), [0], np.array([0.1, 0.2]), np.array([0.0, 0.3]), 1.35, WL, 4.0)
    a = rig.order_jones(*args)
    assert rig.order_jones(*args) is a and a.shape == (2, 2, 2, 1, 5)


def test_boundary_layer_calibration():
    rig = RigorousMask([(MOSI, 68.0)])
    layouts = [MaskLayout.lines_spaces(p, p/2) for p in (120.0, 180.0, 300.0)]
    cal = rcwa.calibrate_boundary_layer(rig, layouts, 1.35, WL, widths=np.linspace(1, 15, 15))
    assert cal.residual.min() < cal.residual[0] and cal.residual.min() < 0.7*cal.residual_base
    assert isinstance(cal.stack, FilmMask) and cal.stack.boundary_layer is cal.boundary_layer
    sim = LithoSimulation(MaskLayout.lines_spaces(150.0, 75.0), rig,
                          Illumination.annular(0.9, 0.6, n_samples=11), immersion())
    cd_rig, cd_film, cd_bl = (sim.with_(stack=s).cd(0.25) for s in (rig, rig.film_mask(), cal.stack))
    assert abs(cd_bl - cd_rig) < 0.3*abs(cd_film - cd_rig)
