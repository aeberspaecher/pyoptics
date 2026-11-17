#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of pyoptics.litho.euv, the block treatment of plane layers in
litho.rcwa, the secondary electron blur of litho.stochastics and the Jones
pupil of coated mirror systems, as used in notebooks
156 (EUV multilayers and reflective masks) to 158 (EUV stochastics and flare).
"""

import numpy as np
import pytest

from pyoptics.litho import (Illumination, LithoSimulation, MaskLayout, Projector, ThinMask, euv, lens,
                            metrics, stochastics as st)
from pyoptics.litho.rcwa import Layer, rcwa
from pyoptics.thin_films import _r_t


DEG = np.pi/180


def test_multilayer_matches_thin_films():
    ml = euv.Multilayer(pairs=10, period=6.9)
    n = np.array([1.0] + [n for n, _ in ml.layers()] + [ml.substrate_index])
    t = [t for _, t in ml.layers()]
    r_s, r_p = ml.amplitudes(0.1)
    assert r_s == pytest.approx(_r_t(np.sin(0.1), n, t, "TE", 13.5)[0])
    assert r_p == pytest.approx(_r_t(np.sin(0.1), n, t, "TM", 13.5)[0])


def test_multilayer_design_and_numbers():
    ml = euv.Multilayer.design(13.5, 6*DEG)
    assert 6.85 < ml.period < 7.05
    R = ml.reflectance(6*DEG)
    assert 0.70 < R < 0.76
    # the peak is at the design wavelength and angle
    assert ml.reflectance(6*DEG, 13.4) < R and ml.reflectance(6*DEG, 13.6) < R
    assert 0.5 < ml.bandwidth() < 0.9
    assert 35 < ml.effective_depth() < 60
    # interlayers and a missing cap change the reflectance a little
    assert ml.with_(interlayer=0.6).reflectance(6*DEG) < R
    assert len(ml.with_(capping=None).layers()) == 80
    with pytest.raises(ValueError):
        euv.Multilayer(interlayer=3.0).layers()


def test_multilayer_jones_axes():
    ml = euv.Multilayer.design()
    r_s, r_p = ml.amplitudes(np.arcsin(0.1))
    J = ml.jones(0.0, 0.1)         # plane of incidence y-z: s along x
    assert J[0, 0] == pytest.approx(r_s)
    assert J[1, 1] == pytest.approx(r_p)
    assert abs(J[0, 1]) < 1e-12
    J = ml.jones(0.1, 0.0)
    assert J[0, 0] == pytest.approx(r_p) and J[1, 1] == pytest.approx(r_s)


def test_rcwa_plane_layers_in_blocks():
    """Plane layers under a grating, done order by order, agree with the
    full matrices (forced by segments of the background index)."""
    Mo, Si = euv.INDEX_13_5["Mo"], euv.INDEX_13_5["Si"]
    blank = [Layer(4.1, Si), Layer(2.8, Mo)]*6
    top = [Layer(50.0, 1.0, [(0.0, 40.0, euv.INDEX_13_5["TaBN"])])]
    fake = [Layer(l.thickness, l.background, [(0.0, 10.0, l.background)]) for l in blank]
    kw = dict(k_x=[0.01, 0.03], k_y=0.1, n_out=euv.INDEX_13_5["SiO2"], n_orders=6)
    a = rcwa(100.0, top + blank, 13.5, **kw)
    b = rcwa(100.0, top + fake, 13.5, **kw)
    assert np.allclose(a.r, b.r, atol=1e-12)
    assert np.allclose(a.t, b.t, atol=1e-12)
    # only plane layers: the 0th order is the thin film result
    c = rcwa(100.0, blank, 13.5, k_x=0.0, k_y=0.1, n_out=euv.INDEX_13_5["SiO2"], n_orders=2)
    n = np.array([1.0] + [l.background for l in blank] + [euv.INDEX_13_5["SiO2"]])
    t = [l.thickness for l in blank]
    assert c.r[0, 2, 0, 0] == pytest.approx(_r_t(0.1, n, t, "TE", 13.5)[0])
    assert c.r[0, 2, 1, 1] == pytest.approx(_r_t(0.1, n, t, "TM", 13.5)[0])


@pytest.fixture(scope="module")
def ml():
    return euv.Multilayer.design()


def _sim(mask, orientation, pitch=32.0, M=4, NA=0.33):
    axis = "x" if orientation == "y" else "y"
    return LithoSimulation(MaskLayout.lines_spaces(pitch, pitch/2, orientation=orientation), mask,
                           Illumination.dipole(0.9, 0.55, axis=axis, n_samples=11),
                           Projector(NA, 13.5, magnification=M))


def test_reflective_mask_films(ml):
    mask = euv.ReflectiveMask([("TaBN", 60.0)], ml)
    R_c, R_a = mask.film_reflectance()
    assert R_c == pytest.approx(ml.reflectance(6*DEG), rel=1e-6)
    assert 0.005 < R_a < 0.05
    J_c, J_a = mask.jones(np.zeros(1), np.zeros(1), 0.33, 13.5, 4)
    assert np.allclose(J_c[0], np.eye(2), atol=0.02)        # normalized at the chief ray
    t = mask.kirchhoff().t_absorber
    assert abs(t)**2 == pytest.approx(R_a/R_c, rel=0.05)


def test_reflective_mask_open_frame_and_orders(ml):
    """An absorber of vacuum gives only the zeroth order, equal to the clear
    blank."""
    mask = euv.ReflectiveMask([("vacuum", 60.0)], ml)
    layout = MaskLayout.lines_spaces(32.0, 16.0)
    m = np.arange(-3, 4)
    s = np.array([0.0, 0.5])
    J = mask.order_jones(layout, m, [0], s, s, 0.33, 13.5, 4)
    J_c = mask.jones(s, s, 0.33, 13.5, 4)[0]
    assert np.allclose(J[:, :, :, 0, 3], J_c, atol=1e-9)
    assert np.allclose(np.delete(J[:, :, :, 0, :], 3, axis=-1), 0.0, atol=1e-9)


def test_shadowing_and_symmetry(ml):
    mask = euv.ReflectiveMask([("TaBN", 60.0)], ml)
    v, h = _sim(mask, "y"), _sim(mask, "x")
    th = 0.3
    u, c = v.cut_coordinates(201)[:2]
    I_v = v.cut(201)[1]
    u_h, c_h = h.cut_coordinates(201)[:2]
    I_h = h.cut(201)[1]
    # vertical lines: symmetric, no shift; horizontal lines: shifted in y
    assert abs(euv.feature_center(u, I_v, th, c) - c) < 0.05
    shift = euv.feature_center(u_h, I_h, th, c_h) - c_h
    assert 1.5 < shift < 4.0
    # the effective mirror plane explains most of the shift
    geo = 2*ml.effective_depth()*np.tan(6*DEG)/4
    assert abs(shift - geo) < 1.0
    # horizontal lines lose contrast
    assert np.ptp(I_h) < np.ptp(I_v)
    # a thin mask knows no orientation
    k = mask.kirchhoff()
    assert np.allclose(_sim(k, "y").cut(101)[1], _sim(k, "x").cut(101)[1], atol=1e-9)


def test_anamorphic_reduces_mask_angles(ml):
    """At NA 0.55 the shift of horizontal lines is smaller with 8x in y at
    5.355 degrees than with 4x at 9 degrees."""
    shifts = []
    for M, cra in ((4, 9.0), ((4, 8), 5.355)):
        mask = euv.ReflectiveMask([("TaBN", 60.0)], euv.Multilayer.design(angle=cra*DEG), cra=cra)
        sim = _sim(mask, "x", pitch=24.0, M=M, NA=0.55)
        u, c = sim.cut_coordinates(201)[:2]
        shifts.append(euv.feature_center(u, sim.cut(201)[1], 0.3, c) - c)
    assert shifts[1] < shifts[0]
    # the mask pitch along y is 8 times the wafer pitch: orders and incidence
    mask = euv.ReflectiveMask([("TaBN", 60.0)], ml, cra=5.355)
    k_x, k_y = mask.incidence(1.0, 1.0, 0.55, (4, 8))
    assert k_x == pytest.approx(0.55/4)
    assert k_y == pytest.approx(np.sin(5.355*DEG) + 0.55/8)


def test_geometry_helpers():
    assert euv.min_cra(0.33, 4) == pytest.approx(np.degrees(np.arcsin(0.33/4)))
    assert euv.min_cra(0.55, 4) > 6.0 > euv.min_cra(0.55, 8)
    img, ill = euv.mask_cones(0.55, (4, 8), 5.355, n=9)
    assert np.allclose(img.mean(axis=0)[1], np.sin(5.355*DEG), atol=0.02)
    assert np.ptp(img[:, 0]) == pytest.approx(2*0.55/4, rel=0.02)
    assert np.ptp(img[:, 1]) == pytest.approx(2*0.55/8, rel=0.05)
    A = euv.obscured_pupil(0.2)
    assert A(0.1, 0.0) == 0.0 and A(0.5, 0.0) == 1.0
    assert euv.shadow_width(60.0) == pytest.approx(120*np.tan(6*DEG))
    assert euv.z_factor(16.0, 4.0, 40.0) == pytest.approx(16**3*16*40*1e-14)
    assert euv.fwhm([0, 1, 2, 3, 4], [0, 1, 2, 1, 0]) == pytest.approx(2.0)


def test_obscured_pupil_forbidden_pitch():
    """The first orders of a dipole falling into the obscuration take the
    contrast."""
    def contrast(obs, pitch):
        P = Projector(0.55, 13.5, apodization=euv.obscured_pupil(obs) if obs else None)
        sim = LithoSimulation(MaskLayout.lines_spaces(pitch, pitch/2), ThinMask.binary(),
                              Illumination.dipole(0.9, 0.6, n_samples=15), P)
        return np.ptp(sim.cut(101)[1])
    assert contrast(0.2, 33.0) < 0.6*contrast(0.0, 33.0)
    assert contrast(0.2, 20.0) == pytest.approx(contrast(0.0, 20.0), rel=0.05)


def test_electron_blur_conserves_acids():
    R = st.StochasticResist(wavelength=13.5, thickness=35.0, absorption=5e-3, pag=0.2)
    I = np.tile(0.5 + 0.4*np.cos(2*np.pi*np.arange(64)/64), (128, 1))
    rng = np.random.default_rng(3)
    counts = rng.poisson(2.0, (128, 64)).astype(float)
    moved = st._scatter(counts, 2.0, 0.5, 0.5, np.random.default_rng(1))
    assert moved.sum() == counts.sum()
    assert np.std(moved) < np.std(counts)
    a = R.with_(electron_blur=2.0).process(I, 40.0, 0.5, rng=5)
    b = R.process(I, 40.0, 0.5, rng=5)
    assert a.acid_0.sum() == b.acid_0.sum()
    # the mean chain blurs instead
    m = R.with_(electron_blur=2.0).mean(I, 40.0, 0.5)
    assert np.ptp(m.acid_0) < np.ptp(R.mean(I, 40.0, 0.5).acid_0)


@pytest.fixture(scope="module")
def schwarzschild():
    return euv.schwarzschild_objective()


def test_schwarzschild_objective(schwarzschild):
    S = schwarzschild
    par = S.paraxial()
    assert par.na == pytest.approx(0.3, rel=1e-3)
    assert par.magnification == pytest.approx(-0.2, rel=1e-3)
    for f in ((0.0, 0.0), (0.0, S.max_field)):
        assert np.nanstd(S.opd(f, N=32)[2]) < 0.07          # diffraction limited
    assert 0.2 < euv.obscuration(S) < 0.5
    r, a = euv.angles_of_incidence(S, 0)
    assert np.degrees(np.nanmax(a)) > 10


def test_mirror_jones_pupil_graded(schwarzschild):
    """lens_jones_pupil() traces coated mirrors; graded multilayers flatten
    the transmission."""
    S = schwarzschild
    out = {}
    for graded in (False, True):
        T = S.copy()
        for i in (0, 1):
            r, a = euv.angles_of_incidence(S, i)
            if graded:
                mli = euv.Multilayer.design(angle=float(a[0]))
                T.surfaces[i].coating = mli.coating(euv.graded_profile(mli, r, a, angle=float(a[0])))
            else:
                T.surfaces[i].coating = euv.Multilayer.design(angle=float(np.mean(a))).coating()
        L = lens.lens_jones_pupil(T, (0.0, 0.0), N=21, normalize=False)
        keep = np.hypot(L.P_x, L.P_y) > 0.4
        tr = np.sum(np.abs(L.J[keep])**2, axis=(1, 2))/2
        out[graded] = (tr.min(), tr.max())
    assert out[False][1] < 0.6 and out[True][1] < 0.6
    assert out[True][1] - out[True][0] < 0.3*(out[False][1] - out[False][0])
