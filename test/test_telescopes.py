#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.telescopes against the closed form third order
results for conic mirrors (Schroeder, Astronomical Optics, chapter 6)."""

import numpy as np
import pytest

from pyoptics import telescopes as tl

D, F1, FS, B = 1000.0, 3000.0, 8000.0, 300.0


def test_two_mirror_geometry():
    g = tl.two_mirror_geometry(F1, FS, B)
    assert g.m == pytest.approx(8/3)
    # the secondary images the primary focus (p in front of it) into the final focus (q)
    assert 1/g.q - 1/g.p == pytest.approx(2/g.R2)
    assert g.d + B == pytest.approx(g.q)
    for kind in ("rc", "classical"):
        S = tl.two_mirror_telescope(D, F1, FS, B, kind)
        assert S.paraxial().efl == pytest.approx(FS, rel=1e-9)
        assert S.paraxial().image_distance == pytest.approx(g.q, rel=1e-9)
    with pytest.raises(ValueError):
        tl.two_mirror_telescope(D, F1, FS, B, "gregorian")


@pytest.mark.parametrize("kind", ["rc", "classical"])
def test_two_mirror_seidel(kind):
    S = tl.two_mirror_telescope(D, F1, FS, B, kind)
    th = np.radians(0.2)
    sd = S.seidel(th)
    a = tl.two_mirror_aberrations(D, F1, FS, B, th, kind)
    lam = S.wavelength
    assert abs(sd.W040) < 1e-9                  # both are free of spherical aberration
    assert 3*sd.W131*lam/(D/2) == pytest.approx(a.ACC, abs=1e-5*th)
    assert 2*sd.W222*lam/(D/2) == pytest.approx(a.AAS, rel=1e-4)
    to = S.third_order(th)
    assert to.petzval_radius == pytest.approx(a.petzval_radius, rel=1e-9)
    assert to.medial_radius == pytest.approx(a.medial_radius, rel=1e-6)


def test_classical_coma_equals_paraboloid():
    th = np.radians(0.2)
    a = tl.two_mirror_aberrations(D, F1, FS, B, th, "classical")
    p = tl.mirror_aberrations(D, FS, -1.0, th)
    assert a.ACC == pytest.approx(p.ACC)
    K1, K2 = tl.rc_conics(8/3, 0.1)
    assert K1 < -1 and K2 < tl.cassegrain_conic(8/3)


def test_newtonian_coma_and_field_curvature():
    D_, F_ = 200.0, 5.0
    f = D_*F_
    th = np.radians(0.5)
    N = tl.newtonian(D_, F_, field_angle=th)
    a = tl.mirror_aberrations(D_, f, -1.0, th)
    sd = N.seidel(th)
    assert sd.W131*N.wavelength == pytest.approx(a.W131, rel=1e-4)
    assert sd.W222*N.wavelength == pytest.approx(a.W222, rel=1e-4)
    fan = N.ray_fan(th, 201)
    # tangential coma: the rim rays meet f*ACC from the chief ray
    assert (fan.ey[0] + fan.ey[-1])/2 == pytest.approx(f*a.ACC, rel=0.01)
    to = N.third_order(th)
    assert to.medial_radius == pytest.approx(a.medial_radius)
    assert to.tangential_radius == pytest.approx(a.tangential_radius)
    dz, _ = tl.best_focus(N, th, search=0.5, n=41, iterations=3)
    h = f*np.tan(th)
    assert dz == pytest.approx(h**2/(2*a.medial_radius), rel=0.02)
    # spherical mirror: W040 = -y^4/(32 f^3)
    sph = tl.newtonian(D_, F_, conic=0.0)
    assert sph.seidel().W040*sph.wavelength == pytest.approx(tl.mirror_aberrations(D_, f, 0.0, th).W040,
                                                            rel=1e-6)


def test_cassegrain_best_focus():
    S = tl.two_mirror_telescope(D, F1, FS, B, "classical")
    th = np.radians(0.2)
    a = tl.two_mirror_aberrations(D, F1, FS, B, th, "classical")
    dz, rms = tl.best_focus(S, th, search=1.0, n=41, iterations=3)
    h = FS*np.tan(th)
    assert dz == pytest.approx(h**2/(2*a.medial_radius), rel=0.01)
    rc = tl.two_mirror_telescope(D, F1, FS, B, "rc")
    _, rms_rc = tl.best_focus(rc, th, search=1.0, n=41, iterations=3)
    assert rms_rc < rms/2


def test_spot_obscuration():
    S = tl.two_mirror_telescope(D, F1, FS, B, "rc")
    s = tl.spot(S, 0.0, obscuration=0.3, n_rings=6)
    r = np.hypot(s.px, s.py)
    assert r.min() > 0.3 and r.max() < 1
    # equal pupil area per ray: half the rays inside the radius sqrt((1 + e^2)/2)
    assert np.mean(r < np.sqrt((1 + 0.3**2)/2)) == pytest.approx(0.5, abs=0.05)
    # pure astigmatism at the medial focus: a uniform disk of diameter AAS f
    th = np.radians(0.1)
    a = tl.two_mirror_aberrations(D, F1, FS, B, th, "rc")
    _, rms = tl.best_focus(S, th, search=1.0, n=41, iterations=3)
    assert rms == pytest.approx(abs(a.AAS)*FS/(2*np.sqrt(2)), rel=0.03)


def test_schmidt():
    D_, F_ = 300.0, 2.0
    S = tl.schmidt_camera(D_, F_, field_angle=np.radians(2.0))
    f = S.paraxial().efl
    assert f == pytest.approx(D_*F_, rel=0.01)
    assert S.rms_opd(0.0) < 0.01
    # focal surface: a sphere of radius f concentric with the mirror
    th = np.radians(2.0)
    dz, rms = tl.best_focus(S, th, search=0.6, n=41, iterations=3)
    assert dz == pytest.approx(-f*(1 - np.cos(th)), rel=0.03)
    assert rms < 2e-3
    # chromatic blur
    n = lambda S_: S_.n[1]
    lam = 486.1e-6
    Sb = S.with_wavelength(lam)
    s = tl.spot(Sb, 0.0, n_rings=24)
    diam = 2*np.max(np.hypot(s.x - s.centroid[0], s.y - s.centroid[1]))/f
    pred = tl.schmidt_chromatic_blur(F_, n(S), n(Sb) - n(S))
    assert diam == pytest.approx(pred, rel=0.05)
    R_p, A4 = tl.schmidt_plate(D_, F_, 1.5, k=0)
    assert np.isinf(R_p) and A4 < 0
