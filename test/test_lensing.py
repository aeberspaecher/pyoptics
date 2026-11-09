#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for pyoptics.lensing: light bending in a GRIN medium, thin lens
images, magnifications and caustics against closed forms."""

import numpy as np
import pytest

from pyoptics import grin
from pyoptics import lensing as ln


def test_physical_scales():
    # deflection at the solar limb: 1.75 arcsec
    R_sun = 6.957e8
    a = ln.deflection_weak(R_sun, ln.schwarzschild_radius(ln.M_SUN))
    assert np.degrees(a)*3600 == pytest.approx(1.75, abs=0.01)
    # a solar mass at 4 kpc lensing a star at 8 kpc: about a milliarcsecond
    tE = ln.einstein_radius(ln.M_SUN, 4*ln.KPC, 8*ln.KPC)
    assert np.degrees(tE)*3.6e6 == pytest.approx(1.0, rel=0.05)


def test_grin_deflection():
    Lz = 5e3
    b = np.array([60.0, 200.0])
    P = np.c_[b, -Lz*np.ones_like(b)]
    for medium, ref in ((ln.weak_field_medium(), ln.deflection_weak),
                        (ln.schwarzschild_medium(), ln.schwarzschild_deflection)):
        r = grin.trace(medium, P, [0, 1], step=0.5, stop=lambda r: Lz - r[:, 1], record=False,
                       max_steps=100000)
        d = r.direction
        a = np.arctan2(-d[:, 0], d[:, 1])
        finite = Lz/np.hypot(Lz, b)              # first order share of a finite path
        assert np.allclose(a, ref(b)*finite, rtol=0.02)
    # the exact medium agrees with the exact integral to second order corrections
    assert ln.schwarzschild_deflection(1000.0) == pytest.approx(ln.deflection_second_order(1000.0), rel=1e-5)
    assert np.isnan(ln.schwarzschild_deflection(2.5))


def test_point_mass_images():
    for u in (0.1, 0.5, 1.5):
        th, mu = ln.find_images(ln.PointMass(), (u, 0.0))
        tp, tm, mp, mm = ln.point_mass_images(u)
        assert np.allclose(np.sort(th[:, 0]), [tm, tp], atol=1e-9)
        assert np.allclose(np.sort(np.abs(mu)), np.sort([mp, mm]), rtol=1e-5)
        assert np.sum(np.abs(mu)) == pytest.approx(ln.paczynski(u), rel=1e-5)
        assert mp - mm == pytest.approx(1.0)


def test_sis_images_and_delay():
    S = ln.SIS(1.0)
    th, mu = ln.find_images(S, (0.3, 0.0))
    t_ana, mu_ana = ln.sis_images(0.3)
    assert np.allclose(np.sort(th[:, 0]), np.sort(t_ana), atol=1e-9)
    assert np.sum(np.abs(mu)) == pytest.approx(2/0.3, rel=1e-5)
    tau = ln.fermat_potential(S, th, (0.3, 0.0))
    assert abs(tau[0] - tau[1]) == pytest.approx(2*0.3, rel=1e-9)
    assert len(ln.find_images(S, (1.3, 0.0))[0]) == 1


def test_finite_source():
    x = np.linspace(-1.6, 1.6, 801)
    rho = 0.2
    disk = lambda bx, by: (bx**2 + by**2 < rho**2).astype(float)
    I = ln.ray_shoot(ln.PointMass(), disk, x, x)
    mu = I.sum()*(x[1] - x[0])**2/(np.pi*rho**2)
    assert mu == pytest.approx(ln.finite_source_magnification(rho), rel=0.01)


def test_shear_caustic():
    g = 0.15
    lens = ln.Composite([ln.SIS(), ln.ExternalShear(g)])
    cc = ln.critical_curve(lens, n_angles=361)
    ca = ln.caustic(lens, cc)[0]
    cx, cy = ln.sis_shear_cusps(g)
    assert np.nanmax(np.abs(ca[:, 0])) == pytest.approx(cx, rel=1e-3)
    assert np.nanmax(np.abs(ca[:, 1])) == pytest.approx(cy, rel=1e-3)
    assert len(ln.find_images(lens, (0.05, 0.03))[0]) == 4
    assert len(ln.find_images(lens, (0.6, 0.0))[0]) == 2
