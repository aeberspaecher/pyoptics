#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the ray traced grating surface and the spectrometer builders against
the grating equation, Shafer's coma condition and the dispersion formulas.
"""

import warnings

import numpy as np
import pytest

from pyoptics import raytrace as rt
from pyoptics import spectrometers as sp
from pyoptics import ultrafast


WL = 550e-6  # mm


def test_transmission_grating_equation():
    g = rt.GratingSurface(period=1/600, order=1)
    for theta in (0.0, 0.1, -0.2):
        d = np.array([[0.0, np.sin(theta), np.cos(theta)]])
        d2 = g.deflect(np.zeros((1, 3)), d, 1.0, 1.0, WL)[0][0]
        assert np.isclose(d2[1], np.sin(theta) + WL*600)
        assert np.isclose(np.linalg.norm(d2), 1.0)
        assert d2[2] > 0
    # against ultrafast.grating_angle (normal incidence, nm and nm):
    d2 = g.deflect(np.zeros((1, 3)), np.array([[0, 0, 1.0]]), 1.0, 1.0, WL)[0][0]
    assert np.isclose(np.arcsin(d2[1]), ultrafast.grating_angle(1e6/600, 1, 550.0))


def test_grating_in_glass_and_evanescent_order():
    g = rt.GratingSurface(period=1/600, order=2)
    d2 = g.deflect(np.zeros((1, 3)), np.array([[0, 0, 1.0]]), 1.0, 1.5, WL)[0][0]
    assert np.isclose(1.5*d2[1], 2*WL*600)
    g3 = rt.GratingSurface(period=1e-3, order=2)  # 2 lambda/d > 1
    d2 = g3.deflect(np.zeros((1, 3)), np.array([[0, 0, 1.0]]), 1.0, 1.0, WL)[0][0]
    assert np.all(np.isnan(d2))


def test_reflection_littrow_and_specular():
    g = rt.GratingSurface(period=1/1200, order=1, mirror=True)
    a = np.arcsin(WL*1200/2)
    d = np.array([[0.0, np.sin(a), np.cos(a)]])
    assert np.allclose(g.deflect(np.zeros((1, 3)), d, 1.0, 1.0, WL)[0], -d)
    d0 = g.with_order(0).deflect(np.zeros((1, 3)), d, 1.0, 1.0, WL)[0][0]
    assert np.allclose(d0, [0, np.sin(a), -np.cos(a)])
    # classical equation m lambda = d (sin alpha + sin beta), any alpha:
    alpha = 0.6
    d = np.array([[0.0, np.sin(alpha), np.cos(alpha)]])
    d2 = g.deflect(np.zeros((1, 3)), d, 1.0, 1.0, WL)[0][0]
    sin_beta = -d2[1]
    assert np.isclose(np.sin(alpha) + sin_beta, WL*1200)


def test_conical_diffraction():
    """Grooves along x: the x component of the direction is kept."""
    g = rt.GratingSurface(period=1/300, order=-1)
    d = np.array([0.2, 0.1, 0.0])
    d[2] = np.sqrt(1 - d @ d)
    d2 = g.deflect(np.zeros((1, 3)), d[None], 1.0, 1.0, WL)[0][0]
    assert np.isclose(d2[0], 0.2)
    assert np.isclose(d2[1], 0.1 - WL*300)
    # grooves rotated by 90 degrees: the grating vector is -x
    g90 = rt.GratingSurface(period=1/300, order=1, groove_angle=np.pi/2)
    d2 = g90.deflect(np.zeros((1, 3)), d[None], 1.0, 1.0, WL)[0][0]
    assert np.isclose(d2[0], 0.2 - WL*300)
    assert np.isclose(d2[1], 0.1)


def test_grating_phase_keeps_the_wavefront():
    """Collimated light, a transmission grating and an ideal lens: the
    diffracted wave focuses without aberration because the optical path
    includes the grating phase."""
    beta = np.arcsin(WL*600)
    surfaces = [rt.GratingSurface(period=1/600, order=1, thickness=5.0, semi_diameter=10.0),
                rt.CoordinateBreak(tilt=(-beta, 0, 0)),
                rt.IdealLens(100.0, thickness=100.0)]
    system = rt.System(surfaces, 0, WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert system.rms_opd(0.0) < 1e-6
    b = system.trace_pupil(0.0, np.array([0.0, 0.7]), np.array([0.5, -0.3]))
    P = system.to_image_frame(b.points[-1])
    assert np.allclose(P[:, :2], 0.0, atol=1e-9)


def test_czerny_turner_dispersion_and_coma():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ct = sp.czerny_turner(wavelength=WL, f_number=8)
        assert np.isclose(ct.alpha - ct.beta, np.radians(30.0))
        s = ct.system
        y = []
        for wl in (WL - 0.5e-6, WL + 0.5e-6):
            b = s.with_wavelength(wl).trace_pupil((0.0, 0.0), 0.0, 0.0)
            y.append(s.to_image_frame(b.points[-1])[0, 1])
        # the tilted camera mirror has the tangential focal length f cos(phi_2):
        assert np.isclose((y[1] - y[0])/1.0, ct.dispersion*np.cos(ct.phi_2), rtol=2e-4)

        def coma(system):
            py = np.linspace(-1, 1, 21)
            b = system.trace_pupil((0.0, 0.0), 0*py, py)
            return np.polyfit(py, system.to_image_frame(b.points[-1])[:, 1], 2)[0]

        c_shafer = abs(coma(s))
        c_equal = abs(coma(sp.czerny_turner(wavelength=WL, f_number=8, phi_2=ct.phi_1).system))
        c_crossed = abs(coma(sp.czerny_turner(wavelength=WL, f_number=8, crossed=True).system))
    assert c_shafer < 0.05*c_equal
    assert c_shafer < 0.05*c_crossed


def test_grating_rotation():
    a, b = sp.grating_rotation(WL, 1200, 1, np.radians(20))
    assert np.isclose(np.sin(a) + np.sin(b), WL*1200)
    assert np.isclose(a - b, np.radians(20))
    with pytest.raises(ValueError):
        sp.grating_rotation(WL, 4000, 1)
