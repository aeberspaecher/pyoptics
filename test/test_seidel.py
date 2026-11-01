#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test powers, Petzval sum, third order image surfaces, distortion and
primary colour of raytrace.System against closed forms and real or paraxial
traces (Welford, "Aberrations of Optical Systems")."""

import numpy as np
import pytest

from pyoptics import abcd
from pyoptics import raytrace as rt


WL = 0.55e-3  # mm
BK7 = (1.5168, 64.17)


def stop_in_front(field=np.radians(5.0), R1=50.0, R2=-80.0, d=5.0):
    """Singlet behind a stop 10 mm in front of it, at its paraxial focus."""

    surfaces = [rt.Surface(np.inf, 10.0, semi_diameter=2.0),
                rt.Surface(R1, d, material=BK7), rt.Surface(R2, 60.0, n_after=1.0)]

    return rt.System(surfaces, 0, WL, field_angle=field).at_paraxial_focus()


def test_powers():
    system = stop_in_front()
    n = system.n[2]
    M = abcd.thick_lens(50.0, -80.0, 5.0, n)
    phi = -M[1, 0]
    assert system.power() == pytest.approx(phi)
    assert system.power() == pytest.approx(1/system.paraxial().efl)
    assert np.allclose(system.surface_powers(), [0.0, (n - 1)/50.0, (1 - n)/-80.0])
    assert system.power_contributions().sum() == pytest.approx(phi)
    assert system.elements() == [(1, 2)]
    assert system.element_powers() == pytest.approx([phi])


def test_elements_with_cemented_group_mirror_and_air_spaces():
    surfaces = [rt.Surface(40.0, 5.0, material=BK7, semi_diameter=5.0),
                rt.Surface(-30.0, 2.0, material=(1.62, 36.4)), rt.Surface(-100.0, 10.0, n_after=1.0),
                rt.Surface(80.0, 3.0, material=BK7), rt.Surface(np.inf, 20.0, n_after=1.0),
                rt.Surface(-200.0, -50.0, mirror=True)]
    system = rt.System(surfaces, 0, WL)
    assert system.elements() == [(0, 2), (3, 4), (5, 5)]
    phi = system.element_powers()
    # an air spaced system: the elements' powers combine like thin lenses
    # only to first order, but the surface contributions always add up:
    assert system.power_contributions().sum() == pytest.approx(system.power())
    assert phi[2] == pytest.approx(2/200.0)  # mirror: -2 n c
    assert phi[1] == pytest.approx(-abcd.thick_lens(80.0, np.inf, 3.0, system.n[4])[1, 0])


def test_petzval_sum_of_thin_lenses_and_mirror():
    # a thin singlet in air: P = phi/n, for any bending and stop position
    for R1, R2 in ((50.0, -80.0), (20.0, 35.0)):
        system = stop_in_front(R1=R1, R2=R2, d=1e-6)
        n = system.n[2]
        t = system.third_order()
        assert t.petzval_sum == pytest.approx(system.power()/n, rel=1e-6)
        assert t.petzval_radius == pytest.approx(-1/t.petzval_sum)
        # S_IV = H**2 P:
        assert system.seidel().total[3] == pytest.approx(system.paraxial().lagrange**2*t.petzval_sum)
    # a concave mirror: the Petzval surface has half its radius and bends
    # the same way (concentric with the mirror for the stop in its centre)
    R = -200.0
    system = rt.System([rt.Surface(np.inf, -R, semi_diameter=5.0),
                        rt.Surface(R, R/2, mirror=True)], 0, WL, field_angle=0.05)
    assert system.third_order().petzval_radius == pytest.approx(R/2)
    assert system.third_order().sagittal_radius == pytest.approx(R/2)  # no astigmatism


def _focus_shift(system, field, axis, dp=1e-3):
    """z of the crossing of two real rays at pupil -+dp along axis ("x":
    sagittal, "y": tangential) relative to the image plane."""

    k = 0 if axis == "x" else 1
    P, D = [], []
    for p in (-dp, dp):
        bundle = system.trace(system.rays(field, px=p if k == 0 else 0.0,
                                          py=p if k == 1 else 0.0))
        P.append(bundle.points[-1, 0])
        D.append(bundle.directions[-1, 0])
    P, D = np.array(P), np.array(D)
    slope = D[:, k]/D[:, 2]

    return -(P[0, k] - P[1, k])/(slope[0] - slope[1])


def test_field_curvature_against_real_ray_foci():
    system = stop_in_front()
    t = system.third_order()
    # all bend towards the lens, the tangential surface most:
    assert 0 > t.tangential_radius > t.medial_radius > t.sagittal_radius > t.petzval_radius
    # the tangential surface lies three times as far from the Petzval surface:
    c_p, c_s, c_t = 1/t.petzval_radius, 1/t.sagittal_radius, 1/t.tangential_radius
    assert c_t - c_p == pytest.approx(3*(c_s - c_p))
    p = system.paraxial()
    z0 = _focus_shift(system, 1e-7, "y")
    for fraction in (0.3, 0.6):
        field = system.max_field*fraction
        h = p.image_height*fraction
        assert _focus_shift(system, field, "x") - z0 == pytest.approx(h**2/(2*t.sagittal_radius),
                                                                      rel=0.02)
        assert _focus_shift(system, field, "y") - z0 == pytest.approx(h**2/(2*t.tangential_radius),
                                                                      rel=0.02)


def test_distortion_against_real_chief_ray():
    for fraction in (0.2, 1.0):
        system = stop_in_front()
        field = system.max_field*fraction
        p = system.paraxial(field)
        eps = system.trace_pupil(field).points[-1, 0, 1] - p.image_height
        assert system.third_order(field).distortion == pytest.approx(eps/p.image_height, rel=0.01)
    # stop in front of a positive lens: barrel distortion
    assert system.third_order().distortion < 0


def test_seidel_per_surface_waves_and_aspheric_part():
    system = stop_in_front()
    system.surfaces[1].conic = -0.7
    system.surfaces[2].aspheric = (2e-6,)
    sd = system.seidel()
    assert np.allclose(sd.W.sum(axis=0), [sd.W040, sd.W131, sd.W222, sd.W220, sd.W311])
    spheres = system.copy()
    spheres.surfaces[1].conic = 0.0
    spheres.surfaces[2].aspheric = ()
    assert np.allclose(sd.S - sd.aspheric, spheres.seidel().S)
    assert np.all(sd.aspheric[:, 3] == 0)  # aspheres add no Petzval curvature
    assert np.all(sd.aspheric[0] == 0)


def test_needs_a_field():
    with pytest.raises(ValueError):
        stop_in_front(field=0.0).third_order()


def test_chromatic_seidel_against_paraxial_traces():
    dl = 0.1e-3
    system = stop_in_front()
    c = system.chromatic_seidel(dl)
    short, long_ = system.with_wavelength(WL - dl/2), system.with_wavelength(WL + dl/2)
    dz = short.paraxial().image_distance - long_.paraxial().image_distance
    assert c.axial == pytest.approx(dz, rel=0.01)
    assert c.axial < 0 and c.total[0] > 0  # undercorrected positive lens

    def height(s):
        (_, _), (y, u) = s._paraxial_rays()
        ys, us = s._paraxial_trace(y, u)
        return ys[-1] + us[-1]*s.surfaces[-1].thickness

    assert c.lateral == pytest.approx(height(short) - height(long_), rel=0.01)
    # a mirror has no colour:
    m = rt.System([rt.Surface(np.inf, 100.0, semi_diameter=5.0),
                   rt.Surface(-200.0, -100.0, mirror=True)], 0, WL, field_angle=0.05)
    assert np.allclose(m.chromatic_seidel(dl).total, 0.0)


def test_caldwell_lens_axial_colour_and_balanced_sums():
    lens = rt.caldwell_krf_lens()
    dl = 1e-9  # 1 pm in mm
    c = lens.chromatic_seidel(dl)
    short, long_ = lens.with_wavelength(lens.wavelength - dl/2), lens.with_wavelength(
        lens.wavelength + dl/2)
    dz = short.paraxial().image_distance - long_.paraxial().image_distance
    assert c.axial == pytest.approx(dz, rel=1e-3)
    assert -0.3e-3 < c.axial < -0.2e-3  # about 0.25 um per pm
    # the monochromatic Seidel sums are small balances of large terms:
    sd = lens.seidel()
    assert np.all(np.abs(sd.total) < 0.01*np.abs(sd.S).sum(axis=0))
    t = lens.third_order()
    assert abs(t.petzval_radius) > 100*1000  # flat Petzval surface (> 100 m)
    assert abs(t.distortion) < 1e-4
