#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the zoom lens tools (notebook 127, zoom lenses)."""

import numpy as np
import pytest

from pyoptics import zoom


def test_thin_image_and_conjugates():
    # object at infinity: image in the focus
    assert np.isclose(zoom.thin_image(-np.inf, 10.0, 50.0), 60.0)
    # 2f-2f
    assert np.isclose(zoom.thin_image(-100.0, 0.0, 50.0), 100.0)


def test_compensator_branches_touch_at_unit_magnification():
    f = 20.0
    z1, z2 = zoom.compensator_positions(0.0, 100.0, f)
    for z in (z1, z2):
        assert np.isclose(zoom.thin_image(0.0, z, f), 100.0)
    # D = 4f: one solution in the middle, m = -1
    a, b = zoom.compensator_positions(0.0, 80.0, f)
    assert np.isclose(a, 40.0) and np.isclose(b, 40.0)
    # D < 4f: none
    assert np.all(np.isnan(zoom.compensator_positions(0.0, 70.0, f)))


def test_two_group_zoom_focal_length_and_track():
    f1, f2 = -40.0, 30.0
    efl = np.array([20.0, 30.0, 45.0])
    d, back = zoom.two_group_zoom(f1, f2, efl)
    for fi, di, bi in zip(efl, d, back):
        f, bf = zoom.thin_system_efl([f1, f2], [0.0, di])
        assert np.isclose(f, fi)
        assert np.isclose(bf - di, bi)   # rear focus behind the second group
    # the track is shortest at f = |f1|, where it is f1 + 4 f2 (in magnitude terms)
    fs = np.linspace(15, 60, 451)
    d, b = zoom.two_group_zoom(f1, f2, fs)
    L = d + b
    assert np.isclose(fs[np.argmin(L)], abs(f1), atol=0.2)
    assert np.isclose(L.min(), f1 + 4*f2, atol=1e-3)


def test_four_group_afocal_zoom_keeps_image_fixed():
    f1, f2, f3, f4 = 60.0, -15.0, 30.0, 40.0
    z2 = np.linspace(10.0, 40.0, 7)
    z3, efl, zi = zoom.four_group_zoom(f1, f2, f3, f4, z2)
    assert np.all(np.isfinite(z3))
    assert np.allclose(zi, zi[0])
    # the focal length changes monotonically with the variator
    assert np.all(np.diff(efl) > 0) or np.all(np.diff(efl) < 0)
    # efl of the thin lens chain agrees
    for k in range(len(z2)):
        f, _ = zoom.thin_system_efl([f1, f2, f3, f4], [0.0, z2[k], z3[k], 4*(60 + 15 + 30)])
        assert np.isclose(f, efl[k])


@pytest.fixture(scope="module")
def olympus():
    return zoom.olympus_zoom_us7583450()


def test_olympus_focal_lengths(olympus):
    for k, f in enumerate(olympus.focal_lengths):
        p = olympus.system(k).paraxial()
        assert np.isclose(p.efl, f, rtol=0.01)
        assert abs(p.image_distance - olympus.system(k).surfaces[-1].thickness) < 0.05


def test_olympus_interpolation_and_refocus(olympus):
    f = float(np.sqrt(olympus.focal_lengths[0]*olympus.focal_lengths[-1]))
    s = olympus.system(f)
    s2, dt = zoom.ZoomLens.refocus(s, (11, 13))
    p = s2.paraxial()
    assert abs(p.image_distance - s2.surfaces[-1].thickness) < 1e-3
    # the total track is constant: the focusing lens only moves between its neighbours
    assert np.isclose(sum(t.thickness for t in s.surfaces[:-1]),
                      sum(t.thickness for t in s2.surfaces[:-1]))


def test_olympus_groups(olympus):
    fg = olympus.group_focal_lengths()
    finite = np.asarray(fg)[np.isfinite(fg)]
    # positive front group, negative variator
    assert finite[0] > 0 and finite[1] < 0
    pos = [olympus.group_positions(olympus.system(k)) for k in range(len(olympus))]
    # the front group moves out towards tele
    assert pos[-1][0] != pos[0][0]
